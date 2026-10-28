"""Paired width512 MLA/GQA inference measurement; quality decisions remain separate."""
import argparse
import gc
import os
import json
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ..models.gpt import GPT
from ..utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
from .memory_guard import start_memory_guard
from .protocol import file_digest
from .runner import atomic_json, session_lock, source_fingerprint
from .tokenizer_attention_bench import timed, check_cached
from ..models.tokenizer_architecture import WIDE_GQA_POLICY, WIDE_MLA_POLICY
KINDS = ('gqa', 'mla')
POLICIES = dict(gqa=WIDE_GQA_POLICY, mla=WIDE_MLA_POLICY)


def benchmark(paths, output):
    if set(paths) != set(KINDS):
        raise ValueError('Both wider attention arms are required')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise RuntimeError('MPS fallback forbidden')
    if not torch.backends.mps.is_available():raise RuntimeError('MPS required')
    torch.set_num_threads(4);start_memory_guard()
    models={};configs={};metadata={};fixed=None
    for family in KINDS:
        path=paths[family]
        c=read_checkpoint(path);cfg=OmegaConf.create(c['cfg'])
        if (cfg.model.tokenizer_architecture_policy != POLICIES[family] or cfg.model.attn_type != family
                or cfg.model.dim != 512 or 'averaged_weights' not in c
                or c['averaged_weights']['snapshots'] != 41 or c['step'] != 1200):
            raise ValueError('Checkpoint attention/width/H200 mismatch')
        if configs and cfg.training.seed != next(iter(configs.values())).training.seed:
            raise ValueError('Different fresh confirmation seeds')
        loader=checkpoint_loader(cfg,c,cfg.data.path,'cpu')
        ids=loader.train_data[:96].clone()
        if fixed is not None and not torch.equal(fixed,ids):raise ValueError('Different timing tokens')
        fixed=ids
        model=GPT(cfg.model).to('mps').eval();load_model_weights(model,c)
        models[family]=model;configs[family]=cfg
        metadata[family]=dict(checkpoint=str(path.resolve()),sha256=file_digest(path),
                              width=cfg.model.dim,parameters=sum(p.numel() for p in model.parameters()),
                              parameter_bytes=sum({p.untyped_storage().data_ptr():p.untyped_storage().nbytes()
                                                   for p in model.parameters()}.values()))
        del c,loader;gc.collect()
    rows=[];correctness=[]
    for batch in (1,8):
        ids=fixed.unsqueeze(0).repeat(batch,1).to('mps')
        for family in KINDS:
            error=check_cached(models[family],configs[family],ids)
            correctness.append(dict(batch=batch,family=family,max_absolute_error=error))
            for _ in range(2):timed(models[family],configs[family],ids)
        for repeat in range(8):
            for family in KINDS[repeat%len(KINDS):]+KINDS[:repeat%len(KINDS)]:
                rows.append(dict(batch=batch,repeat=repeat,family=family,**timed(models[family],configs[family],ids)))
    result=dict(status='completed',device='mps',dtype='float32',metadata=metadata,
                fixed_train_token_ids=fixed.tolist(),prompt_tokens=32,decode_tokens=64,
                warmups=2,repeats=8,correctness=correctness,records=rows,
                scope='Two models resident; model loading/cache allocation outside timing. Memory observations are not single-model deployment peaks. Teacher-forced fixed training text; no test access or quality selection.')
    atomic_json(output,result)
    return result


def run_review(root):
    from .tokenizer_wide_study import SEEDS, specs, checked_panel, RECIPE, PROTOCOL, RESOURCE
    from .tokenizer_study import ROOT, runtime_identity
    from .protocol import digest
    plan = json.loads((root/'plan.json').read_text()); state = json.loads((root/'state.json').read_text())
    if state['status'] != 'completed' or state['plan_sha256'] != digest(plan):
        raise ValueError('Completed frozen quality panel required')
    checked_panel(state['records'], specs(state['selection']))
    def verify():
        if source_fingerprint() != plan['source'] or runtime_identity() != plan['runtime']:
            raise ValueError('Frozen source/runtime changed')
        for path,key in [(RECIPE,'recipe_sha256'),(PROTOCOL,'protocol_sha256'),(RESOURCE,'resource_review_sha256'),
                         (Path(plan['config']['data']['path']),'corpus_sha256')]:
            if file_digest(path) != plan[key]:raise ValueError('Frozen evidence changed')
        if (root/'STOP').exists() or shutil.disk_usage(root).free < 8*1024**3+4*1024**2:
            raise RuntimeError('Stop or timing storage guard')
    verify()
    out = root/'timing'
    with session_lock(root) as fd, session_lock(ROOT/'research_state/tokenizer-worker-lock') as gfd:
        out.mkdir(exist_ok=False) # Any prior attempt requires explicit review, never retry.
        atomic_json(out/'state.json',dict(status='running',completed=[]))
        def stop(signum, frame): (root/'STOP').write_text(f'Timing stop signal {signum}\n')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        completed=[];proc=None
        try:
            for seed in SEEDS:
                verify(); args=[]
                for kind in KINDS:
                    rows=[r for r in state['records'] if r['stage']=='confirm' and r['seed']==seed and r['family']==kind]
                    if len(rows)!=1:raise ValueError('Missing fresh pair')
                    c=rows[0]['h200'];path=Path(c['checkpoint'])
                    if file_digest(path)!=c['checkpoint_sha256']:raise ValueError('Fresh checkpoint changed')
                    args+=['--'+kind.replace('_','-'),str(path)]
                target=out/f'seed-{seed}.json'
                atomic_json(out/f'seed-{seed}-request.json',dict(seed=seed,checkpoint_args=args,quality_plan_sha256=state['plan_sha256']))
                with (out/f'seed-{seed}.log').open('x') as log:
                    proc=subprocess.Popen([sys.executable,'-m','src.research.tokenizer_wide_bench',*args,'--output',str(target)],
                        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,gfd))
                    started=time.monotonic()
                    while proc.poll() is None:
                        verify()
                        if time.monotonic()-started>600 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>4*1024**2:
                            raise RuntimeError('Timing duration/output guard')
                        try:proc.wait(timeout=2)
                        except subprocess.TimeoutExpired:pass
                if proc.returncode or not target.is_file():raise RuntimeError('Timing worker failed')
                result=json.loads(target.read_text())
                if result.get('status')!='completed' or set(result.get('metadata',{}))!=set(KINDS) or len(result.get('records',[]))!=32 or len(result.get('correctness',[]))!=4:
                    raise ValueError('Incomplete timing result')
                for kind in KINDS:
                    ref=next(r['h200'] for r in state['records'] if r['stage']=='confirm' and r['seed']==seed and r['family']==kind)
                    if result['metadata'][kind]['sha256']!=ref['checkpoint_sha256']:
                        raise ValueError('Timing result checkpoint mismatch')
                verify();completed.append(seed)
                atomic_json(out/'state.json',dict(status='running',completed=completed))
            atomic_json(out/'state.json',dict(status='completed',completed=completed,quality_state_sha256=file_digest(root/'state.json')))
        except BaseException as error:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            atomic_json(out/'state.json',dict(status='failed',completed=completed,error=str(error)))
            raise


def main():
    p = argparse.ArgumentParser();p.add_argument('--study-run',type=Path)
    for kind in KINDS: p.add_argument('--'+kind.replace('_','-'), type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.study_run:
        run_review(a.study_run.absolute());return
    if a.output is None or any(getattr(a,k) is None for k in KINDS):raise ValueError('Two paths and output required')
    if a.output.exists(): raise FileExistsError(a.output)
    benchmark({k:getattr(a,k) for k in KINDS},a.output)


if __name__ == '__main__':
    main()
