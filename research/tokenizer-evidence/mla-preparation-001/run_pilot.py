"""One frozen 60-step MLA MPS feasibility trial, then cache checks; never quality selection."""
import argparse
import copy
import json
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from src.research.protocol import digest, file_digest
from src.research.runner import atomic_json, session_lock, source_fingerprint
from src.research.tokenizer_study import ROOT, runtime_identity, worker
from src.models.tokenizer_architecture import MLA_POLICY, architecture_contract
RECIPE = ROOT/'research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json'
PROTOCOL = ROOT/'research/TOKENIZER_MLA_PREPARATION.md'


def config(root):
    cfg = copy.deepcopy(json.loads(RECIPE.read_text())['config'])
    cfg['model'].update(attn_type='mla', attention_width_policy='mla_quarter_head32',
                        depth_width_policy='depth4_unscaled', tokenizer_architecture_policy=MLA_POLICY)
    t = cfg['training']
    t.update(seed=2891, max_iters=60, eval_every=60, checkpoint_dir=str(root/'checkpoints'),
             metrics_file=str(root/'metrics.jsonl'), summary_file=str(root/'training-summary.json'))
    t['lr_schedule'].update(total_steps=60, start_step=40)
    t['weight_averaging'].update(total_steps=60, start_step=40, interval=10)
    architecture_contract(cfg['model'])
    return cfg


def verify(root):
    plan = json.loads((root/'plan.json').read_text())
    assert plan['source']==source_fingerprint() and plan['runtime']==runtime_identity()
    for path,key in [(RECIPE,'recipe_sha256'),(PROTOCOL,'protocol_sha256'),(Path(__file__),'helper_sha256'),(Path(plan['request']['config']['data']['path']),'corpus_sha256')]:
        assert file_digest(path)==plan[key]
    assert json.loads((root/'request.json').read_text())==plan['request']
    assert plan['request']['config']==config(root)
    return plan


def cached_check(root):
    import gc
    import torch
    from omegaconf import OmegaConf
    from src.models.gpt import GPT
    from src.attention.cache import KVCache
    from src.utils.checkpoint import read_checkpoint, load_model_weights, checkpoint_loader
    from src.research.tokenizer_attention_bench import check_cached, buffer_bytes
    gc.collect();torch.mps.empty_cache()
    result=json.loads((root/'result.json').read_text())
    c=read_checkpoint(result['h200']['checkpoint']);cfg=OmegaConf.create(c['cfg'])
    loader=checkpoint_loader(cfg,c,cfg.data.path,'cpu')
    model=GPT(cfg.model).to('mps').eval();load_model_weights(model,c)
    rows=[]
    for b in (1,8):
        ids=loader.train_data[:12].unsqueeze(0).repeat(b,1).to('mps')
        error=check_cached(model,cfg,ids)
        cache=[KVCache(cfg,'mps',batch_size=b,dtype=torch.float32) for _ in range(4)]
        size=buffer_bytes(cache);assert size==4*b*128*(64+32)*4
        rows.append(dict(batch=b,max_absolute_error=error,allocated_cache_bytes=size))
        del cache
    atomic_json(root/'cache-check.json',dict(status='passed',records=rows,scope='Feasibility and allocated cache only; no latency or quality adoption evidence.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','run','worker']);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();root=a.output.resolve()
    if a.command=='prepare':
        assert not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()
        assert shutil.disk_usage(ROOT).free>9*1024**3
        root.mkdir(exist_ok=False)
        spec=dict(id=root.name,stage='pilot',family='mla',width=256,seed=2891,lr=.00125,steps=60)
        request=dict(spec=spec,config=config(root),result_path=str(root/'result.json'))
        plan=dict(source=source_fingerprint(),runtime=runtime_identity(),request=request,job_seconds=600,
                  min_free_bytes=8*1024**3,max_output_bytes=1024**3,
                  recipe_sha256=file_digest(RECIPE),protocol_sha256=file_digest(PROTOCOL),helper_sha256=file_digest(Path(__file__)),
                  corpus_sha256=file_digest(Path(request['config']['data']['path'])),
                  source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip())
        atomic_json(root/'plan.json',plan);atomic_json(root/'request.json',request)
        atomic_json(root/'state.json',dict(status='prepared',plan_sha256=digest(plan)))
    elif a.command=='worker':
        verify(root);worker(root/'request.json');cached_check(root);verify(root)
    else:
        plan=verify(root);state=json.loads((root/'state.json').read_text())
        assert state['plan_sha256']==digest(plan)
        if state['status']!='prepared':raise RuntimeError('No retries or duplicate pilot launch')
        lock=ROOT/'research_state/tokenizer-worker-lock';lock.mkdir(exist_ok=True)
        with session_lock(root) as fd,session_lock(lock) as gfd:
            if (root/'STOP').exists():raise RuntimeError('Pilot stopped')
            assert shutil.disk_usage(root).free>plan['min_free_bytes']+plan['max_output_bytes']
            with (root/'worker.log').open('x') as log:
                proc=subprocess.Popen([sys.executable,'-m','research_state.tokenizer-mla-validation.run_pilot','worker','--output',str(root)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd,gfd))
                def stop(signum,frame):
                    (root/'STOP').write_text('Stop requested\n');os.killpg(proc.pid,signal.SIGTERM)
                signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
                state.update(status='running',controller_pid=os.getpid(),worker_pid=proc.pid);atomic_json(root/'state.json',state)
                try:code=proc.wait(timeout=600)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    code=124
            verify(root)
            size=sum(f.stat().st_size for f in root.rglob('*') if f.is_file())
            ok=code==0 and (root/'cache-check.json').exists() and size<=plan['max_output_bytes'] and shutil.disk_usage(root).free>=plan['min_free_bytes']
            state.update(status='completed' if ok else 'failed',exit_code=code,output_bytes=size);atomic_json(root/'state.json',state)
            print(json.dumps(state))


if __name__=='__main__':
    main()
