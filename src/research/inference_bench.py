"""Bounded paired cached-inference diagnostic using immutable saved checkpoints."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from omegaconf import OmegaConf

from .protocol import digest, file_digest
from .runner import atomic_json, disk_mb, session_lock, terminate
from .team.suite import source_digest
from ..attention.cache import KVCache
from ..models.gpt import GPT
from ..train.train import _make_loader, _training_contract, _contracts_match
from ..utils.checkpoint import read_checkpoint, load_model_weights
from ..utils.device import synchronize_device

PANEL = [('architecture-stage2d-001', 'gqa_half', 444, 2),
         ('architecture-stage2f-001', 'mla', 1011, 4)]


def panel_jobs(root):
    """Fixed smallest-seed selected checkpoints; never choose from timings."""
    jobs = []
    for study, candidate, seed, layers in PANEL:
        directory = Path(root)/'research_runs'/study
        report = json.loads((directory/'report.json').read_text())
        for width in [64, 128, 256]:
            models = {}
            for role, family in [('reference', 'mha_tuned'), ('candidate', candidate)]:
                row, = [r for r in report['records'] if r['family'] == family
                        and r['width'] == width and r['seed'] == seed
                        and r['lr'] == report['selection']['choices'][family]['lr']]
                models[role] = {k: row[k] for k in ['family', 'checkpoint', 'checkpoint_sha256']}
            for batch in [1, 8]:
                for length in [32, 64]:
                    jobs.append(dict(id=f'{candidate}_w{width}_b{batch}_p{length}',
                        study=study, width=width, seed=seed, layers=layers, models=models,
                        batch=batch, prompt=length, decode=length, capacity=128,
                        warmups=3, repeats=9, first_role_offset=len(jobs)%2,
                        atol=1e-4, rtol=2e-5))
    return jobs


def cache_bytes(caches):
    return sum(b.numel()*b.element_size() for c in caches for b in c.buffers())


def make_caches(model, cfg, batch):
    return [KVCache(cfg, next(model.parameters()).device, batch_size=batch,
                    dtype=next(model.parameters()).dtype) for _ in model.transformer.h]


@torch.no_grad()
def cached_pass(model, tokens, caches, prompt, *, collect=False):
    """Teacher-forced sequential decoding; no sampling or tokenization in timing."""
    for c in caches:
        c.reset()
    synchronize_device(tokens.device)
    begin = time.perf_counter()
    logits = model(tokens[:, :prompt], cache=caches)[0]
    synchronize_device(tokens.device)
    prefill = time.perf_counter()-begin
    parts = [logits] if collect else None
    begin = time.perf_counter()
    for pos in range(prompt, tokens.shape[1]):
        logits = model(tokens[:, pos:pos+1], cache=caches, start_pos=pos)[0]
        if collect:
            parts.append(logits)
    synchronize_device(tokens.device)
    decode = time.perf_counter()-begin
    return {'prefill_seconds': prefill, 'decode_seconds': decode}, (torch.cat(parts, 1) if collect else None)


def summarize_pair(job, measurements):
    if len(measurements) != job['repeats']:
        raise ValueError('Wrong repetition count')
    result = {}
    for phase in ['prefill', 'decode']:
        key = phase+'_seconds'
        a = np.asarray([r['candidate'][key] for r in measurements])
        b = np.asarray([r['reference'][key] for r in measurements])
        if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.min(), b.min()) <= 0:
            raise ValueError('Nonpositive or nonfinite timing')
        ratios = a/b
        result[phase] = dict(candidate_median_seconds=float(np.median(a)),
                            reference_median_seconds=float(np.median(b)),
                            paired_ratio_median=float(np.median(ratios)),
                            paired_ratio_iqr=np.quantile(ratios, [.25, .75]).tolist())
        if phase == 'decode':
            result[phase]['candidate_ms_per_step'] = float(np.median(a)*1000/job['decode'])
            result[phase]['reference_ms_per_step'] = float(np.median(b)*1000/job['decode'])
            result[phase]['candidate_batch_tokens_per_second'] = float(job['batch']*job['decode']/np.median(a))
            result[phase]['reference_batch_tokens_per_second'] = float(job['batch']*job['decode']/np.median(b))
    return result


def evaluate_job(job):
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1' or not torch.backends.mps.is_available():
        raise ValueError('Strict MPS required without fallback')
    torch.set_num_threads(4)
    models, caches, correctness = {}, {}, {}
    panel = None
    for role in ['reference', 'candidate']:
        spec = job['models'][role];path = Path(spec['checkpoint'])
        if file_digest(path) != spec['checkpoint_sha256']:
            raise ValueError('Checkpoint changed')
        saved = read_checkpoint(path);cfg = OmegaConf.create(saved['cfg'])
        if (saved['step'] != 3000 or cfg.model.dim != job['width']
                or cfg.model.n_layers != job['layers'] or cfg.model.block_size != 128
                or cfg.training.seed != job['seed'] or cfg.model.dropout != 0
                or cfg.data.train_fraction != .8 or cfg.data.val_fraction != .1):
            raise ValueError('Checkpoint configuration mismatch')
        if file_digest(Path(cfg.data.path)) != job['data_sha256']:
            raise ValueError('Corpus changed')
        loader = _make_loader(cfg, 'mps')
        if not _contracts_match(saved['training_state']['contract'], _training_contract(cfg, 'mps', loader)):
            raise ValueError('Historical training contract mismatch')
        length = job['prompt']+job['decode']
        # Disjoint validation strings, same for both models; no test access.
        tokens = loader.val_data[:job['batch']*length].reshape(job['batch'], length).to('mps')
        if panel is not None and not torch.equal(panel, tokens):
            raise ValueError('Mismatched timing tokens')
        panel = tokens
        model = GPT(cfg.model).to('mps').eval();load_model_weights(model, saved)
        expected_type = 'mha' if role == 'reference' else ('gqa' if job['models'][role]['family'] == 'gqa_half' else 'mla')
        if cfg.model.attn_type != expected_type:
            raise ValueError('Attention family mismatch')
        cache = make_caches(model, cfg, job['batch'])
        with torch.no_grad():
            full = model(tokens)[0]
            _, cached = cached_pass(model, tokens, cache, job['prompt'], collect=True)
            torch.testing.assert_close(cached, full, atol=job['atol'], rtol=job['rtol'])
            correctness[role] = {'max_absolute_logit_error': float((cached-full).abs().max().item()),
                                 'cache_allocated_bytes': cache_bytes(cache),
                                 'cache_logical_used_bytes': cache_bytes(cache)*length//128}
        models[role], caches[role] = model, cache
        del saved, full, cached, loader
    for _ in range(job['warmups']):
        for role in ['reference', 'candidate']:
            cached_pass(models[role], panel, caches[role], job['prompt'])
    measurements = []
    for i in range(job['repeats']):
        order = ['reference', 'candidate'] if (i+job['first_role_offset'])%2 == 0 else ['candidate', 'reference']
        row = {'order': order}
        for role in order:
            row[role], _ = cached_pass(models[role], panel, caches[role], job['prompt'])
        measurements.append(row)
    for spec in job['models'].values():
        if file_digest(Path(spec['checkpoint'])) != spec['checkpoint_sha256']:
            raise ValueError('Checkpoint changed during benchmark')
    return dict(job_sha256=digest(job), device='mps', torch=str(torch.__version__), cpu_threads=4,
                dtype='float32', both_models_resident=True, holdout_consumed=False,
                peak_device_memory_bytes=None, mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                correctness=correctness, measurements=measurements, summary=summarize_pair(job, measurements))


def validate_result(job, result):
    if (result['job_sha256'] != digest(job) or result['device'] != 'mps'
            or result['torch'] != str(torch.__version__) or result['cpu_threads'] != 4
            or result['dtype'] != 'float32' or not result['both_models_resident']
            or result['holdout_consumed'] or result['peak_device_memory_bytes'] is not None):
        raise ValueError('Worker contract mismatch')
    for i, row in enumerate(result['measurements']):
        order = ['reference', 'candidate'] if (i+job['first_role_offset'])%2 == 0 else ['candidate', 'reference']
        if row['order'] != order:
            raise ValueError('Pair order changed')
    if result['summary'] != summarize_pair(job, result['measurements']):
        raise ValueError('Timing aggregate changed')
    w, b, l = job['width'], job['batch'], job['layers']
    for role, value in result['correctness'].items():
        channels = 2*w if role == 'reference' else (w if job['models'][role]['family']=='gqa_half' else w//4+32)
        expected = l*b*128*channels*4
        if (value['cache_allocated_bytes'] != expected
                or value['cache_logical_used_bytes'] != expected*(job['prompt']+job['decode'])//128
                or not math.isfinite(value['max_absolute_logit_error']) or value['max_absolute_logit_error'] < 0):
            raise ValueError('Cache/correctness result mismatch')
    if set(result['correctness']) != {'reference', 'candidate'}:
        raise ValueError('Missing correctness result')


def verify_plan(plan, check_checkpoints=True):
    root = Path(plan['source_root'])
    if (plan['job_seconds'], plan['session_seconds'], plan['max_disk_mb'], plan['max_jobs']) != (120,3600,512,24):
        raise ValueError('Frozen limits changed')
    if (plan['runtime'] != dict(torch=str(torch.__version__),device='mps',cpu_threads=4)
            or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'):
        raise ValueError('Runtime changed')
    if (source_digest(root) != plan['source_sha256']
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=plan['git_commit']
            or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()):
        raise ValueError('Source changed or dirty')
    for item in [plan['knowledge_snapshot'], *plan['artifacts']]:
        if file_digest(Path(item['path'])) != item['sha256']:
            raise ValueError('Frozen artifact changed')
    expected = panel_jobs(root)
    for job in expected:
        original = json.loads((root/'research_runs'/job['study']/'plan.json').read_text())
        job['data_sha256'] = original['data_sha256']
    if plan['jobs'] != expected:
        raise ValueError('Frozen checkpoint/workload panel changed')
    for study, _, _, _ in PANEL:
        directory=root/'research_runs'/study
        if json.loads((directory/'review.json').read_text())['status'] != 'verified':
            raise ValueError('Historical review missing')
        required={str(directory/f) for f in ['plan.json','report.json','selection.json','review.json']}
        if not required.issubset({x['path'] for x in plan['artifacts']}):
            raise ValueError('Historical artifact missing')
    if check_checkpoints:
        for job in plan['jobs']:
            for item in job['models'].values():
                if file_digest(Path(item['checkpoint'])) != item['checkpoint_sha256']:
                    raise ValueError('Checkpoint changed')


def build_report(plan, records):
    if len(records)!=24 or any(r['status']!='completed' for r in records):
        raise ValueError('Incomplete benchmark')
    return {'plan_sha256':digest(plan), 'records':records, 'holdout_consumed':False,
            'scope':'Adaptive saved-checkpoint implementation diagnostic; teacher-forced sequential decoding, not serving throughput or quality selection. Current MPS allocation is not peak memory.'}


def run(plan, output):
    output=Path(output).resolve();output.mkdir(exist_ok=True,parents=True);verify_plan(plan)
    with session_lock(output) as lock_fd:
        state_path=output/'state.json'
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(plan_sha256=digest(plan),records=[],status='running',active_seconds=0.)
        if state['plan_sha256']!=digest(plan):raise ValueError('Frozen plan changed')
        for row in state['records']:
            if row['status']!='completed' or file_digest(output/row['id']/'result.json')!=row['result_sha256']:
                raise ValueError('Interrupted job or changed completed result; no automatic retry')
        if state['status']=='completed':return state
        if state['status']=='failed':raise ValueError('No automatic retry')
        state['status']='running';state.pop('error',None)
        start=time.monotonic();prior=state['active_seconds']
        def save(reserve=0):
            state['active_seconds']=prior+time.monotonic()-start+reserve;atomic_json(state_path,state)
        try:
            for job in plan['jobs']:
                if any(r['id']==job['id'] for r in state['records']):continue
                if (output/'STOP').exists():raise InterruptedError('Boundary stop')
                remaining=plan['session_seconds']-prior-(time.monotonic()-start)
                if remaining<=0:raise TimeoutError('Round budget exhausted')
                allowance=min(remaining,plan['job_seconds']);verify_plan(plan,False)
                row=dict(job,status='running');state['records'].append(row);save(allowance)
                directory=output/job['id'];directory.mkdir(exist_ok=False);atomic_json(directory/'request.json',job)
                before=time.monotonic()
                with (directory/'worker.log').open('w') as log:
                    proc=subprocess.Popen([sys.executable,'-m','src.research.inference_bench','worker','--request',str(directory/'request.json')],cwd=plan['source_root'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
                    try:
                        while proc.poll() is None:
                            if time.monotonic()-before>allowance or disk_mb(output)>plan['max_disk_mb']:
                                raise TimeoutError('Worker time/disk budget exhausted')
                            time.sleep(.2)
                        if proc.returncode:raise RuntimeError(f'Worker failed: {job["id"]}')
                    except BaseException:
                        terminate(proc);raise
                raw=json.loads((directory/'result.json').read_text());validate_result(job,raw)
                row.update(raw,status='completed',duration_seconds=time.monotonic()-before,
                           result_sha256=file_digest(directory/'result.json'));save()
            verify_plan(plan);atomic_json(output/'report.json',build_report(plan,state['records']));state['status']='completed'
        except BaseException as exc:
            state.update(status='paused' if isinstance(exc,InterruptedError) else 'failed',error=str(exc));raise
        finally:save()
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','run']);p.add_argument('--request',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.mode=='worker':atomic_json(args.request.parent/'result.json',evaluate_job(json.loads(args.request.read_text())))
    else:run(json.loads(args.plan.read_text()),args.output)

if __name__=='__main__':main()
