"""Bounded validation-only coverage audit of immutable saved checkpoints.

Every target except the split's first character is scored once in disjoint
windows. Context resets at each window; this is not sliding maximum-context PPL.
No test scoring, optimization, checkpoint rewriting, or historical re-selection.
"""
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
from ..models.gpt import GPT
from ..train.train import _make_loader, _training_contract, _contracts_match, _resolve_device
from ..utils.checkpoint import read_checkpoint, load_model_weights


def score_validation(model, data, context=128, batch_size=8, block_tokens=4096):
    if min(context, batch_size, block_tokens) < 1 or block_tokens % context:
        raise ValueError('Positive settings and text blocks divisible by context required')
    if data.ndim != 1 or len(data) < 2 or context > model.cfg.block_size:
        raise ValueError('Invalid validation data or context exceeds training context')
    model.eval()
    device = next(model.parameters()).device
    n = len(data)-1
    sums = np.zeros(math.ceil(n/block_tokens), dtype=np.float64)
    counts = np.zeros(len(sums), dtype=np.int64)
    # Each target index 1..len(data)-1 appears exactly once. Tail is unpadded.
    starts = list(range(0,n,context))
    with torch.no_grad():
        for offset in range(0,len(starts),batch_size):
            group = starts[offset:offset+batch_size]
            # Separate the short tail so padding cannot affect metrics.
            chunks = [group] if n-group[-1]>=context else [group[:-1],group[-1:]]
            for chunk in chunks:
                if not chunk: continue
                length = min(context,n-chunk[-1])
                x=torch.stack([data[i:i+length] for i in chunk]).to(device)
                y=torch.stack([data[i+1:i+length+1] for i in chunk]).to(device)
                logits=model(x)[0]
                if logits.shape[:2]!=y.shape: raise ValueError('Logit shape mismatch')
                losses=torch.nn.functional.cross_entropy(logits.reshape(-1,logits.shape[-1]),y.reshape(-1),reduction='none').reshape(y.shape)
                if not torch.isfinite(losses).all(): raise FloatingPointError('Nonfinite CE')
                values=losses.double().cpu().numpy() if device.type!='mps' else losses.cpu().double().numpy()
                for i,v in zip(chunk,values):
                    b=i//block_tokens
                    sums[b]+=v.sum();counts[b]+=len(v)
    assert counts.sum()==n
    return dict(context=context,eval_batch_size=batch_size,block_tokens=block_tokens,
                target_count=n,window_count=len(starts),mean_ce=float(sums.sum()/n),
                block_loss_sums=sums.tolist(),block_target_counts=counts.tolist())


def paired_blocks(candidates,references,block_tokens=4096,draws=2000,seed=20261024):
    """Conditional text-block bootstrap, not independent training-seed inference."""
    if set(candidates)!=set(references) or not candidates: raise ValueError('Unpaired seeds')
    deltas=[];count=None
    for key in sorted(candidates):
        a,b=candidates[key],references[key]
        if (a['context']!=b['context'] or a['block_target_counts']!=b['block_target_counts']
                or a['block_tokens']!=b['block_tokens'] or block_tokens%a['block_tokens']):
            raise ValueError('Different evaluation panels')
        new_count=np.asarray(a['block_target_counts'])
        if count is not None and not np.array_equal(count,new_count):
            raise ValueError('Different evaluation panels across seeds')
        count=new_count;deltas.append(np.asarray(a['block_loss_sums'])-b['block_loss_sums'])
    # Inputs use 2048-character blocks; merge adjacent blocks for sensitivity.
    factor=block_tokens//a['block_tokens'];index=np.arange(0,len(count),factor)
    delta=np.add.reduceat(np.mean(deltas,axis=0),index)
    count=np.add.reduceat(count,index)
    rng=np.random.default_rng(seed);sample=rng.integers(len(count),size=(draws,len(count)))
    estimates=delta[sample].sum(1)/count[sample].sum(1)
    return {'mean_delta':float(delta.sum()/count.sum()),'paired_seed_deltas':{
        str(k):candidates[k]['mean_ce']-references[k]['mean_ce'] for k in sorted(candidates)},
        'text_block_tokens':block_tokens,'text_block_count':len(count),
        'conditional_text_bootstrap_95pct':np.quantile(estimates,[.025,.975]).tolist(),
        'bootstrap_draws':draws,'bootstrap_seed':seed,
        'scope':'Text-block resampling conditional on these trained seeds and this adaptively reused validation split. Not seed uncertainty, independent holdout, or a multiple-comparison-adjusted significance test.'}


def evaluate_job(job):
    path=Path(job['checkpoint'])
    if file_digest(path)!=job['checkpoint_sha256']: raise ValueError('Checkpoint changed')
    c=read_checkpoint(path);cfg=OmegaConf.create(c['cfg']);cfg.training.strict_device=True
    if c['step']!=job['completed_steps']: raise ValueError('Step contract mismatch')
    if file_digest(Path(cfg.data.path))!=job['data_sha256']: raise ValueError('Corpus changed')
    if cfg.data.train_fraction!=.8 or cfg.data.val_fraction!=.1: raise ValueError('Expected reserved 80/10/10 split')
    torch.set_num_threads(4);device=_resolve_device(cfg);loader=_make_loader(cfg,device)
    if not _contracts_match(c['training_state']['contract'],_training_contract(cfg,device,loader)):
        raise ValueError('Saved training contract mismatch')
    model=GPT(cfg.model).to(device);load_model_weights(model,c)
    result={str(context):score_validation(model,loader.val_data,context,1024//context,2048)
            for context in job['contexts']}
    if file_digest(path)!=job['checkpoint_sha256']:raise ValueError('Checkpoint changed during scoring')
    return {'evaluations':result,'checkpoint_sha256':job['checkpoint_sha256'],
            'holdout_consumed':False,'device':device,'torch':str(torch.__version__)}


def build_report(plan,records):
    if len(records)!=len(plan['jobs']) or any(r['status']!='completed' for r in records):
        raise ValueError('Incomplete coverage audit')
    comparisons=[]
    for study in sorted({j['study'] for j in plan['jobs']}):
        for width in [64,128,256]:
            rows=[r for r in records if r['study']==study and r['width']==width]
            families=sorted({r['family'] for r in rows})
            for family in families:
                for reference in families:
                    if family==reference:continue
                    a={r['seed']:r['evaluations']['128'] for r in rows if r['family']==family}
                    b={r['seed']:r['evaluations']['128'] for r in rows if r['family']==reference}
                    comparisons.append(dict(study=study,width=width,family=family,reference=reference,
                        sensitivity=[paired_blocks(a,b,n) for n in [2048,4096,8192]]))
    return {'plan_sha256':digest(plan),'records':records,'comparisons':comparisons,'holdout_consumed':False,
            'scope':'Post-selection full-validation coverage audit; historical choices remain frozen. No training reruns or test access.'}


def validate_evaluation(value, expected_targets):
    counts = value['block_target_counts']
    sums = value['block_loss_sums']
    expected_counts = [min(2048, expected_targets-i) for i in range(0, expected_targets, 2048)]
    if (value['context'] != 128 or value['eval_batch_size'] != 8 or value['block_tokens'] != 2048
            or counts != expected_counts or len(sums) != len(counts)
            or value['target_count'] != expected_targets
            or value['window_count'] != math.ceil(expected_targets/128)
            or not all(math.isfinite(x) and x >= 0 for x in sums)
            or not math.isfinite(value['mean_ce'])
            or not math.isclose(value['mean_ce'], sum(sums)/expected_targets, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('Invalid full-validation coverage result')


def verify_plan(plan, *, check_checkpoints=True):
    root=Path(plan['source_root'])
    if (plan['job_seconds'],plan['session_seconds'],plan['max_disk_mb'])!=(120,3600,512):raise ValueError('Frozen evaluation limits changed')
    if source_digest(root)!=plan['source_sha256']:raise ValueError('Evaluator source changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Dirty source')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=plan['git_commit']:raise ValueError('Commit changed')
    if file_digest(Path(plan['knowledge_snapshot']['path']))!=plan['knowledge_snapshot']['sha256']:raise ValueError('Knowledge changed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Unset MPS fallback')
    if plan['max_jobs']!=81 or len(plan['jobs'])!=81 or len({j['id'] for j in plan['jobs']})!=81:raise ValueError('Frozen 81-checkpoint audit required')
    if plan['runtime'] != {'torch':str(torch.__version__),'device':'mps','cpu_threads':4}:
        raise ValueError('Evaluator runtime changed')
    for job in plan['jobs']:
        if job['contexts']!=[128] or job['completed_steps']!=3000:raise ValueError('Historical evaluation protocol changed')
        if check_checkpoints and file_digest(Path(job['checkpoint']))!=job['checkpoint_sha256']:raise ValueError('Saved checkpoint changed')
    for artifact in plan['artifacts']:
        if file_digest(Path(artifact['path']))!=artifact['sha256']:raise ValueError('Historical evidence changed')
    expected = []
    for study, families in [('architecture-stage2e-001',4),('architecture-stage2f-001',5)]:
        study_root = root/'research_runs'/study
        report = json.loads((study_root/'report.json').read_text())
        review = json.loads((study_root/'review.json').read_text())
        if review['status'] != 'verified' or len(report['families']) != families:
            raise ValueError('Historical review missing')
        required = {str(study_root/name) for name in ['plan.json','report.json','selection.json','review.json']}
        if not required.issubset({a['path'] for a in plan['artifacts']}):
            raise ValueError('Missing frozen historical artifacts')
        for row in report['records']:
            if row['lr'] == report['selection']['choices'][row['family']]['lr']:
                expected.append((study,row['family'],row['width'],row['seed'],row['checkpoint'],row['checkpoint_sha256']))
    actual = [(j['study'],j['family'],j['width'],j['seed'],j['checkpoint'],j['checkpoint_sha256']) for j in plan['jobs']]
    if sorted(actual) != sorted(expected):
        raise ValueError('Checkpoint panel differs from frozen historical selections')


def run(plan,output):
    output=Path(output).resolve();output.mkdir(exist_ok=True,parents=True);verify_plan(plan)
    with session_lock(output) as lock_fd:
        path=output/'state.json'
        state=json.loads(path.read_text()) if path.exists() else dict(plan_sha256=digest(plan),records=[],status='running',active_seconds=0.)
        if state['plan_sha256']!=digest(plan):raise ValueError('Frozen plan changed')
        if state['status']=='completed':return state
        if state['status']=='failed' or any(r['status']!='completed' for r in state['records']):raise ValueError('No automatic retry of failed/interrupted job')
        for row in state['records']:
            result_path=output/row['id']/'result.json'
            if file_digest(result_path)!=row['result_sha256']:
                raise ValueError('Completed evaluation result changed')
        state['status']='running';state.pop('error',None)
        start=time.monotonic();prior=state['active_seconds']
        def save(reserve=0):
            state['active_seconds']=prior+time.monotonic()-start+reserve;atomic_json(path,state)
        try:
            for job in plan['jobs']:
                if any(r['id']==job['id'] for r in state['records']):continue
                if (output/'STOP').exists():raise InterruptedError('Human boundary stop')
                remaining=plan['session_seconds']-prior-(time.monotonic()-start)
                if remaining<=0:raise TimeoutError('Session budget exhausted')
                allowance=min(remaining,plan['job_seconds']);verify_plan(plan,check_checkpoints=False)
                row=dict(job,status='running');state['records'].append(row);save(allowance)
                directory=output/job['id'];directory.mkdir(exist_ok=False)
                atomic_json(directory/'request.json',job)
                with (directory/'worker.log').open('w') as log:
                    proc=subprocess.Popen([sys.executable,'-m','src.research.coverage_eval','worker','--request',str(directory/'request.json')],cwd=plan['source_root'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(lock_fd,))
                    before=time.monotonic()
                    try:
                        while proc.poll() is None:
                            if time.monotonic()-before>allowance or disk_mb(output)>plan['max_disk_mb']:
                                raise TimeoutError('Evaluation process time/disk budget exhausted')
                            time.sleep(.2)
                        if proc.returncode:raise RuntimeError(f'Evaluation worker failed: {job["id"]}')
                    except BaseException:
                        terminate(proc);raise
                raw=json.loads((directory/'result.json').read_text())
                if raw['checkpoint_sha256']!=job['checkpoint_sha256'] or raw['holdout_consumed']:raise ValueError('Invalid worker contract')
                if raw['device'] != plan['runtime']['device'] or raw['torch'] != plan['runtime']['torch'] or set(raw['evaluations']) != {'128'}:
                    raise ValueError('Worker runtime or evaluation panel changed')
                validate_evaluation(raw['evaluations']['128'],plan['validation_targets'])
                row.update(raw,status='completed',duration_seconds=time.monotonic()-before,
                           result_sha256=file_digest(directory/'result.json'));save()
            verify_plan(plan);atomic_json(output/'report.json',build_report(plan,state['records']));state['status']='completed'
        except BaseException as exc:
            state.update(status='paused' if isinstance(exc,InterruptedError) else 'failed',error=str(exc));raise
        finally:save()
    return state


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['worker','run']);p.add_argument('--request',type=Path);p.add_argument('--plan',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    if args.mode=='worker':
        result=evaluate_job(json.loads(args.request.read_text()));atomic_json(args.request.parent/'result.json',result)
    else:run(json.loads(args.plan.read_text()),args.output)

if __name__=='__main__':main()
