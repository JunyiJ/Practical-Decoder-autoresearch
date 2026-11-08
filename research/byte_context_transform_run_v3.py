"""CPU-reference recovery: unchanged training and primary MPS scoring."""
import argparse
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from research.byte_context_transform_pilot_v1 import ROOT, BASE as ORIGINAL, sha, put, read, runtime, pins
from research.byte_context_cpu_reference_v1 import score
PREVIOUS=ORIGINAL/'quality-recovery001'
FIX=ORIGINAL/'cpu-reference-recovery001'
BASE=ORIGINAL/'quality-recovery002'
ORIGINAL_WEIGHTS=ROOT/'research_runs/byte-context-transform-001'

WEIGHTS=ROOT/'research_runs/byte-context-transform-recovery-001'
CAP=int(1.5*1024**3)
ARMS=('byte','linear','tanh')
TORCH=None


def prepare():
    assert not BASE.exists()
    prior=read(PREVIOUS/'quality-plan.json');pins(prior)
    pins(read(FIX/'plan.json'))
    proof=read(FIX/'acceptance.json');verified=read(FIX/'result.json')
    assert proof['accepted'] and proof['result_sha256']==sha(FIX/'result.json') and proof['guard_sha256']==sha(FIX/'guard.json')
    assert read(FIX/'guard.json')['status']=='finished' and verified['completed']
    assert verified['primary_scores_unchanged'] and verified['no_training'] and verified['no_threshold_change']
    assert all(not (PREVIOUS/f'{i:02d}').exists() for i in range(3,9))
    assert read(PREVIOUS/'resource-recovery001/controller-failure.json')==dict(returncode=1,slot=2)
    preserved=read(PREVIOUS/'slot02-failure-preservation.json')
    for record in preserved['evidence']:
        assert sha(record['path'])==record['sha256'] and Path(record['path']).stat().st_size==record['size']
    assert all(row['original_cpu_exact'] and row['fp64_max_error']<=3e-5 for row in verified['rows'])
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('research','src') for p in (ROOT/folder).rglob('*.py')}
    bound=[PREVIOUS/'quality-plan.json',PREVIOUS/'slot02-failure-preservation.json',
        ORIGINAL/'cpu-reference-recovery-decision.md',FIX/'plan.json',FIX/'acceptance.json',FIX/'result.json',FIX/'guard.json',
        PREVIOUS/'resource-recovery001/slot-01.json',ORIGINAL/'numerical-diagnostic002/plan.json',ORIGINAL/'numerical-diagnostic002/result.json']
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in bound})
    roots=(ORIGINAL,ORIGINAL_WEIGHTS,WEIGHTS)
    assert len(set(roots))==3
    used=sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
    free=shutil.disk_usage(ROOT).free
    assert free>=CAP-used+10*1024**3
    elapsed=prior['carried_elapsed_seconds']+sum(read(PREVIOUS/f'{i:02d}/guard.json')['elapsed_seconds'] for i in (1,2))
    elapsed+=read(ORIGINAL/'numerical-diagnostic002/guard.json')['elapsed_seconds']+read(FIX/'guard.json')['elapsed_seconds']
    plan=dict(prior,kind='byte_context_transform_cpu_reference_recovery_v3',sources=sources,runtime=runtime(),
        previous_plan_sha256=sha(PREVIOUS/'quality-plan.json'),cpu_reference_plan_sha256=sha(FIX/'plan.json'),
        cpu_reference_result_sha256=sha(FIX/'result.json'),scoring_policy='MPS original v2; CPU stable FP32 reference v1',
        reused_training_slots=[0,1,2],unattempted_slots=list(range(3,9)),
        carried_elapsed_seconds=elapsed,free_at_recovery=free,used_at_recovery=used,
        resource_roots=[str(p) for p in roots],
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    assert plan['specs']==prior['specs']
    BASE.mkdir();put(BASE/'quality-plan.json',plan)
    import torch
    for index in range(3):
        old=PREVIOUS/f'{index:02d}';out=BASE/f'{index:02d}';out.mkdir();spec=plan['specs'][index]
        original_guard=ORIGINAL/'00/guard.json' if index==0 else old/'guard.json'
        guard=read(original_guard)
        assert guard['status']==('finished' if index==1 else 'fixture_exception')
        history=[json.loads(x) for x in (old/'training.jsonl').read_text().splitlines()]
        assert len(history)==1200 and all(h['step']==i for i,h in enumerate(history,1))
        for name in ['config.json','training.jsonl','validation-mps.json',*(f'curve-{i}.json' for i in (300,600,900,1200))]:
            with (out/name).open('xb') as f:f.write((old/name).read_bytes())
        for name,origin in [('validation-cpu.json',FIX/f'{index:02d}-cpu.json'),('guard.json',FIX/'guard.json')]:
            with (out/name).open('xb') as f:f.write(origin.read_bytes())
        cpu=read(out/'validation-cpu.json');mps=read(out/'validation-mps.json')
        numerical=dict(mean_ce_error=abs(cpu['ce']-mps['ce']),max_target_ce_error=max(abs(a-b) for a,b in zip(cpu['per_target_losses'],mps['per_target_losses'],strict=True)))
        assert numerical['mean_ce_error']<=3e-5 and numerical['max_target_ce_error']<=3e-4
        put(out/'numerical.json',numerical)
        previous=read(old/'result.json') if index<2 else None
        records={}
        for kind in ('raw','h200'):
            p=(ORIGINAL_WEIGHTS if index==0 else WEIGHTS)/f'{index:02d}'/(kind+'.pt')
            records[kind]=dict(path=str(p),size=p.stat().st_size,sha256=sha(p))
        generator=torch.Generator(device='cpu').manual_seed(spec['roles']['train_sampler'])
        for _ in range(1200):torch.randint(267688-512,(2,),generator=generator)
        expected=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
        observed=previous['train_sampler_sha256'] if previous else None
        assert observed is None or observed==expected
        put(out/'attempt.json',dict(kind='import_completed_training_verified_cpu_reference',plan_sha256=sha(BASE/'quality-plan.json'),
            original_attempt_sha256=sha(old/'attempt.json'),original_guard_path=str(original_guard),original_guard_sha256=sha(original_guard),
            correction_acceptance_sha256=sha(FIX/'acceptance.json')))
        put(out/'result.json',dict(completed=True,spec=spec,steps=1200,targets_seen=1228800,averaging_snapshots=41,
            ce=mps['ce'],raw=records['raw'],h200=records['h200'],parameter_count=16351859 if index==0 else 16417395,
            train_sampler_sha256=observed,expected_sampler_sha256=expected,
            sampler_digest_scope='Observed digest retained for slot01. Slots00/02 lack a published terminal digest; expected state uses frozen seed and1200 two-integer draws, not recovered runtime state or optimizer resume.',
            curves=[read(out/f'curve-{i}.json') for i in (300,600,900,1200)],
            curves_policy='Original curves preserved; slot00 used old no-grad scorer, slots01/02 use v2. None are primary endpoints.',
            elapsed_seconds=guard['elapsed_seconds'],elapsed_scope='Original complete training guard including scoring',
            numerical=numerical,holdout_scored=False,adopted=False,imported_training=True))
        put(out/'acceptance.json',dict(accepted=True,kind='completed_training_plus_verified_cpu_reference',result_sha256=sha(out/'result.json'),
            guard_sha256=sha(out/'guard.json'),original_training_sha256=sha(old/'training.jsonl'),original_guard_sha256=sha(original_guard),
            original_primary_sha256=sha(old/'validation-mps.json'),correction_acceptance_sha256=sha(FIX/'acceptance.json')))
    print(json.dumps(dict(plan_sha256=sha(BASE/'quality-plan.json'),carried_seconds=elapsed,free=free)),flush=True)


def clean():
    gc.collect()
    if TORCH.backends.mps.is_available():TORCH.mps.synchronize();TORCH.mps.empty_cache()


def objects(model):
    from omegaconf import OmegaConf
    from research.byte_context_transform_v1 import Optimizer
    from research.byte_input_training_v6 import recipe,CanonicalParameters
    from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
    from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
    cfg=OmegaConf.create(recipe(1200,.0025));opt=Optimizer(model)
    return opt,LearningRateSchedule(opt,cfg),TailWeightAverages(CanonicalParameters(model),cfg)


def save(path,model,spec,plan,kind):
    import torch
    from research.byte_context_transform_v1 import weights
    payload=dict(kind=kind,identity=model.identity,spec=spec,plan_sha256=sha(BASE/'quality-plan.json'),
                 codebook_identity=model.codebook_identity,weights=weights(model),optimizer_resume=False)
    with path.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
    copy=torch.load(path,map_location='cpu',weights_only=True)
    assert copy.keys()==payload.keys()
    for k in payload:
        if k!='weights':assert copy[k]==payload[k]
    assert list(copy['weights'])==list(payload['weights'])
    assert all(torch.equal(copy['weights'][n],v) for n,v in payload['weights'].items())
    return dict(path=str(path),sha256=sha(path),size=path.stat().st_size,exact_readback=True)


def trial(plan,index,out):
    import torch
    from research.byte_context_transform_v1 import construct,load_weights
    from research.byte_input_model_v3 import build_codebook
    from research.byte_input_training_v6 import load_view,check_view
    from research.byte_context_transform_pilot_v1 import one_step
    spec=plan['specs'][index];r=spec['roles'];arm=spec['arm']
    book=build_codebook(*(Path(row['path']).read_bytes() for row in plan['codebook_assets']),'raw_bytes')
    model=construct(arm,r,spec['projection_seed'],book,'mps');view=load_view(r,'mps')
    assert view.data_identity==plan['data']
    opt,sched,avg=objects(model);curves=[]
    put(out/'config.json',dict(spec=spec,data=view.data_identity,model_identity=model.identity,
                              parameters=sum(p.numel() for _,p in model.ordered_parameters())))
    tick=time.perf_counter()
    with (out/'training.jsonl').open('x') as stream:
        for step in range(1,1201):
            x,y=view.get_batch('train');start=time.perf_counter()
            row=one_step(model,opt,sched,avg,x,y,step);torch.mps.synchronize();row['seconds']=time.perf_counter()-start
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush()
            if step in (300,600,900,1200):
                measured=score(model,view,'mps');curves.append(dict(step=step,ce=measured['ce'],seconds=measured['seconds']))
                put(out/f'curve-{step}.json',curves[-1]);del measured
                print(f'{index:02d} {arm} step{step} online_validation={curves[-1]["ce"]:.6f}',flush=True)
        os.fsync(stream.fileno())
    elapsed=time.perf_counter()-tick
    assert opt.steps==avg.completed_steps==1200 and avg.snapshots==41
    model.validate();check_view(view,r)
    folder=WEIGHTS/f'{index:02d}';folder.mkdir()
    raw=save(folder/'raw.pt',model,spec,plan,'final_raw_weights_not_resume')
    with torch.no_grad():
        for name,p in model.ordered_parameters():p.copy_(avg.averages['200'][name])
    model.validate();h200=save(folder/'h200.pt',model,spec,plan,'h200_inference_weights')
    del opt,sched,avg;clean()
    saved=torch.load(h200['path'],map_location='cpu',weights_only=True)
    assert saved['identity']==model.identity and saved['spec']==spec and saved['plan_sha256']==sha(BASE/'quality-plan.json')
    assert saved['codebook_identity']==book.identity
    load_weights(model,saved['weights']);del saved
    mps=score(model,view,'mps');put(out/'validation-mps.json',mps)
    model.to('cpu');cpu=score(model,view,'cpu');put(out/'validation-cpu.json',cpu)
    numerical=dict(mean_ce_error=abs(cpu['ce']-mps['ce']),
                   max_target_ce_error=max(abs(a-b) for a,b in zip(cpu['per_target_losses'],mps['per_target_losses'])))
    # Original failed reports remain available even when acceptance fails.
    put(out/'numerical.json',numerical)
    assert numerical['mean_ce_error']<=plan['cpu_mps_mean_ce_atol'] and numerical['max_target_ce_error']<=plan['cpu_mps_target_ce_atol']
    check_view(view,r);model.validate()
    put(out/'result.json',dict(completed=True,spec=spec,steps=1200,targets_seen=1228800,averaging_snapshots=41,
         ce=mps['ce'],raw=raw,h200=h200,parameter_count=sum(p.numel() for _,p in model.ordered_parameters()),
         train_sampler_sha256=hashlib.sha256(view.generators['train'].get_state().numpy().tobytes()).hexdigest(),
         curves=curves,elapsed_seconds=elapsed,numerical=numerical,holdout_scored=False,adopted=False))


def worker(index):
    global TORCH
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=read(BASE/'quality-plan.json');pins(plan)
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample()
            if TORCH is not None:
                sample['mps_bytes']=TORCH.mps.driver_allocated_memory()
                if sample['mps_bytes']>=6*1024**3:reason='mps_limit'
            sample['retained_bytes']=sum(p.stat().st_size for root in (ORIGINAL,ORIGINAL_WEIGHTS,WEIGHTS) for p in root.rglob('*') if p.is_file())
            if sample['retained_bytes']>CAP:reason='output_limit'
            if (BASE/'STOP').exists():reason='explicit_stop'
            return sample,reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        used=sum(p.stat().st_size for root in (ORIGINAL,ORIGINAL_WEIGHTS,WEIGHTS) for p in root.rglob('*') if p.is_file())
        assert shutil.disk_usage(ROOT).free>=CAP-used+10*1024**3
        out=BASE/f'{index:02d}';out.mkdir()
        put(out/'attempt.json',dict(pid=os.getpid(),slot=index,plan_sha256=sha(BASE/'quality-plan.json')))
        try:
            with Guard(out/'guard.json',ROOT,timeout_seconds=plan['per_trial_seconds'],poll_seconds=.5):
                import torch
                TORCH=torch;torch.set_num_threads(4)
                assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and sys.flags.optimize==0
                trial(plan,index,out);pins(plan)
            put(out/'acceptance.json',dict(accepted=True,result_sha256=sha(out/'result.json'),guard_sha256=sha(out/'guard.json')))
        except BaseException:
            put(out/'failure.json',dict(traceback=traceback.format_exc(),classification_pending=True));raise


def finish(plan):
    import numpy as np
    pairs=[];rows=[]
    for index,spec in enumerate(plan['specs']):
        out=BASE/f'{index:02d}';accept=read(out/'acceptance.json');row=read(out/'result.json')
        assert accept['accepted'] and accept['result_sha256']==sha(out/'result.json') and accept['guard_sha256']==sha(out/'guard.json')
        assert read(out/'guard.json')['status']=='finished' and row['spec']==spec
        for key in ('raw','h200'):assert sha(row[key]['path'])==row[key]['sha256']
        scores=read(out/'validation-mps.json');assert scores['ce']==math.fsum(scores['per_target_losses'])/34278==row['ce']
        if index%3==0:pairs.append({})
        pairs[-1][spec['arm']]=np.array(scores['per_target_losses'],dtype=np.float64);rows.append(row)
    for start in (0,3,6):
        digests={r['train_sampler_sha256'] if r['train_sampler_sha256'] is not None else r['expected_sampler_sha256'] for r in rows[start:start+3]}
        assert len(digests)==1
        assert all(r['train_sampler_sha256'] is not None or (start==0 and r['spec']['slot'] in (0,2) and r.get('imported_training')) for r in rows[start:start+3])
    contrasts={}
    for control in ('byte','linear'):
        diffs=np.stack([p['tanh']-p[control] for p in pairs]);deltas=diffs.mean(axis=1);bounds=[]
        for width in (2048,4096,8192):
            starts=range(0,34278,width);counts=np.array([min(width,34278-i) for i in starts])
            blocks=np.stack([diffs[:,i:i+width].sum(axis=1) for i in starts],axis=1)
            draws=np.random.Generator(np.random.PCG64(plan['bootstrap_seed'])).integers(0,len(counts),(2000,len(counts)))
            values=(blocks[:,draws].sum(axis=2)/counts[draws].sum(axis=1)).mean(axis=0)
            bounds.append(dict(block_tokens=width,upper=float(np.quantile(values,plan['bootstrap_quantile']))))
        gates=dict(mean_gain=float(deltas.mean())<-.02,every_seed=bool((deltas<0).all()),all_bounds=all(b['upper']<0 for b in bounds))
        contrasts[control]=dict(deltas=deltas.tolist(),mean=float(deltas.mean()),bounds=bounds,gates=gates,passes=all(gates.values()))
    put(BASE/'final.json',dict(completed=True,trajectories=9,updates=10800,
        means={a:sum(r['ce'] for r in rows if r['spec']['arm']==a)/3 for a in ARMS},contrasts=contrasts,
        quality_success=all(r['passes'] for r in contrasts.values()),adopted=False,holdout_scored=False,
        scope='Fixed .0025 recipe, three trained bundles, reused validation; independent review and consumer proof required.'))


def run():
    plan=read(BASE/'quality-plan.json');pins(plan);started=time.monotonic()
    put(BASE/'controller-attempt.json',dict(pid=os.getpid(),plan_sha256=sha(BASE/'quality-plan.json')))
    for index in range(3,9):
        assert not (BASE/f'{index:02d}').exists(), 'No implicit replay'
        remaining=plan['total_seconds']-plan['carried_elapsed_seconds']-(time.monotonic()-started)
        assert remaining>=plan['per_trial_seconds']
        child=subprocess.run([sys.executable,'-m','research.byte_context_transform_run_v3','worker',str(index)],cwd=ROOT)
        if child.returncode:
            put(BASE/'controller-failure.json',dict(slot=index,returncode=child.returncode));return child.returncode
    pins(plan);finish(plan);put(BASE/'controller-complete.json',dict(completed=True,elapsed_seconds=time.monotonic()-started))
    print(json.dumps(read(BASE/'final.json')),flush=True);return 0


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run','worker'));parser.add_argument('index',nargs='?',type=int)
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='worker':worker(args.index)
    else:sys.exit(run())
