"""Recovery executor: unchanged training/comparisons, corrected no-backward scoring."""
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
from research.byte_context_scoring_v2 import score
BASE=ORIGINAL/'quality-recovery001'
ORIGINAL_WEIGHTS=ROOT/'research_runs/byte-context-transform-recovery-001'

WEIGHTS=ROOT/'research_runs/byte-context-transform-recovery-001'
CAP=int(1.5*1024**3)
ARMS=('byte','linear','tanh')
TORCH=None


def prepare():
    assert not BASE.exists() and not WEIGHTS.exists()
    old=read(ORIGINAL/'quality-plan.json');pins(old)
    assert sha(ORIGINAL/'quality-plan.json')=='e5d26b5af0cf473f62abe4d25b0555127a52c5531b38b9a7b3faf46f0517586c'
    fix=ORIGINAL/'scoring-recovery001';pins(read(fix/'plan.json'));proof=read(fix/'acceptance.json');result=read(fix/'result.json')
    assert proof['accepted'] and proof['result_sha256']==sha(fix/'result.json') and proof['guard_sha256']==sha(fix/'guard.json')
    assert read(fix/'guard.json')['status']=='finished' and result['completed'] and result['training_updates']==0
    assert all(result[k] for k in ('cpu_original_exact','weights_unchanged','gradients_absent','cpu_rng_unchanged','mps_rng_unchanged'))
    assert result['mean_ce_error']<=old['cpu_mps_mean_ce_atol'] and result['max_target_ce_error']<=old['cpu_mps_target_ce_atol']
    assert read(ORIGINAL/'controller-failure.json')==dict(returncode=1,slot=0)
    assert all(not (ORIGINAL/f'{i:02d}').exists() for i in range(1,9))
    preserved=read(ORIGINAL/'original-failure-preservation.json')
    for w in preserved['weights']:assert sha(w['path'])==w['sha256'] and Path(w['path']).stat().st_size==w['size']
    oldguard=read(ORIGINAL/'00/guard.json')
    assert oldguard['status']=='fixture_exception' and oldguard['reason']=='AssertionError()'
    history=[json.loads(x) for x in (ORIGINAL/'00/training.jsonl').read_text().splitlines()]
    assert len(history)==1200 and all(r['step']==i and r['gradient_tensors']==105 for i,r in enumerate(history,1))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('research','src') for p in (ROOT/folder).rglob('*.py')}
    bound=[ORIGINAL/'quality-plan.json',ORIGINAL/'original-failure-preservation.json',ORIGINAL/'recovery-decision.md',
           fix/'plan.json',fix/'acceptance.json',fix/'result.json',fix/'guard.json']
    sources.update({str(p.relative_to(ROOT)):sha(p) for p in bound})
    used=sum(p.stat().st_size for root in (ORIGINAL,ORIGINAL_WEIGHTS) for p in root.rglob('*') if p.is_file())
    free=shutil.disk_usage(ROOT).free
    assert free>=CAP-used+10*1024**3
    elapsed=oldguard['elapsed_seconds']+read(ORIGINAL/'pilot/guard.json')['elapsed_seconds']
    elapsed+=sum(read(ORIGINAL/n/'guard.json')['elapsed_seconds'] for n in ('numerical-diagnostic001','rms-diagnostic001','scoring-recovery001'))
    plan=dict(old,kind='byte_context_transform_quality_recovery_v2',sources=sources,runtime=runtime(),
        original_plan_sha256=sha(ORIGINAL/'quality-plan.json'),original_failure_sha256=sha(ORIGINAL/'original-failure-preservation.json'),
        correction_plan_sha256=sha(fix/'plan.json'),correction_result_sha256=sha(fix/'result.json'),
        scoring_policy='byte_context_fp32_grad_enabled_no_backward_v2',reused_training_slots=[0],unattempted_slots=list(range(1,9)),
        carried_elapsed_seconds=elapsed,free_at_recovery=free,used_at_recovery=used,
        forecast_scope='Original per-trial1200s scoring allowance retained; corrected full two-backend score measured by recovery proof.',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    assert plan['specs']==old['specs']
    BASE.mkdir();WEIGHTS.mkdir();put(BASE/'quality-plan.json',plan)
    # Copy immutable evidence, not weight payloads or synthetic guard success.
    out=BASE/'00';out.mkdir()
    for name in ['config.json','training.jsonl',*(f'curve-{i}.json' for i in (300,600,900,1200))]:
        with (out/name).open('xb') as f:f.write((ORIGINAL/'00'/name).read_bytes())
    for name in ('validation-cpu.json','validation-mps.json','guard.json'):
        with (out/name).open('xb') as f:f.write((fix/name).read_bytes())
    numerical={k:result[k] for k in ('mean_ce_error','max_target_ce_error')};put(out/'numerical.json',numerical)
    records={Path(w['path']).stem:w for w in preserved['weights']}
    curves=[read(out/f'curve-{i}.json') for i in (300,600,900,1200)]
    # Derive the expected private sampler state; original observed digest is missing.
    # This audits only1200 two-integer CPU draws, without data, model or training.
    import torch
    generator=torch.Generator(device='cpu').manual_seed(old['specs'][0]['roles']['train_sampler'])
    for _ in range(1200):torch.randint(267688-512,(2,),generator=generator)
    derived_sampler=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
    put(out/'attempt.json',dict(kind='import_verified_completed_training',plan_sha256=sha(BASE/'quality-plan.json'),
        original_attempt_sha256=sha(ORIGINAL/'00/attempt.json'),original_guard_sha256=sha(ORIGINAL/'00/guard.json'),
        correction_acceptance_sha256=sha(fix/'acceptance.json')))
    put(out/'result.json',dict(completed=True,spec=old['specs'][0],steps=1200,targets_seen=1228800,averaging_snapshots=41,
        ce=read(out/'validation-mps.json')['ce'],raw=records['raw'],h200=records['h200'],parameter_count=16351859,
        train_sampler_sha256=None,expected_sampler_sha256=derived_sampler,sampler_digest_scope='Original observed terminal sampler digest is absent; expected state is derived from frozen seed and1200 declared two-integer draws. This is source/count-based pairing evidence, not recovered runtime state or optimizer resume.',
        curves=curves,curves_policy='Original no-grad online diagnostics preserved; not primary quality evidence.',
        elapsed_seconds=oldguard['elapsed_seconds'],elapsed_scope='Original complete guard duration including failed scorer',
        numerical=numerical,holdout_scored=False,adopted=False,imported_training=True))
    put(out/'acceptance.json',dict(accepted=True,kind='original_training_plus_verified_corrected_scoring',
        result_sha256=sha(out/'result.json'),guard_sha256=sha(out/'guard.json'),
        original_training_sha256=sha(ORIGINAL/'00/training.jsonl'),correction_acceptance_sha256=sha(fix/'acceptance.json')))
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
        assert all(r['train_sampler_sha256'] is not None or (start==0 and r['spec']['slot']==0 and r.get('imported_training')) for r in rows[start:start+3])
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
    for index in range(1,9):
        assert not (BASE/f'{index:02d}').exists(), 'No implicit replay'
        remaining=plan['total_seconds']-plan['carried_elapsed_seconds']-(time.monotonic()-started)
        assert remaining>=plan['per_trial_seconds']
        child=subprocess.run([sys.executable,'-m','research.byte_context_transform_run_v2','worker',str(index)],cwd=ROOT)
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
