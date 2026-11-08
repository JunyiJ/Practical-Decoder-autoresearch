"""Fresh fixed-recipe byte-context panel. Separate from every closed study."""
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
from research.byte_context_transform_pilot_v1 import ROOT, BASE, sha, put, read, runtime, pins

WEIGHTS=ROOT/'research_runs/byte-context-transform-001'
CAP=int(1.5*1024**3)
ARMS=('byte','linear','tanh')
TORCH=None


def prepare():
    assert not (BASE/'quality-plan.json').exists() and not WEIGHTS.exists()
    p=BASE/'pilot';accept=read(p/'acceptance.json');pilot=read(p/'result.json')
    assert accept['accepted'] and sha(p/'result.json')==accept['result_sha256']
    assert sha(p/'guard.json')==accept['guard_sha256'] and read(p/'guard.json')['status']=='finished'
    assert pilot['passed'] and pilot['training_updates']==32
    old=read(BASE/'pilot-plan.json');pins(old)
    # Complete forecast, retaining a generous scoring/serialization allowance.
    warmed={r['arm']:max(r['seconds'][1:]) for r in pilot['cases'] if r['device']=='mps'}
    forecast={a:1200*warmed[a]*1.5+1200 for a in ARMS}
    assert all(v<3600 for v in forecast.values()), forecast
    assert sum(forecast.values())*3+3600<=36000
    prior=subprocess.run(['rg','-l','6210220000[1-6]|6210230000[1-6]|6210240000[1-7]',
                          'research','research_state','-g','*.json','-g','*.jsonl'],cwd=ROOT,capture_output=True,text=True)
    assert prior.returncode==1 and not prior.stdout, ('Existing roles',prior.stdout,prior.stderr)
    paths=sorted(p for folder in ('research','src') for p in (ROOT/folder).rglob('*.py'))
    paths += [ROOT/'research/BYTE_CONTEXT_TRANSFORM_PROTOCOL.md',BASE/'pilot-plan.json',
              p/'acceptance.json',p/'result.json',p/'guard.json',BASE/'quality-admission.md']
    specs=[]
    for i,bundle in enumerate((62102200000,62102300000,62102400000)):
        roles=dict(zip(('native_initializer','composer_initializer','nonlexical_key','train_sampler','validation_sampler'),range(bundle+1,bundle+6)))
        for arm in ARMS[i:]+ARMS[:i]:
            specs.append(dict(slot=len(specs),bundle=bundle,arm=arm,roles=roles,projection_seed=bundle+6,steps=1200,learning_rate=.0025))
    free=shutil.disk_usage(ROOT).free
    assert free>=CAP+10*1024**3
    plan=dict(kind='byte_context_transform_quality_v1',sources={str(p.relative_to(ROOT)):sha(p) for p in paths},
              runtime=runtime(),assets=old['assets'],codebook_assets=old['codebook_assets'],specs=specs,
              data=read(ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json')['data'],
              bootstrap_seed=62102400007,prior_role_matches=[],primary_min_gain=.02,bootstrap_quantile=1-.05/6,
              cpu_mps_mean_ce_atol=3e-5,cpu_mps_target_ce_atol=3e-4,output_cap=CAP,reserve=8*1024**3,
              extra_headroom=2*1024**3,free_at_freeze=free,per_trial_seconds=3600,total_seconds=36000,
              forecast=dict(warmed_step_max_seconds=warmed,per_trial_seconds=forecast,
                            total_quality_seconds=3*sum(forecast.values()),method='1.5x warm update maximum plus1200s per trial for scoring/serialization'),
              holdout_scored=False,pilot_plan_sha256=sha(BASE/'pilot-plan.json'),pilot_result_sha256=sha(p/'result.json'),
              source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    WEIGHTS.mkdir();put(BASE/'quality-plan.json',plan)
    print(json.dumps(dict(plan_sha256=sha(BASE/'quality-plan.json'),forecast=plan['forecast'],free=free)),flush=True)


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


def score(model,view,device):
    import torch
    from torch.nn import functional as F
    was_training=model.training;model.eval();losses=[]
    sampler=view.generators['train'].get_state().clone();tick=time.perf_counter()
    with torch.no_grad():
        for start in range(0,34278,128):
            count=min(128,34278-start)
            ids=view.val_data[start:start+count].unsqueeze(0).to(device)
            targets=view.val_data[start+1:start+count+1].to(device)
            logits,_=model(ids)
            values=F.cross_entropy(logits[0],targets,reduction='none').cpu().double()
            assert bool(torch.isfinite(values).all()) and bool((values>=0).all())
            losses.extend(values.tolist())
    model.train(was_training)
    assert torch.equal(sampler,view.generators['train'].get_state()) and len(losses)==34278
    return dict(ce=math.fsum(losses)/34278,per_target_losses=losses,target_count=34278,
                seconds=time.perf_counter()-tick,holdout_scored=False,device=device)


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
            sample['retained_bytes']=sum(p.stat().st_size for root in (BASE,WEIGHTS) for p in root.rglob('*') if p.is_file())
            if sample['retained_bytes']>CAP:reason='output_limit'
            if (BASE/'STOP').exists():reason='explicit_stop'
            return sample,reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        used=sum(p.stat().st_size for root in (BASE,WEIGHTS) for p in root.rglob('*') if p.is_file())
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
    for start in (0,3,6):assert len({r['train_sampler_sha256'] for r in rows[start:start+3]})==1
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
    for index in range(9):
        assert not (BASE/f'{index:02d}').exists(), 'No implicit replay'
        remaining=plan['total_seconds']-(time.monotonic()-started)
        assert remaining>=plan['per_trial_seconds']
        child=subprocess.run([sys.executable,'-m','research.byte_context_transform_run_v1','worker',str(index)],cwd=ROOT)
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
