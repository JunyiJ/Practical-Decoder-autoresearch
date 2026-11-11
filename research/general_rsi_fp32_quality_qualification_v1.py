"""Bounded new optimizer integration/restart/timing check, no quality selection."""
import argparse
import copy
import fcntl
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import resource
import shutil
import statistics
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'research/tokenizer-evidence/general-rsi-composer-fp32-quality-001/qualification'
WEIGHTS = ROOT/'research_runs/general-rsi-composer-fp32-quality-qualification-001'

def put(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False);stream.write('\n')

def digest(value):
    import torch
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            h.update(str((x.dtype,tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():h.update(str(k).encode());visit(v)
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def prepare():
    import torch
    from torch.optim import _muon
    from research.general_rsi_training_v2 import sha
    prior=json.loads((ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001/plan.json').read_text())
    assert (ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001/completion-review.json').exists()
    # Bind the already audited complete source/runtime inventory and new integration.
    sources=dict(prior['sources'])
    for name in ('research/general_rsi_fp32_quality_runtime_v1.py','research/general_rsi_fp32_quality_qualification_v1.py','research/general_rsi_training_v2.py','research/general_rsi_qualification_v1.py'):
        sources[name]=sha(ROOT/name)
    sources[str(Path(inspect.getsourcefile(_muon)).resolve())]=sha(inspect.getsourcefile(_muon))
    plan=dict(kind='new_composer_fp32_quality_integration_001',sources=sources,
        torch_version=torch.__version__,codebook_assets=prior['codebook_assets'],
        roles=dict(zip(('native_initializer','composer_initializer','train_sampler','validation_sampler'),range(62111117001,62111117005))),
        seed_audit='Exact role search in research/research_state JSON and JSONL returned no matches before assignment.',
        arms=['existing_byte','byte_composer_fp32'],lr=.0025,horizon=40,restart_step=35,
        checks=['identical initial weights and optimizer groups','first-step noncomposer parameters and all momentum/Adam states exact across arms','all finite parameters/gradients/state','complete exact restart at35 and replay to40 including two EMA snapshots','nonfinite momentum and wrong-arm restoration rejected before mutation','synchronized whole-update timing'],
        arithmetic_agreement='FP32 vs CPU FP64 direction diagnostics on first clipped real-training Nesterov inputs. No cross-precision tolerance or quality veto.',
        timeout_seconds=900,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,reserve_bytes=10*1024**3,
        complete_incremental_forecast_bytes=9*1024**3,required_free_bytes=19*1024**3,
        forecast={'final_weights':12*64*1024**2,'restart_overlap':2*512*1024**2,'metadata':256*1024**2,'failed_write_allowance':5*1024**3,'qualification_retention':2*1024**3,'reserve':10*1024**3},
        numerical_qualification_only=True,test_opened=False,old_failure_unchanged=True,
        checkpoint_disposition='Retain fixture checkpoints through integration review; eligible for exact audited retirement only after downstream state qualification closure, never fresh scientific weights.')
    OUT.mkdir(parents=True);put(OUT/'plan.json',plan)
    put(OUT/'source-audit.json',dict(torch_version=torch.__version__,installed_muon_source=inspect.getsourcefile(_muon),installed_muon_sha256=sha(inspect.getsourcefile(_muon)),
        policy=dict(momentum=.95,nesterov=True,coefficients=[3.4445,-4.775,2.0315],ns_steps=5,norm='Frobenius norm clamped at1e-7',lr='base*.5*.2*sqrt(max(rows,columns))',child_weight_decay=0.,independent_decay=.00006,adam_betas=[.9,.999],adam_eps=1e-6),
        change='Nine composer Muon matrices only: replace pre-normalization BF16 cast by FP32 clone. Noncomposer Muon calls installed functional; AdamW unchanged. No batch reuse.',
        old_component_policy_difference='None in normalization/momentum/coefficients/scaling; old component was isolated and omitted enclosing model independent decay. New integration retains enclosing decay.',
        original_failure_sha256=sha(ROOT/'research/tokenizer-evidence/general-rsi-composer-fp32-fixture-001/attempt-01/failed-comparison.json')))
    print(sha(OUT/'plan.json'),flush=True)

def run(plan,binding):
    import torch
    from research.general_rsi_fp32_quality_runtime_v1 import Context, orthogonalize
    from research.general_rsi_training_v2 import sha,write_checkpoint,read_checkpoint
    from research.general_rsi_qualification_v1 import equal
    from research.byte_input_model_v6 import build_codebook
    from research.gradient_clip_model_v1.clipping import clip_gradients
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    torch.set_num_threads(4)
    raw=[]
    for asset in plan['codebook_assets']:
        assert sha(asset['path'])==asset['sha256'];raw.append(Path(asset['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes')
    def make(arm):return Context(arm,plan['lr'],40,plan['roles'],book,binding)
    # A fresh diagnostic context uses only a training batch. No validation is read.
    c=make('byte_composer_fp32');x,y=c.data.batch();c.model.forward_train(x,y).loss.backward()
    clip_gradients((p for _,p in c.model.ordered_parameters()),1.)
    names={id(p):n for n,p in c.model.ordered_parameters()};diagnostics=[]
    for group in c.optimizer._optimizers[0].param_groups:
        p=group['params'][0];name=names[id(p)]
        if not name.startswith('composer.'):continue
        grad=p.grad;buf=torch.zeros_like(grad);buf.lerp_(grad,1-group['momentum']);u=grad.lerp(buf,group['momentum'])
        a=orthogonalize(u,group['ns_coefficients'],group['ns_steps'],group['eps']).cpu().double()
        ref=orthogonalize(u.cpu().double(),group['ns_coefficients'],group['ns_steps'],group['eps'])
        diagnostics.append(dict(name=name,relative_direction_error=float((a-ref).norm()/ref.norm().clamp(min=1e-30)),maximum_absolute_error=float((a-ref).abs().max()),role='diagnostic_only'))
    assert len(diagnostics)==9;put(OUT/'fp64-diagnostics.json',diagnostics)
    del c,x,y,grad,buf,u,a,ref,p,group
    gc.collect();torch.mps.empty_cache()
    reports=[];baseline_initial=None;baseline_first=None
    for arm in plan['arms']:
        c=make(arm);initial=c.capture();initial_hash=digest(initial['parameters'])
        if baseline_initial is None:baseline_initial=initial_hash
        assert initial_hash==baseline_initial
        groups=[[(g['name'],g['options']) for g in child['groups']] for child in initial['children']]
        del initial
        durations=[]
        while c.step<35:
            torch.mps.synchronize();start=time.perf_counter();row=c.update();torch.mps.synchronize();durations.append(time.perf_counter()-start)
            assert torch.mps.driver_allocated_memory()<plan['mps_limit_bytes']
            if c.step==1:
                first=c.capture()
                if arm=='existing_byte':baseline_first=first;baseline_groups=groups
                else:
                    assert groups==baseline_groups
                    equal(first['children'],baseline_first['children'])
                    equal(first['samplers'],baseline_first['samplers'])
                    for name,v in first['parameters'].items():
                        if not (name.startswith('composer.') and name in first['children'][0]['state']):equal(v,baseline_first['parameters'][name],name)
                    composer=[n for n in first['children'][0]['state'] if n.startswith('composer.')]
                    assert len(composer)==9
                    put(OUT/'routing.json',dict(initial_weights_equal=True,groups_equal=True,first_step_all_optimizer_states_equal=True,first_step_noncomposer_parameters_equal=True,composer_matrices=composer,composer_parameter_max_abs_differences={n:float((first['parameters'][n]-baseline_first['parameters'][n]).abs().max()) for n in composer},diagnostic_only=True))
                    del baseline_first
                del first
        saved=c.capture();start=time.perf_counter();receipt=write_checkpoint(WEIGHTS/f'{arm}-step35.pt',saved);save_seconds=time.perf_counter()-start
        while c.step<40:c.update()
        expected=c.capture();assert expected['snapshots']==2 and expected['history'][-1]['lr_ratio']==.1
        del c,saved;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
        c=make(arm);loaded=read_checkpoint(receipt);start=time.perf_counter();c.restore(loaded);restore_seconds=time.perf_counter()-start
        equal(c.capture(),loaded)
        while c.step<40:c.update()
        actual=c.capture();equal(actual,expected)
        final_digest=digest(actual)
        del c,actual,expected;gc.collect();torch.mps.empty_cache()
        faults=[]
        for fault in ('nonfinite_momentum','wrong_arm'):
            broken=copy.deepcopy(loaded)
            if fault=='nonfinite_momentum':next(iter(broken['children'][0]['state'].values()))['momentum_buffer'].flatten()[0]=float('nan')
            else:broken['config']['quality_arm']='byte_composer_fp32' if arm=='existing_byte' else 'existing_byte'
            c=make(arm);before=digest(c.capture())
            try:c.restore(broken)
            except (AssertionError,ValueError):pass
            else:raise AssertionError('Invalid restoration accepted: '+fault)
            assert digest(c.capture())==before
            faults.append(fault);del c,broken;gc.collect();torch.mps.empty_cache()
        del loaded
        report=dict(arm=arm,checkpoint=receipt,initial_weights_sha256=initial_hash,final_complete_state_sha256=final_digest,exact_restore_at35=True,exact_replay_at40=True,ema_snapshots=2,invalid_restore_atomic=faults,whole_update_seconds=durations,median_update_seconds=statistics.median(durations[1:]),mean_update_seconds=statistics.mean(durations),save_seconds=save_seconds,restore_seconds=restore_seconds,rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,mps_driver_bytes=torch.mps.driver_allocated_memory())
        put(OUT/f'{arm}-result.json',report);reports.append(report);print(arm,'qualified',report['median_update_seconds'],flush=True)
    return dict(integration_qualified=True,arms=reports,fp64_diagnostic_only=True,test_opened=False,validation_scoring_repeated=False,scientific_quality_claim=False)

def main():
    from research.general_rsi_training_v2 import sha
    a=argparse.ArgumentParser();a.add_argument('action',choices=['prepare','run']);a.add_argument('binding',nargs='?');args=a.parse_args()
    if args.action=='prepare':return prepare()
    assert sha(OUT/'plan.json')==args.binding
    plan=json.loads((OUT/'plan.json').read_text())
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
        free=shutil.disk_usage(ROOT).free;assert free>=plan['required_free_bytes']
        WEIGHTS.mkdir();put(OUT/'attempt.json',dict(pid=os.getpid(),started=time.time(),binding=args.binding,free_bytes=free,required_free_bytes=plan['required_free_bytes']))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=900,rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):result=run(plan,args.binding)
            assert sum(p.stat().st_size for p in WEIGHTS.iterdir())<=2*1024**3
            for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
            put(OUT/'result.json',result);print('QUALIFIED',flush=True)
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
