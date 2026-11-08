"""Bounded real-model implementation check; no scientific quality comparison."""
import argparse
import fcntl
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from research.byte_context_transform_pilot_v1 import ROOT, sha, put, read, runtime, pins

BASE = ROOT/'research/tokenizer-evidence/byte-residual-readout-001'


def prepare():
    assert not BASE.exists()
    audit = subprocess.run(['rg','-l','6210250000[1-6]','research','research_state','-g','*.json','-g','*.jsonl'], cwd=ROOT, capture_output=True, text=True)
    assert audit.returncode == 1 and not audit.stdout and not audit.stderr
    prior = read(ROOT/'research/tokenizer-evidence/public-inference-correction-001/plan.json')
    sources = dict(prior['sources'])
    for name in ('research/byte_residual_readout_v1.py','research/byte_residual_readout_pilot_v1.py',
                 'research/byte_context_transform_v1.py','research/byte_context_transform_pilot_v1.py',
                 'research/byte_context_cpu_reference_v1.py','research/fp32_rmsnorm_inference_v1.py',
                 'src/research/cpu_prerequisite_guard.py'):
        sources[name] = sha(ROOT/name)
    old = read(ROOT/'research/tokenizer-evidence/byte-context-transform-001/quality-recovery002/quality-plan.json')
    assets = [Path(r['path']) for r in old['assets']+old['codebook_assets']]
    assets += [ROOT/'research/BYTE_RESIDUAL_READOUT_PROTOCOL.md']
    for folder, leaf in (('final-analysis001','result.json'), ('feature-diagnostic001','result.json'),
                         ('feature-diagnostic001','acceptance.json'), ('weight-dynamics001','result.json')):
        assets.append(ROOT/'research/tokenizer-evidence/byte-context-transform-001'/folder/leaf)
    plan = dict(kind='byte_residual_readout_pilot_v1', sources=sources, runtime=runtime(),
        assets=[dict(path=str(p),sha256=sha(p)) for p in dict.fromkeys(assets)], codebook_assets=old['codebook_assets'],
        roles=dict(zip(('native_initializer','composer_initializer','nonlexical_key','train_sampler','validation_sampler'),range(62102500001,62102500006))),
        projection_seed=62102500006, steps=4, learning_rate=.0025, prior_role_matches=[],
        seconds=600, rss=4*1024**3, mps=6*1024**3, output_cap=8*1024**2,
        reserve=8*1024**3, extra_headroom=2*1024**3,
        gradient_atol=2e-6, gradient_rtol=2e-4, mean_ce_atol=3e-5, target_ce_atol=3e-4,
        fp64_ce_atol=3e-5, quality_panel_admitted=False, holdout_scored=False)
    assert shutil.disk_usage(ROOT).free >= plan['reserve']+plan['extra_headroom']+plan['output_cap']
    pins(plan); BASE.mkdir(); put(BASE/'pilot-plan.json',plan)
    print(json.dumps(dict(plan_sha256=sha(BASE/'pilot-plan.json'),source_count=len(sources))),flush=True)


def digest(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def analytic(device, plan):
    import torch
    from research.byte_residual_readout_v1 import transform
    z=(torch.arange(3*256,dtype=torch.float32,device=device).reshape(3,256)%29-14)/13
    a=(torch.arange(256*256,dtype=torch.float32,device=device).reshape(256,256)%31-15)/160
    g=(torch.arange(3*256,dtype=torch.float32,device=device).reshape(3,256)%17-8)/19
    rows=[]
    for arm in ('residual_linear','residual_tanh'):
        for alpha in (0., .25):
            x=z.clone().requires_grad_(); w=a.clone().requires_grad_(); gate=torch.tensor(alpha,device=device,requires_grad=True)
            actual=transform(x,w,gate,arm); (actual*g).sum().backward()
            u=z@a.T; branch=u if arm=='residual_linear' else u.tanh()
            q=g if arm=='residual_linear' else g*(1-branch.square())
            wants=(z+alpha*branch,g+alpha*(q@a),alpha*(q.T@z),(g*branch).sum())
            errors=[]
            for got,want in zip((actual,x.grad,w.grad,gate.grad),wants):
                errors.append(float((got.detach()-want).abs().max()))
                torch.testing.assert_close(got,want,atol=plan['gradient_atol'],rtol=plan['gradient_rtol'])
            if alpha==0:
                assert torch.equal(actual,z) and torch.equal(x.grad,g) and torch.count_nonzero(w.grad)==0
            rows.append(dict(arm=arm,alpha=alpha,max_errors=errors))
    return rows


def endpoint(model, values, book, plan, arm, name, out):
    import torch
    from torch.nn import functional as F
    from research.byte_residual_readout_v1 import construct
    from research.byte_context_transform_v1 import load_weights
    from research.byte_context_cpu_reference_v1 import cpu_cross_entropy
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    from research.byte_input_training_v6 import load_view
    payload=dict(kind='byte_residual_readout_weights_v1',identity=model.identity,
                 weights=values,codebook_identity=book.identity,optimizer_resume=False,endpoint=name)
    data=io.BytesIO(); torch.save(payload,data); size=data.tell(); data.seek(0)
    saved=torch.load(data,map_location='cpu',weights_only=True)
    assert saved['identity']==model.identity and saved['codebook_identity']==book.identity
    restored=construct(arm,plan['roles'],plan['projection_seed'],book,'cpu')
    load_weights(restored,saved['weights']); restored.eval()
    expected={n:digest(v) for n,v in values.items()}
    assert expected=={n:digest(p) for n,p in restored.ordered_parameters()}
    view=load_view(plan['roles'],'cpu'); x=view.train_data[:128].unsqueeze(0); y=view.train_data[1:129]
    losses={}; fp64_error=None
    for device in ('cpu','mps'):
        restored.to(device)
        with torch.no_grad(),FP32RMSNormInference() as correction:
            logits=restored(x.to(device))[0][0]
        assert correction.corrected_calls==8
        if device=='cpu':
            v=cpu_cross_entropy(logits,y)
            ref=F.cross_entropy(logits.double(),y,reduction='none')
            fp64_error=float((v.double()-ref).abs().max())
        else:
            v=F.cross_entropy(logits,y.to(device),reduction='none')
        assert torch.isfinite(v).all()
        losses[device]=v.detach().cpu().double()
        assert expected=={n:digest(p) for n,p in restored.ordered_parameters()}
        del logits,v
    row=dict(arm=arm,endpoint=name,serialized_bytes=size,roundtrip_exact=True,
        mean_ce_error=abs(float(losses['cpu'].mean()-losses['mps'].mean())),
        max_target_ce_error=float((losses['cpu']-losses['mps']).abs().max()),fp64_max_ce_error=fp64_error)
    put(out/f'{model.base.validate().type}-{arm}-{name}-numerical.json',row)
    assert row['mean_ce_error']<=plan['mean_ce_atol'] and row['max_target_ce_error']<=plan['target_ce_atol']
    assert fp64_error<=plan['fp64_ce_atol']
    return row


def pilot(plan,out):
    import torch
    from omegaconf import OmegaConf
    from research.byte_residual_readout_v1 import ARMS,construct,Optimizer
    from research.byte_context_transform_v1 import weights
    from research.byte_input_model_v3 import build_codebook
    from research.byte_input_training_v6 import load_view,recipe,CanonicalParameters
    from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
    from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
    from research.gradient_clip_model_v1.clipping import clip_gradients
    def clean():
        gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()
    book=build_codebook(*(Path(r['path']).read_bytes() for r in plan['codebook_assets']),'raw_bytes')
    global_before=(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
    reports=[]; analytical={}; paired_terminal=None; paired_batches=None
    for device in ('cpu','mps'):
        analytical[device]=analytic(device,plan)
        diagnostic_view=load_view(plan['roles'],device); x,y=diagnostic_view.get_batch('train')
        reference=construct('byte',plan['roles'],plan['projection_seed'],book,device)
        initial_hashes={n:digest(p) for n,p in reference.ordered_parameters()}
        official=reference.base.forward_train(x,y); official.loss.backward()
        reference_logits=official.primary.values.detach().cpu().clone()
        reference_grads={n:p.grad.detach().cpu().clone() for n,p in reference.ordered_parameters()}
        del official,reference; clean(); initial_a=None
        for arm in ARMS:
            model=construct(arm,plan['roles'],plan['projection_seed'],book,device)
            assert initial_hashes=={n:digest(p) for n,p in model.base.ordered_parameters()}
            if arm!='byte':
                assert float(model.context_gate.detach())==0
                if initial_a is None: initial_a=digest(model.context_weight)
                else: assert initial_a==digest(model.context_weight)
            logits,loss=model(x,y); loss.backward()
            assert torch.equal(logits.detach().cpu(),reference_logits)
            for n,p in model.base.ordered_parameters(): assert torch.equal(p.grad.detach().cpu(),reference_grads[n]),n
            if arm!='byte':
                assert torch.count_nonzero(model.context_weight.grad)==0 and float(model.context_gate.grad)!=0
            del logits,loss; model.zero_grad(set_to_none=True)
            opt=Optimizer(model,plan['learning_rate']); cfg=OmegaConf.create(recipe(4,plan['learning_rate']))
            schedule=LearningRateSchedule(opt,cfg); averages=TailWeightAverages(CanonicalParameters(model),cfg)
            view=load_view(plan['roles'],device); validation_before=digest(view.generators['val'].get_state())
            rows=[]; batches=[]; manual_average=None
            for step in range(1,5):
                tx,ty=view.get_batch('train'); batches.append((digest(tx),digest(ty)))
                opt.zero_grad(); schedule.apply(step-1); tick=time.perf_counter()
                logits,loss=model(tx,ty); loss.backward()
                row=dict(step=step,loss=float(loss.detach()))
                if arm!='byte':
                    row.update(gate_before=float(model.context_gate.detach()),gate_gradient=float(model.context_gate.grad),
                               matrix_gradient_norm=float(model.context_weight.grad.detach().norm()))
                    assert (row['matrix_gradient_norm']==0) if step==1 else (row['matrix_gradient_norm']>0)
                    assert row['gate_gradient']!=0
                row.update(clip_gradients((p for _,p in model.ordered_parameters()),1.))
                opt.step(); averages.update(step); opt.zero_grad()
                if step==2: manual_average=weights(model)
                elif step>2:
                    for n,p in model.ordered_parameters(): manual_average[n].lerp_(p.detach().cpu(),1.-2.**(-1/200))
                if arm!='byte':
                    row['gate_after']=float(model.context_gate.detach()); assert row['gate_after']!=0
                if device=='mps': torch.mps.synchronize()
                row['seconds']=time.perf_counter()-tick; rows.append(row)
                del logits,loss
            assert opt.steps==averages.completed_steps==4 and averages.snapshots==3
            assert sum(len(child.state) for child in opt.children)==len(model.ordered_parameters())
            actual_average={n:v.detach().cpu().clone() for n,v in averages.averaged_parameters(200).items()}
            assert list(actual_average)==[n for n,p in model.ordered_parameters()]
            for n in actual_average:
                torch.testing.assert_close(actual_average[n],manual_average[n],atol=2e-7,rtol=2e-6)
            terminal=digest(view.generators['train'].get_state())
            assert validation_before==digest(view.generators['val'].get_state())
            if paired_terminal is None: paired_terminal=terminal; paired_batches=batches
            else: assert paired_terminal==terminal and paired_batches==batches
            endpoints=[]
            for name,values in (('raw',weights(model)),('h200',actual_average)):
                endpoints.append(endpoint(model,values,book,plan,arm,name,out)); clean()
            row=dict(device=device,arm=arm,initial_common_weights_and_gradients_exact=True,
                initial_logits_exact=True,steps=rows,optimizer_all_parameters=True,h200_all_parameters=True,
                snapshots=3,terminal_sampler_sha256=terminal,batch_hashes=batches,endpoints=endpoints,
                parameters=sum(p.numel() for _,p in model.ordered_parameters()))
            reports.append(row); put(out/f'{device}-{arm}.json',row)
            print(json.dumps(dict(device=device,arm=arm,completed_updates=4)),flush=True)
            del model,opt,schedule,averages,view,manual_average,actual_average,values,tx,ty; clean()
        del reference_logits,reference_grads,x,y,diagnostic_view; clean()
    assert global_before==(digest(torch.random.get_rng_state()),digest(torch.mps.get_rng_state()))
    pins(plan)
    put(out/'result.json',dict(passed=True,training_updates=24,cases=reports,analytic=analytical,
        global_rng_unchanged=True,validation_scored=False,holdout_scored=False,quality_panel_admitted=False,
        scope='Four-step implementation and resource pilot; no quality ranking or retained weights'))


def run(expected):
    import torch
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    assert sha(BASE/'pilot-plan.json')==expected
    plan=read(BASE/'pilot-plan.json'); pins(plan)
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample()
            sample['mps_bytes']=torch.mps.driver_allocated_memory()
            sample['output_bytes']=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
            if sample['mps_bytes']>=plan['mps']: reason='mps_limit'
            if sample['output_bytes']>=plan['output_cap']: reason='output_limit'
            if (BASE/'STOP').exists(): reason='explicit_stop'
            return sample,reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free>=plan['reserve']+plan['extra_headroom']+plan['output_cap']
        out=BASE/'pilot'; out.mkdir(); put(out/'attempt.json',dict(pid=os.getpid(),plan_sha256=expected))
        try:
            with Guard(out/'guard.json',ROOT,timeout_seconds=plan['seconds'],rss_limit_bytes=plan['rss'],reserve_bytes=plan['reserve'],poll_seconds=.5):
                torch.set_num_threads(4)
                assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
                assert __debug__
                pilot(plan,out)
            put(out/'acceptance.json',dict(accepted=True,result_sha256=sha(out/'result.json'),guard_sha256=sha(out/'guard.json')))
        except BaseException:
            put(out/'failure.json',dict(traceback=traceback.format_exc())); raise


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('prepare','run')); p.add_argument('plan_sha',nargs='?'); a=p.parse_args()
    prepare() if a.action=='prepare' else run(a.plan_sha)
