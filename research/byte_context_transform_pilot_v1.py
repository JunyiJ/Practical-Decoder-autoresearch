"""Finite source-bound CPU/MPS equivalence and resource pilot; no quality panel."""
import argparse
import fcntl
import gc
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/tokenizer-evidence/byte-context-transform-001'
TORCH = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(1024**2): h.update(block)
    return h.hexdigest()


def put(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())


def read(path): return json.loads(Path(path).read_text())


def runtime():
    import torch, tiktoken
    return dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
                torch_git=torch.version.git_version, tiktoken=tiktoken.__version__)


def pins(plan):
    for path, digest in plan['sources'].items(): assert sha(ROOT/path) == digest, path
    for row in plan['assets']: assert sha(row['path']) == row['sha256']
    assert runtime() == plan['runtime']


def prepare():
    assert not BASE.exists()
    result = subprocess.run(['rg','-l','6210210000[1-6]','research','research_state','-g','*.json','-g','*.jsonl'],
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1 and not result.stdout, ('Existing roles',result.stdout,result.stderr)
    old = read(ROOT/'research/tokenizer-evidence/byte-composition-boundary-001/quality001/plan.json')
    paths = sorted(p for folder in ('research','src') for p in (ROOT/folder).rglob('*.py'))
    paths.append(ROOT/'research/BYTE_CONTEXT_TRANSFORM_PROTOCOL.md')
    for relative in ('research/tokenizer-evidence/byte-composition-boundary-001/bulk-hash-adoption001/recipe.json',
                     'research/tokenizer-evidence/readout-quality-preparation-001/readout-quality001-primary-statistics.json',
                     'research/tokenizer-evidence/readout-quality-preparation-001/readout-quality001-selection.json',
                     'research/tokenizer-evidence/shared-table-geometry-001/result.json'):
        paths.append(ROOT/relative)
    plan = dict(kind='byte_context_transform_pilot_v1', sources={str(p.relative_to(ROOT)):sha(p) for p in paths},
        assets=old['assets'], codebook_assets=old['codebook_assets'],runtime=runtime(),
        roles=dict(zip(('native_initializer','composer_initializer','nonlexical_key','train_sampler','validation_sampler'),range(62102100001,62102100006))),
        projection_seed=62102100006, prior_role_matches=[], steps=4, learning_rate=.0025,
        cpu_mps_mean_ce_atol=3e-5,cpu_mps_target_ce_atol=3e-4,
        projection_gradient_atol=2e-6,projection_gradient_rtol=2e-4,
        limits=dict(seconds=1800,rss=4*1024**3,mps=6*1024**3,output=8*1024**2,reserve=8*1024**3),
        proposed_full_output_cap=int(1.5*1024**3),extra_headroom=2*1024**3,
        free_at_prepare=shutil.disk_usage(ROOT).free,quality_panel_admitted=False,holdout_scored=False)
    assert plan['free_at_prepare'] >= plan['proposed_full_output_cap']+10*1024**3
    BASE.mkdir();put(BASE/'pilot-plan.json',plan)
    print(json.dumps(dict(plan_sha256=sha(BASE/'pilot-plan.json'),source_count=len(paths),free=plan['free_at_prepare'])),flush=True)


def sync(device):
    if device == 'mps': TORCH.mps.synchronize()


def clean():
    gc.collect()
    if TORCH.backends.mps.is_available(): TORCH.mps.synchronize();TORCH.mps.empty_cache()


def compare_tree(a,b):
    import torch
    if isinstance(a,torch.Tensor): assert torch.equal(a.cpu(),b.cpu())
    elif isinstance(a,dict):
        assert list(a)==list(b)
        for key in a: compare_tree(a[key],b[key])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b): compare_tree(x,y)
    else: assert a==b,(a,b)


def parameters(model): return model.ordered_parameters()


def objects(model, official=False):
    from omegaconf import OmegaConf
    from research.byte_input_model_v6 import ByteInputOptimizer
    from research.byte_context_transform_v1 import Optimizer
    from research.byte_input_training_v6 import recipe, CanonicalParameters
    from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
    from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
    cfg=OmegaConf.create(recipe(4,.0025))
    opt=ByteInputOptimizer(model,.0025) if official else Optimizer(model)
    return opt,LearningRateSchedule(opt,cfg),TailWeightAverages(CanonicalParameters(model),cfg)


def one_step(model, optimizer, schedule, averages, x,y,step, official=False):
    from research.gradient_clip_model_v1.clipping import clip_gradients
    optimizer.zero_grad();schedule.apply(step-1);model.train()
    if official:
        result=model.forward_train(x,y);logits=result.primary.values;loss=result.loss
    else: logits,loss=model(x,y)
    assert math.isfinite(float(loss.detach()))
    loss.backward()
    observed=clip_gradients((p for _,p in parameters(model)),1.)
    assert observed['gradient_tensors']==len(parameters(model))
    optimizer.step();averages.update(step);optimizer.zero_grad()
    return dict(step=step,loss=float(loss.detach()),**observed)


def analytic_projection(device,plan):
    import torch
    from research.byte_context_transform_v1 import transform
    # Fixed literals: no scientific RNG; full D but small context count.
    z=(torch.arange(3*256,dtype=torch.float32,device=device).reshape(3,256)%29-14)/13
    a=(torch.arange(256*256,dtype=torch.float32,device=device).reshape(256,256)%31-15)/160
    g=(torch.arange(3*256,dtype=torch.float32,device=device).reshape(3,256)%17-8)/19
    rows=[]
    for arm in ('linear','tanh'):
        x=z.clone().requires_grad_();w=a.clone().requires_grad_()
        h=transform(x,w,arm);(h*g).sum().backward()
        u=z@a.T; expected=u if arm=='linear' else u.tanh()
        grad_u=g if arm=='linear' else g*(1-expected.square())
        for got,want in ((h,expected),(x.grad,grad_u@a),(w.grad,grad_u.T@z)):
            torch.testing.assert_close(got,want,atol=plan['projection_gradient_atol'],rtol=plan['projection_gradient_rtol'])
        rows.append(dict(arm=arm,input_gradient_max_error=float((x.grad-grad_u@a).abs().max()),
                         weight_gradient_max_error=float((w.grad-grad_u.T@z).abs().max())))
    return rows


def pilot(plan,out):
    import torch
    from torch.nn import functional as F
    from research.byte_context_transform_v1 import construct,weights,load_weights
    from research.byte_input_model_v6 import ByteInputModel
    from research.byte_input_model_v3 import Spec,build_codebook
    from research.byte_input_training_v6 import load_view
    book=build_codebook(*(Path(r['path']).read_bytes() for r in plan['codebook_assets']),'raw_bytes')
    roles=plan['roles'];reports=[];initial_ce={};initial_targets={};role_digests={}
    for device in ('cpu','mps'):
        analytic=analytic_projection(device,plan)
        view=load_view(roles,device);x,y=view.get_batch('train')
        # Genuine adopted baseline reference, same explicit initialization role.
        mps=torch.mps.get_rng_state().clone()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(roles['native_initializer'])
            reference=ByteInputModel(Spec('raw_bytes',roles['composer_initializer'],book.identity),book).to(device)
        torch.mps.set_rng_state(mps)
        model=construct('byte',roles,plan['projection_seed'],book,device)
        compare_tree({n:p for n,p in parameters(reference)},weights(model))
        baseline_initial={n:hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest() for n,p in parameters(reference)}
        result=reference.forward_train(x,y);result.loss.backward()
        ref_logits=result.primary.values.detach().cpu().clone()
        ref_grad={n:p.grad.detach().cpu().clone() for n,p in parameters(reference)}
        del result
        logits,loss=model(x,y);loss.backward()
        assert torch.equal(logits.detach().cpu(),ref_logits)
        compare_tree({n:p.grad for n,p in parameters(model)},ref_grad)
        initial_targets[(device,'byte')]=F.cross_entropy(logits.detach().reshape(-1,50257),y.reshape(-1),reduction='none').cpu().double()
        initial_ce[(device,'byte')]=float(initial_targets[(device,'byte')].mean())
        del logits,loss,ref_logits,ref_grad;clean()
        ref_opt,ref_sched,ref_avg=objects(reference,True)
        opt,sched,avg=objects(model)
        reference.zero_grad(set_to_none=True);model.zero_grad(set_to_none=True)
        # Identical fresh sampler states, independent of initial numerical batch.
        rv=load_view(roles,device);mv=load_view(roles,device);timings=[]
        for step in range(1,5):
            rx,ry=rv.get_batch('train');mx,my=mv.get_batch('train')
            assert torch.equal(rx,mx) and torch.equal(ry,my)
            ref_row=one_step(reference,ref_opt,ref_sched,ref_avg,rx,ry,step,True)
            tick=time.perf_counter();row=one_step(model,opt,sched,avg,mx,my,step);sync(device);timings.append(time.perf_counter()-tick)
            assert row==ref_row
            compare_tree({n:p for n,p in parameters(reference)},weights(model))
            compare_tree([c.state_dict() for c in ref_opt._optimizers],[c.state_dict() for c in opt.children])
            compare_tree(ref_avg.state_dict(),avg.state_dict())
        assert avg.snapshots==ref_avg.snapshots==3
        reports.append(dict(device=device,arm='byte',updates=4,reference_updates=4,exact_all_gradients=True,
                            exact_all_updates_optimizer_ema=True,seconds=timings,analytic=analytic))
        del reference,model,ref_opt,ref_sched,ref_avg,opt,sched,avg,rv,mv,rx,ry,mx,my;clean()
        # Paired initial common weights and private A; finite complete training.
        initial_common=None;initial_a=None
        for arm in ('linear','tanh'):
            model=construct(arm,roles,plan['projection_seed'],book,device)
            common={n:p.detach().cpu().clone() for n,p in model.base.ordered_parameters()}
            assert {n:hashlib.sha256(p.numpy().tobytes()).hexdigest() for n,p in common.items()}==baseline_initial
            if initial_common is None: initial_common=common;initial_a=model.context_weight.detach().cpu().clone()
            else:
                compare_tree(common,initial_common)
                assert torch.equal(model.context_weight.detach().cpu(),initial_a)
            del common
            logits,loss=model(x,y)
            vals=F.cross_entropy(logits.detach().reshape(-1,50257),y.reshape(-1),reduction='none').cpu().double()
            initial_targets[(device,arm)]=vals;initial_ce[(device,arm)]=float(vals.mean())
            del logits,loss
            opt,sched,avg=objects(model);view=load_view(roles,device);timings=[];rows=[]
            for step in range(1,5):
                tx,ty=view.get_batch('train');tick=time.perf_counter()
                rows.append(one_step(model,opt,sched,avg,tx,ty,step));sync(device);timings.append(time.perf_counter()-tick)
            assert opt.steps==avg.completed_steps==4 and avg.snapshots==3
            model.validate()
            payload=dict(kind='byte_context_transform_weights_v1',identity=model.identity,weights=weights(model),
                         codebook_identity=book.identity,optimizer_resume=False)
            data=io.BytesIO();tick=time.perf_counter();torch.save(payload,data);size=data.tell();data.seek(0)
            saved=torch.load(data,map_location='cpu',weights_only=True)
            assert saved['identity']==model.identity and saved['codebook_identity']==book.identity
            restored=construct(arm,roles,plan['projection_seed'],book,'cpu');load_weights(restored,saved['weights'])
            compare_tree(weights(restored),payload['weights']);serialization_seconds=time.perf_counter()-tick
            with torch.no_grad():
                trained=F.cross_entropy(model(x)[0].reshape(-1,50257),y.reshape(-1),reduction='none').cpu().double()
                restored_values=F.cross_entropy(restored(x.cpu())[0].reshape(-1,50257),y.cpu().reshape(-1),reduction='none').cpu().double()
            diagnostic=dict(mean_error=abs(float(trained.mean()-restored_values.mean())),
                            max_target_error=float((trained-restored_values).abs().max()))
            # Preserve actual failing numbers before the acceptance predicate.
            put(out/f'{device}-{arm}-roundtrip-numerical.json',diagnostic)
            assert diagnostic['mean_error']<=plan['cpu_mps_mean_ce_atol'] and diagnostic['max_target_error']<=plan['cpu_mps_target_ce_atol']
            reports.append(dict(device=device,arm=arm,updates=4,seconds=timings,rows=rows,parameters=sum(p.numel() for _,p in parameters(model)),
                                full_parameter_exposure=True,roundtrip_exact=True,serialized_bytes=size,
                                serialization_seconds=serialization_seconds,trained_cpu_mps=diagnostic))
            del model,opt,sched,avg,view,restored,payload,saved,data,tx,ty,trained,restored_values;clean()
        del initial_common,initial_a,x,y;clean()
    numerical=[]
    for arm in ('byte','linear','tanh'):
        a=initial_targets[('cpu',arm)];b=initial_targets[('mps',arm)]
        numerical.append(dict(arm=arm,mean_ce_error=abs(float(a.mean()-b.mean())),max_target_ce_error=float((a-b).abs().max())))
    put(out/'initial-numerical.json',dict(cases=numerical))
    assert all(r['mean_ce_error']<=plan['cpu_mps_mean_ce_atol'] and r['max_target_ce_error']<=plan['cpu_mps_target_ce_atol'] for r in numerical)
    put(out/'result.json',dict(passed=True,training_updates=32,cases=reports,initial_numerical=numerical,
                             validation_scored=False,holdout_scored=False,quality_panel_admitted=False,
                             identity=book.identity,scope='Four-step implementation/resource proof, not quality evidence'))


def run():
    global TORCH
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    plan=read(BASE/'pilot-plan.json');pins(plan)
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample()
            if TORCH is not None:
                sample['mps_bytes']=TORCH.mps.driver_allocated_memory()
                if sample['mps_bytes']>=plan['limits']['mps']:reason='mps_limit'
            sample['retained_bytes']=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
            if sample['retained_bytes']>=plan['limits']['output']:reason='output_limit'
            if (BASE/'STOP').exists():reason='explicit_stop'
            return sample,reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        out=BASE/'pilot';out.mkdir()
        put(out/'attempt.json',dict(pid=os.getpid(),plan_sha256=sha(BASE/'pilot-plan.json')))
        try:
            with Guard(out/'guard.json',ROOT,timeout_seconds=plan['limits']['seconds'],poll_seconds=.5):
                import torch
                TORCH=torch;torch.set_num_threads(4)
                assert torch.backends.mps.is_available() and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0' and sys.flags.optimize==0
                pilot(plan,out);pins(plan)
            put(out/'acceptance.json',dict(accepted=True,result_sha256=sha(out/'result.json'),guard_sha256=sha(out/'guard.json')))
            print(json.dumps(read(out/'result.json')),flush=True)
        except BaseException:
            put(out/'failure.json',dict(traceback=traceback.format_exc(),classification_pending=True));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('prepare','run'))
    args=parser.parse_args()
    prepare() if args.action=='prepare' else run()
