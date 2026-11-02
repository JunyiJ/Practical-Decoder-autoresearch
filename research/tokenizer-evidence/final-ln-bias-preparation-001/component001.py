"""Frozen generated final-LayerNorm bias arithmetic and incumbent inspection."""
import gc
import hashlib
import json
import math
from pathlib import Path
import torch
from torch import nn
from research.final_ln_bias_v1.component import FinalLayerNorm, KINDS
from research.adamw_beta2_worker_v1 import validation as v
from research.adamw_beta2_quality_v1.support import idle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock

B = Path(__file__).resolve().parent
ROOT = B.parents[2]


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def ref(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))


def save(name, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    assert len(raw) < 1024**2
    assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file()) + len(raw) < 8*1024**2
    with (B/name).open('xb') as f:
        f.write(raw)


def generated(shape, offset):
    x = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    return torch.sin(x*1.137+offset) + .3*torch.cos(x*.731+offset) + .4


def error(x, y):
    return float((x.detach().cpu().double()-y.detach().cpu().double()).abs().max())


def oracle(x, w, b):
    z = x-x.sum(-1, keepdim=True)/x.shape[-1]
    return z/(z.square().sum(-1, keepdim=True)/x.shape[-1]+1e-5).sqrt()*w+b


torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle()
    assert not (B/'component001-plan.json').exists()
    old = ROOT/'research/tokenizer-evidence/adamw-beta2-preparation-001'
    prior = json.loads((old/'quality001-plan.json').read_text())['sources']
    for path, digest in prior.items():
        assert sha(ROOT/path) == digest
    closure = json.loads((old/'quality001-closure.json').read_text())
    assert closure['status'] == 'closed_not_adopted'
    anchor = json.loads((old/'quality001-accepted-04.json').read_text())
    binding = {k:next(f for f in anchor['files'] if Path(f['path']).name == name)
               for k, name in [('config','config.json'), ('h200','h200.pt')]}
    binding['parent'] = next(f for f in anchor['files'] if Path(f['path']).name.startswith('final_'))
    files = [Path(__file__), ROOT/'research/final_ln_bias_v1/__init__.py', ROOT/'research/final_ln_bias_v1/component.py']
    sources = {str(p.relative_to(ROOT)):sha(p) for p in files}
    plan = dict(sources=sources, predecessors=prior, decision=ref(B/'direction-decision.json'),
        closure=ref(old/'quality001-closure.json'), accepted_anchor=ref(old/'quality001-accepted-04.json'), binding=binding,
        cases=[dict(device=d,dtype=t,batch=b,width=w,kind=k) for d,t in [('cpu','float64'),('cpu','float32'),('mps','float32')]
               for b in [1,3] for w in [7,256] for k in KINDS], tokens=9, prefix=5,
        normwise_tolerances=dict(float64=1e-10,float32=5e-5),
        oracle='CPU float64 explicit mean/variance equation; max error / max(1, max oracle magnitude). Exact same-backend native forward/all-gradient control in addition.',
        finite_difference=dict(step=1e-6,atol=1e-7,rtol=1e-5),
        output_cap_bytes=8*1024**2, training_allowed=False, scientific_seeds_used=False,
        scope='Generated tokenwise LayerNorm only, plus read-only accepted incumbent inspection. No candidate full-model/optimizer/EMA/cache or scientific quality evidence.')
    save('component001-plan.json', plan)
    rows=[]; observation=None; failure=None
    try:
        with StrictMpsGuard(B/'component001-guard.json', ROOT):
            model,data,cfg=v.load(binding,'cpu')
            norm=model.transformer.ln_f
            assert type(norm) is nn.LayerNorm and norm.eps == 1e-5
            assert norm.weight.shape == norm.bias.shape == (256,)
            observation=dict(binding=binding,final_type=type(norm).__name__,epsilon=norm.eps,
                scale_parameters=norm.weight.numel(),bias_parameters=norm.bias.numel(),
                total_parameters=sum(p.numel() for p in model.parameters()),
                training_updates=0,holdout_scored=False,original_weights_unchanged=True)
            assert observation['total_parameters']==28492416
            del model,data,cfg,norm;gc.collect()
            for case in plan['cases']:
                device=case['device'];dtype=getattr(torch,case['dtype']);width=case['width'];kind=case['kind']
                rng=torch.get_rng_state().clone()
                m=FinalLayerNorm(width,kind).to(device=device,dtype=dtype)
                twin=FinalLayerNorm(width,KINDS[1] if kind==KINDS[0] else KINDS[0]).to(device=device,dtype=dtype)
                assert torch.equal(rng,torch.get_rng_state()) and torch.equal(m.weight,twin.weight)
                x=generated((case['batch'],9,width),.2).to(device=device,dtype=dtype).requires_grad_()
                assert torch.equal(m(x),twin(x))
                assert sum(p.numel() for p in m.parameters()) == width*(2 if kind=='learned' else 1)
                assert set(m.state_dict()) == ({'weight','bias'} if kind=='learned' else {'weight'})
                assert kind=='learned' or m.bias is None
                with torch.no_grad():
                    m.weight.copy_(generated((width,),.7))
                    if m.bias is not None:m.bias.copy_(generated((width,),1.3)*.1)
                xr=x.detach().cpu().double().requires_grad_()
                w=m.weight.detach().cpu().double().requires_grad_()
                b=m.bias.detach().cpu().double().requires_grad_() if m.bias is not None else torch.zeros(width,dtype=torch.float64)
                params=(x,m.weight,m.bias) if m.bias is not None else (x,m.weight)
                refs=(xr,w,b) if m.bias is not None else (xr,w)
                probe=generated(tuple(x.shape),.8).to(device=device,dtype=dtype)
                actual=m(x);expected=oracle(xr,w,b)
                ga=torch.autograd.grad((actual*probe).sum(),params,retain_graph=True)
                gb=torch.autograd.grad((expected*probe.cpu().double()).sum(),refs)
                tol=plan['normwise_tolerances'][case['dtype']]
                errors=[error(a,z)/max(1.,float(z.detach().abs().max())) for a,z in zip((actual,*ga),(expected,*gb))]
                assert max(errors)<=tol and all(torch.isfinite(a).all() for a in (actual,*ga))
                native=nn.LayerNorm(width,eps=1e-5,bias=(kind=='learned')).to(device=device,dtype=dtype)
                native.load_state_dict(m.state_dict());nx=x.detach().clone().requires_grad_();ny=native(nx)
                np=(nx,native.weight,native.bias) if native.bias is not None else (nx,native.weight)
                ng=torch.autograd.grad((ny*probe).sum(),np)
                assert torch.equal(actual,ny) and all(torch.equal(a,z) for a,z in zip(ga,ng))
                bias_error=None
                if m.bias is not None:
                    bias_error=error(ga[-1],probe.sum((0,1)))
                    assert bias_error<=tol*max(1.,float(probe.sum((0,1)).abs().max()))
                future=torch.autograd.grad(actual[:,:5].square().sum(),x)[0][:,5:]
                assert torch.count_nonzero(future)==0
                changed=x.detach().clone();changed[:,5:]+=3.7
                assert torch.equal(m(changed)[:,:5],actual[:,:5])
                fd=[]
                if dtype==torch.float64:
                    for index,p in enumerate(params):
                        with torch.no_grad():
                            original=p.view(-1)[0].item();p.view(-1)[0]=original+1e-6;plus=(m(x)*probe).sum().item()
                            p.view(-1)[0]=original-1e-6;minus=(m(x)*probe).sum().item();p.view(-1)[0]=original
                        numerical=(plus-minus)/2e-6;analytic=ga[index].view(-1)[0].item()
                        assert abs(numerical-analytic)<=1e-7+1e-5*abs(analytic)
                        fd.append(abs(numerical-analytic))
                rows.append(dict(**case,forward_error=error(actual,expected),gradient_error=max(error(a,z) for a,z in zip(ga,gb)),
                    max_normwise_error=max(errors),bias_sum_error=bias_error,finite_difference_errors=fd,
                    initial_function_exact=True,shared_scale_exact=True,construction_rng_unchanged=True,
                    native_forward_all_gradients_exact=True,exact_suffix_independence=True,exact_future_gradient_zero=True,
                    trainable_parameters=sum(p.numel() for p in m.parameters())))
                del m,twin,x,xr,w,b,actual,expected,ga,gb,probe,changed,future,native,nx,ny,ng,params,refs,np
                if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
            for path,digest in {**prior,**sources}.items():assert sha(ROOT/path)==digest
            v.artifacts(binding)
    except BaseException as exc:
        failure=repr(exc)
    result=dict(status='passed' if failure is None else 'failed',error=failure,cases=rows,
        incumbent_observation=observation,plan=ref(B/'component001-plan.json'),guard=ref(B/'component001-guard.json'),
        predecessor_sources_unchanged=len(prior),source_count=len(prior)+len(sources),
        full_model_integrated=False,scientific_training_launched=False,quality_adopted=False,no_retry=True,scope=plan['scope'])
    save('component001-result.json',result)
    print(json.dumps(dict(status=result['status'],cases=len(rows),error=failure,incumbent=observation)))
    if failure:raise RuntimeError(failure)
