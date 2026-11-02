"""Read-only incumbent inspection and generated normalization arithmetic checks."""
import gc, hashlib, json, math
from pathlib import Path
import torch
from torch import nn
from research.readout_centering_v1.component import ReadoutNormalization, KINDS
from research.gradient_clip_quality_v2.quality_execution import idle
from research.gradient_clip_worker_v1 import validation as v
from research.gradient_clip_model_v1.integration import build_optimizer
from research.branch_scale_model_v1.runtime.utils.checkpoint import read_checkpoint
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock

B = Path(__file__).resolve().parent
ROOT = B.parents[2]

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(name, value):
    raw = (json.dumps(value, indent=2) + '\n').encode()
    assert len(raw) < 1024**2
    assert sum(p.stat().st_size for p in B.rglob('*') if p.is_file()) + len(raw) < 8*1024**2
    with (B/name).open('xb') as f:
        f.write(raw)

def ref(p):
    return dict(path=str(Path(p).resolve()), sha256=sha(p))

def generated(shape, offset):
    a = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    return torch.sin(a*.137+offset) + .4

def err(a, b):
    return float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max())

def oracle(x, w, b, kind):
    z = x - x.sum(-1, keepdim=True)/x.shape[-1] if kind == 'layer_affine' else x
    return z / (z.pow(2).sum(-1, keepdim=True)/x.shape[-1]+1e-5).sqrt() * w + b

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle()
    assert not (B/'component001-plan.json').exists()
    old = ROOT/'research/tokenizer-evidence/gradient-clip-preparation-001'
    previous = json.loads((old/'quality002-plan.json').read_text())['sources']
    for p, h in previous.items(): assert sha(ROOT/p) == h
    anchor = json.loads((old/'quality002-accepted-04.json').read_text())
    binding = {k:next(f for f in anchor['files'] if Path(f['path']).name == name)
               for k, name in [('config','config.json'),('h200','h200.pt')]}
    binding['parent'] = next(f for f in anchor['files'] if Path(f['path']).name.startswith('final_'))
    files = [ROOT/'research/readout_centering_v1/__init__.py', ROOT/'research/readout_centering_v1/component.py', Path(__file__)]
    sources = {str(p.relative_to(ROOT)):sha(p) for p in files}
    plan = dict(sources=sources, predecessors=previous, decision=ref(B/'direction-decision.json'),
        accepted_anchor=ref(old/'quality002-accepted-04.json'), binding=binding,
        cases=[dict(device=d,dtype=t,batch=b,width=w,kind=k) for d,t in [('cpu','float64'),('cpu','float32'),('mps','float32')] for b in [1,3] for w in [7,256] for k in KINDS],
        tokens=9, prefix=5, tolerances=dict(float64=1e-10,float32=5e-5),
        finite_difference=dict(step=1e-6,atol=1e-7,rtol=1e-5),
        output_cap_bytes=8*1024**2, training_allowed=False, scientific_seeds_used=False,
        scope='Read-only accepted incumbent inspection plus standalone tokenwise readout normalization. No candidate full-model/optimizer/cache integration or quality evidence.')
    save('component001-plan.json',plan)
    rows=[]; observation=None; failure=None
    try:
        with StrictMpsGuard(B/'component001-guard.json',ROOT):
            model,data,cfg = v.load(binding,'cpu')
            norm = model.transformer.ln_f
            assert type(norm) is nn.LayerNorm and norm.eps == 1e-5
            assert norm.weight.shape == norm.bias.shape == (256,)
            optimizer = build_optimizer(model,cfg)
            adam = [o for o in optimizer.optimizers if type(o) is torch.optim.AdamW]
            assert len(adam)==1 and all(g['betas']==(.9,.999) for g in adam[0].param_groups)
            raw = read_checkpoint(binding['parent']['path'])
            saved_adam = [g for o in raw['optimizer_state']['optimizers'] for g in o['param_groups'] if 'betas' in g]
            assert len(saved_adam)==len(adam[0].param_groups) and all(tuple(g['betas'])==(.9,.999) for g in saved_adam)
            observation = dict(binding=binding, final_type=type(norm).__name__,epsilon=norm.eps,
                weight_shape=list(norm.weight.shape),bias_shape=list(norm.bias.shape),
                affine_parameters=sum(p.numel() for p in norm.parameters()),
                total_parameters=sum(p.numel() for p in model.parameters()),
                saved_adam_groups=len(saved_adam),saved_adam_betas=[.9,.999],
                original_weights_unchanged=True,training_updates=0,holdout_scored=False)
            del model,data,cfg,norm,optimizer,adam,raw,saved_adam;gc.collect()
            for case in plan['cases']:
                device=case['device'];dtype=getattr(torch,case['dtype']);width=case['width'];kind=case['kind']
                rng=torch.get_rng_state().clone()
                m=ReadoutNormalization(width,kind).to(device=device,dtype=dtype)
                twin=ReadoutNormalization(width,KINDS[1] if kind==KINDS[0] else KINDS[0]).to(device=device,dtype=dtype)
                assert torch.equal(rng,torch.get_rng_state())
                assert all(torch.equal(v,twin.state_dict()[n]) for n,v in m.state_dict().items())
                # Nontrivial affine values ensure bias/scale gradients are exercised.
                with torch.no_grad():
                    m.weight.copy_(generated((width,),.7));m.bias.copy_(generated((width,),1.3)*.1)
                twin.load_state_dict(m.state_dict())
                x=generated((case['batch'],9,width),.2).to(device=device,dtype=dtype).requires_grad_()
                xr=x.detach().cpu().double().requires_grad_()
                w=m.weight.detach().cpu().double().requires_grad_();b=m.bias.detach().cpu().double().requires_grad_()
                probe=generated(tuple(x.shape),.8).to(device=device,dtype=dtype)
                actual=m(x);expected=oracle(xr,w,b,kind)
                ga=torch.autograd.grad((actual*probe).sum(),(x,m.weight,m.bias),retain_graph=True)
                gb=torch.autograd.grad((expected*probe.cpu().double()).sum(),(xr,w,b))
                tol=plan['tolerances'][case['dtype']]
                torch.testing.assert_close(actual.cpu().double(),expected,rtol=tol,atol=tol)
                for a,z in zip(ga,gb):
                    assert torch.isfinite(a).all()
                    torch.testing.assert_close(a.cpu().double(),z,rtol=tol,atol=tol)
                future=torch.autograd.grad(actual[:,:5].square().sum(),x)[0][:,5:]
                assert torch.count_nonzero(future)==0
                changed=x.detach().clone();changed[:,5:]+=3.7
                assert torch.equal(m(changed)[:,:5],actual[:,:5])
                gap=err(actual,twin(x.detach()));assert gap>1e-4
                if kind=='layer_affine':
                    legacy=nn.LayerNorm(width,eps=1e-5).to(device=device,dtype=dtype)
                    legacy.load_state_dict(m.state_dict());assert torch.equal(actual,legacy(x));del legacy
                fd=[]
                if dtype==torch.float64:
                    for index,(name,p) in enumerate([('input',x),('weight',m.weight),('bias',m.bias)]):
                        with torch.no_grad():
                            original=p.view(-1)[0].item();p.view(-1)[0]=original+1e-6;plus=(m(x)*probe).sum().item()
                            p.view(-1)[0]=original-1e-6;minus=(m(x)*probe).sum().item();p.view(-1)[0]=original
                        numerical=(plus-minus)/2e-6;analytic=ga[index].view(-1)[0].item()
                        assert abs(numerical-analytic)<=1e-7+1e-5*abs(analytic)
                        fd.append(dict(tensor=name,error=abs(numerical-analytic)))
                rows.append(dict(**case,forward_error=err(actual,expected),gradient_error=max(err(a,b) for a,b in zip(ga,gb)),
                    finite_difference=fd,shared_initialization=True,construction_rng_unchanged=True,
                    exact_suffix_independence=True,exact_future_gradient_zero=True,
                    incumbent_layernorm_exact=kind=='layer_affine',contrast_difference=gap,affine_parameters=2*width))
                del m,twin,x,xr,w,b,actual,expected,ga,gb,probe,changed,future
                if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
            for p,h in {**previous,**sources}.items():assert sha(ROOT/p)==h
            v.artifacts(binding)
    except BaseException as exc:failure=repr(exc)
    result=dict(status='passed' if failure is None else 'failed',error=failure,cases=rows,
        incumbent_observation=observation,plan=ref(B/'component001-plan.json'),guard=ref(B/'component001-guard.json'),
        predecessor_sources_unchanged=len(previous),source_count=len(previous)+len(sources),
        full_model_integrated=False,scientific_training_launched=False,quality_adopted=False,no_retry=True,scope=plan['scope'])
    save('component001-result.json',result)
    print(json.dumps(dict(status=result['status'],cases=len(rows),error=failure,incumbent=observation)))
    if failure:raise RuntimeError(failure)
