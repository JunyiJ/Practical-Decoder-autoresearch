"""Generated fixed-gain serial equations/all-gradient/causality checks, no corpus/training."""
import copy,hashlib,json,math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from research.branch_scale_v1.component import ResidualScaleComposition,KINDS,GAINS
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.block_norm_v1.pilot_execution import exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent;ROOT=B.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def generated(shape,offset):
    return (torch.arange(math.prod(shape),dtype=torch.float64).reshape(shape)*.137+offset).sin()/math.sqrt(shape[-1])
class Norm(nn.Module):
    def __init__(self,width):super().__init__();self.gain=nn.Parameter(torch.ones(width,dtype=torch.float64))
    def forward(self,x):return x*torch.rsqrt(x.square().mean(dim=-1,keepdim=True)+1e-5)*self.gain
class Attention(nn.Module):
    def __init__(self,width):
        super().__init__();self.width=width
        for i,name in enumerate(('query','key','value','output')):self.register_parameter(name,nn.Parameter(generated((width,width),.3+i)))
    def forward(self,x):
        q,k,v=(F.linear(x,getattr(self,name)) for name in ('query','key','value'))
        scores=(q@k.transpose(-2,-1))*(.5/math.sqrt(self.width))
        mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
        return F.linear(scores.masked_fill(mask,float('-inf')).softmax(dim=-1)@v,self.output)
class FFN(nn.Module):
    def __init__(self,width,hidden):
        super().__init__();self.gate=nn.Parameter(generated((hidden,width),1.1));self.up=nn.Parameter(generated((hidden,width),2.2));self.down=nn.Parameter(generated((width,hidden),3.3))
    def forward(self,x):return F.linear(F.silu(F.linear(x,self.gate))*F.linear(x,self.up),self.down)
def make(kind,device,dtype):return ResidualScaleComposition(Attention(7),FFN(7,19),Norm(7),Norm(7),kind).to(device=device,dtype=dtype)
def reference(x,params,kind):
    # Independent tokenwise causal sums, explicit SiLU and square-root division.
    def norm(z,name):return z/(z.pow(2).sum(-1,keepdim=True)/7+1e-5).sqrt()*params[name+'.gain']
    z=norm(x,'norm1');q=z@params['attention.query'].T;k=z@params['attention.key'].T;v=z@params['attention.value'].T
    outputs=[]
    for t in range(x.shape[1]):
        scores=(q[:,t:t+1]*k[:,:t+1]).sum(-1)*(.5/math.sqrt(7))
        probs=torch.exp(scores-scores.max(-1,keepdim=True).values);probs=probs/probs.sum(-1,keepdim=True)
        outputs.append((probs.unsqueeze(-1)*v[:,:t+1]).sum(1))
    attention=torch.stack(outputs,dim=1)@params['attention.output'].T;u=x+GAINS[kind]*attention
    z=norm(u,'norm2');gate=z@params['ffn.gate'].T;up=z@params['ffn.up'].T
    return u+GAINS[kind]*(((gate*torch.sigmoid(gate))*up)@params['ffn.down'].T)

def error(a,b):return float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max())

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    from research.cooldown_quality_v1.support import idle
    idle();assert not (B/'component001-plan.json').exists()
    previous=json.loads((ROOT/'research/tokenizer-evidence/cooldown-preparation-001/quality001-plan.json').read_text())['sources']
    for n,h in previous.items():assert sha(ROOT/n)==h
    files=[ROOT/'research/branch_scale_v1/__init__.py',ROOT/'research/branch_scale_v1/component.py',Path(__file__)]
    sources={str(p.relative_to(ROOT)):sha(p) for p in files}
    plan=dict(sources=sources,predecessors=previous,decision=ref(B/'direction-decision.json'),cases=[dict(device=d,dtype=t,batch=b,kind=k) for d,t in [('cpu','float64'),('cpu','float32'),('mps','float32')] for b in [1,3] for k in KINDS],width=7,hidden=19,tokens=9,causal_prefix=5,tolerances=dict(float64=1e-10,float32=8e-5),finite_difference=dict(step=1e-6,atol=1e-7,rtol=1e-5),training_allowed=False,scientific_seed_used=False,corpus_used=False,scope='Generic one-head causal-attention/SwiGLU fixed-gain serial composition only; no RoPE MLA/full-model/cache/optimizer integration claim.')
    exclusive(B/'component001-plan.json',plan);rows=[];failure=None
    try:
      with StrictMpsGuard(B/'component001-guard.json',ROOT):
        for case in plan['cases']:
            device=case['device'];dtype=getattr(torch,case['dtype']);batch=case['batch'];kind=case['kind'];tol=plan['tolerances'][case['dtype']]
            rng=torch.get_rng_state().clone();m=make(kind,device,dtype);twin=make('unit' if kind!='unit' else 'fixed_half',device,dtype)
            assert torch.equal(rng,torch.get_rng_state()),'Construction consumed RNG'
            assert list(m.state_dict())==list(twin.state_dict()) and all(torch.equal(v,twin.state_dict()[n]) for n,v in m.state_dict().items()),'Shared initialization differs'
            assert m.norm1 is not m.norm2 and m.norm1.gain is not m.norm2.gain
            x=generated((batch,9,7),.21).to(device=device,dtype=dtype).requires_grad_();xr=x.detach().clone().requires_grad_()
            actual=m(x);params={n:p.detach().clone().requires_grad_() for n,p in m.named_parameters()};expected=reference(xr,params,kind)
            weight=generated(tuple(actual.shape),.83).to(device=device,dtype=dtype)
            ga=torch.autograd.grad((actual*weight).sum(),(x,*m.parameters()),retain_graph=True);gb=torch.autograd.grad((expected*weight).sum(),(xr,*params.values()))
            torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
            for a,b in zip(ga,gb):assert torch.isfinite(a).all() and torch.isfinite(b).all();torch.testing.assert_close(a,b,rtol=tol,atol=tol)
            future=torch.autograd.grad(actual[:,:5].square().sum(),x)[0][:,5:]
            assert torch.count_nonzero(future)==0,'Future input gradient entered prefix'
            changed=x.detach().clone();changed[:,5:]+=3.7
            torch.testing.assert_close(m(changed)[:,:5],actual[:,:5],rtol=0,atol=0)
            gap=error(actual,twin(x.detach()));assert gap>1e-6,'Gain contrast accidentally collapsed'
            if kind=='unit':
                legacy=x+m.attention(m.norm1(x));legacy=legacy+m.ffn(m.norm2(legacy))
                assert torch.equal(actual,legacy),'Unit gain changed serial reference'
                del legacy
            fd=[]
            if dtype==torch.float64:
                for name in ('attention.query','attention.output','ffn.gate','ffn.down','norm1.gain','norm2.gain'):
                    p=dict(m.named_parameters())[name];idx=list(dict(m.named_parameters())).index(name)+1
                    with torch.no_grad():
                        original=p.view(-1)[0].item();p.view(-1)[0]=original+1e-6;plus=(m(x.detach())*weight).sum().item();p.view(-1)[0]=original-1e-6;minus=(m(x.detach())*weight).sum().item();p.view(-1)[0]=original
                    numerical=(plus-minus)/2e-6;analytic=ga[idx].view(-1)[0].item();assert abs(numerical-analytic)<=1e-7+1e-5*abs(analytic)
                    fd.append(dict(parameter=name,error=abs(numerical-analytic)))
            rows.append(dict(**case,forward_error=error(actual,expected),gradient_error=max(error(a,b) for a,b in zip(ga,gb)),gradient_tensors=len(ga),causal_suffix_perturbation=True,causal_future_gradient_exact_zero=True,shared_weights_exact=True,construction_rng_unchanged=True,gain_output_difference=gap,unit_reference_exact=(kind=='unit'),finite_difference=fd))
            del m,twin,x,xr,actual,expected,params,ga,gb,changed,future
            if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
        # Shared normalizer and unknown gain kind are rejected without model work.
        norm=Norm(7)
        for other,kind in [(norm,'unit'),(Norm(7),'unknown')]:
            try:ResidualScaleComposition(Attention(7),FFN(7,19),norm,other,kind)
            except ValueError:pass
            else:raise AssertionError('Invalid composition accepted')
        for n,h in {**previous,**sources}.items():assert sha(ROOT/n)==h
    except BaseException as exc:failure=repr(exc)
    result=dict(status='passed' if failure is None else 'failed',error=failure,plan=ref(B/'component001-plan.json'),guard=ref(B/'component001-guard.json'),cases=rows,predecessor_sources_unchanged=len(previous),full_model_integrated=False,scientific_training_launched=False,quality_adopted=False,no_retry=True,scope=plan['scope'])
    exclusive(B/'component001-result.json',result);print(json.dumps(dict(status=result['status'],cases=len(rows),error=failure)))
    if failure:raise SystemExit(1)
