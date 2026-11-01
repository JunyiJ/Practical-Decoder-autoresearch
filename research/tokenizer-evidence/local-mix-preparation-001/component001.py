"""Generated component oracle/gradient/cache checks; no corpus or optimizer training."""
from pathlib import Path
import json,hashlib,copy
import torch
from torch.nn import functional as F
from research.local_mix_v1.component import CausalThreeTap
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.block_norm_v1.pilot_execution import exclusive
B=Path(__file__).resolve().parent;ROOT=B.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def oracle(x,m):
    weight=torch.stack((m.lag2,m.lag1,m.current),dim=1).unsqueeze(1)
    return F.conv1d(F.pad(x.transpose(1,2),(2,0)),weight,groups=m.width).transpose(1,2)
def err(a,b):return float((a.detach().cpu().double()-b.detach().cpu().double()).abs().max())
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    from research.warmup_campaign_v1.pilot import idle
    idle()
    sources={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'research/local_mix_v1/__init__.py',ROOT/'research/local_mix_v1/component.py',Path(__file__)]}
    previous=json.loads((ROOT/'research/tokenizer-evidence/warmup-preparation-001/quality001-plan.json').read_text())['sources']
    for n,h in previous.items():assert sha(ROOT/n)==h
    plan=dict(sources=sources,predecessors=previous,decision=ref(B/'direction-decision.json'),cases=[dict(device=d,dtype=t,batch=b) for d,t in [('cpu','float64'),('cpu','float32'),('mps','float32')] for b in [1,3]],width=7,tokens=11,chunks=[4,1,6],training_allowed=False,corpus_used=False,seed_used=False)
    exclusive(B/'component001-plan.json',plan);rows=[];failure=None
    try:
      with StrictMpsGuard(B/'component001-guard.json',ROOT):
        for case in plan['cases']:
            device=case['device'];dtype=getattr(torch,case['dtype']);batch=case['batch'];tol=1e-11 if dtype==torch.float64 else 3e-5
            m=CausalThreeTap(7).to(device=device,dtype=dtype)
            x=(torch.arange(batch*11*7,dtype=torch.float64).reshape(batch,11,7).sin()/3).to(device=device,dtype=dtype)
            assert torch.equal(m(x),x),'Identity initialization differs'
            with torch.no_grad():
                m.current.copy_(torch.arange(7,device=device,dtype=dtype)/17+.8)
                m.lag1.copy_(torch.arange(7,device=device,dtype=dtype)/23-.2)
                m.lag2.copy_(torch.arange(7,device=device,dtype=dtype)/31+.05)
            twin=copy.deepcopy(m);a=x.clone().requires_grad_();b=x.clone().requires_grad_();actual=m(a);expected=oracle(b,twin)
            weight=(torch.arange(actual.numel(),device=device,dtype=dtype).reshape_as(actual)+1)/actual.numel()
            ga=torch.autograd.grad((actual*weight).sum(),(a,*m.parameters()));gb=torch.autograd.grad((expected*weight).sum(),(b,*twin.parameters()))
            torch.testing.assert_close(actual,expected,rtol=tol,atol=tol)
            for u,v in zip(ga,gb):torch.testing.assert_close(u,v,rtol=tol,atol=tol)
            with torch.no_grad():
                parts=[];history=None;start=0
                for n in plan['chunks']:
                    y,history=m.forward_chunk(x[:,start:start+n],history);parts.append(y);start+=n
                cached=torch.cat(parts,1);torch.testing.assert_close(cached,actual,rtol=tol,atol=tol)
                assert history.numel()*history.element_size()==batch*2*7*torch.tensor([],dtype=dtype).element_size()
                changed=x.clone();changed[:,6:]+=9;torch.testing.assert_close(m(changed)[:,:6],actual[:,:6],rtol=0,atol=0)
                reset,_=m.forward_chunk(x[:,:4],None);torch.testing.assert_close(reset,actual[:,:4],rtol=0,atol=0)
                try:m.forward_chunk(x,x[:,:1])
                except ValueError:pass
                else:raise AssertionError('Wrong cache shape accepted')
            rows.append(dict(**case,forward_error=err(actual,expected),gradient_error=max(err(u,v) for u,v in zip(ga,gb)),cache_error=err(cached,actual),identity_exact=True,causality=True,reset=True,cache_bytes=history.numel()*history.element_size()))
            del m,twin,a,b,x,actual,expected,ga,gb,cached,parts,history
            if device=='mps':torch.mps.synchronize();torch.mps.empty_cache()
        for n,h in {**previous,**sources}.items():assert sha(ROOT/n)==h
    except BaseException as exc:
        failure=repr(exc)
    result=dict(status='passed' if failure is None else 'failed',error=failure,plan=ref(B/'component001-plan.json'),guard=ref(B/'component001-guard.json'),cases=rows,predecessor_sources_unchanged=len(previous),scope='Standalone generated causal three-tap component only. No full model, optimizer, checkpoint, corpus or scientific quality evidence.',quality_adopted=False,training_launched=False,no_retry=True)
    exclusive(B/'component001-result.json',result)
    print(json.dumps(dict(status=result['status'],cases=len(rows),error=failure)))
    if failure:raise SystemExit(1)
