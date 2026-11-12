"""New sampler/indexing and detached-copy-loss checks; no model or validation load."""
import fcntl
import json
import math
from pathlib import Path
import shutil
import time
import traceback
from research.general_rsi_pseudo_spans_v1 import ROOT,OUT,DATA,Sampler,repetition_loss,sha


def main():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    dest=OUT/'component-attempt-01';dest.mkdir()
    plan=dict(source_sha256=sha(__file__),component_sha256=sha(ROOT/'research/general_rsi_pseudo_spans_v1.py'),
        inventory_sha256=sha(OUT/'inventory.json'),seed=62111205080001,total_seconds=180,
        rss_limit_bytes=2*2**30,output_cap_bytes=64*2**20,reserve_bytes=10*2**30,
        checks=['full inventory document bounds and eligibility','detached scalar and gradient oracle, both signs',
                'FP32 agreement with double oracle','masked/padding invariance','numerical saturation and invalid input rejection',
                'sampler state roundtrip and global RNG isolation','real batch alignment/masks/counts'],
        free_bytes=shutil.disk_usage(ROOT).free,test_opened=False,validation_opened=False,model_loads=0)
    assert plan['free_bytes']>=plan['reserve_bytes']+plan['output_cap_bytes']
    (dest/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with CpuPrerequisiteGuard(dest/'guard.json',ROOT,timeout_seconds=180,rss_limit_bytes=2*2**30,reserve_bytes=10*2**30):
            try:
                import numpy as np
                import torch
                start=time.monotonic();torch.set_num_threads(2)
                sampler=Sampler(plan['seed']);a=sampler.spans
                docs=[json.loads(l) for l in (DATA/'train.index.jsonl').read_text().splitlines()]
                offsets=np.array([d['offset'] for d in docs]);ends=offsets+np.array([d['tokens']-1 for d in docs])
                di,begin,length,prefix=[a[:,i].astype('int64') for i in range(4)]
                assert np.all(begin-prefix>=offsets[di]) and np.all(begin+length<=ends[di])
                assert np.all((length>=4)&(length<=36)) and np.all((prefix>=1)&(prefix<=16))
                assert np.all((129-prefix)//length>=3)
                # Scalar oracle with two opposite-sign differences and unscored anchors.
                raw=[[2.,0.,0.],[.2,-.3,.1],[-2.,0.,0.],[-2.,0.,0.],[2.,0.,0.],[.7,.8,.9]]
                x=torch.tensor([raw],dtype=torch.float64,requires_grad=True)
                y=torch.zeros((1,6),dtype=torch.long);previous=torch.tensor([[0,0,0,0,3,0]])
                mask=torch.tensor([[False,False,True,False,True,False]])
                loss,info=repetition_loss(x,y,previous,mask);loss.backward()
                probs=[[math.exp(z)/sum(math.exp(v) for v in row) for z in row] for row in raw]
                expected_loss=0.;grad=torch.zeros_like(x)
                for t,prev in [(2,0),(4,3)]:
                    p=probs[t][0];difference=p-.5*probs[prev][0]
                    expected_loss+=-math.log1p(-abs(difference))/2
                    scale=(1 if difference>0 else -1)/(1-abs(difference))/2
                    for j in range(3):grad[0,t,j]=scale*p*((1 if j==0 else 0)-probs[t][j])
                assert abs(loss.item()-expected_loss)<1e-12
                error=float((x.grad-grad).abs().max());assert error<1e-12
                assert torch.equal(x.grad[0,[0,1,3,5]],torch.zeros((4,3),dtype=torch.float64))
                fp=x.detach().float().requires_grad_();lp,_=repetition_loss(fp,y,previous,mask);lp.backward()
                fp_error=float((fp.grad.double()-grad).abs().max());assert fp_error<1e-6 and abs(lp.item()-expected_loss)<1e-6
                altered=x.detach().clone();altered[0,5]=torch.tensor([50.,-50.,0.]);lm,_=repetition_loss(altered,y,previous,mask);assert lm.item()==loss.item()
                extreme=torch.tensor([[[-10000.,10000.,0.],[10000.,-10000.,0.]]],requires_grad=True)
                le,ie=repetition_loss(extreme,torch.zeros((1,2),dtype=torch.long),torch.zeros((1,2),dtype=torch.long),torch.tensor([[False,True]]));le.backward()
                assert ie['saturated']==1 and bool(torch.isfinite(le)) and bool(torch.isfinite(extreme.grad).all())
                rejected=[]
                for name in ['nonfinite','future_anchor','mismatched_token','empty_mask']:
                    xx=x.detach().clone();yy=y.clone();pp=previous.clone();mm=mask.clone()
                    if name=='nonfinite':xx[0,0,0]=float('nan')
                    elif name=='future_anchor':pp[0,2]=2
                    elif name=='mismatched_token':yy[0,2]=1
                    else:mm[:]=False
                    try:repetition_loss(xx,yy,pp,mm)
                    except AssertionError:rejected.append(name)
                    else:raise AssertionError(f'accepted invalid {name}')
                global_before=torch.random.get_rng_state().clone();sampler.batch();state=sampler.state_dict()
                torch.save(state,dest/'sampler-state.pt')
                expected=sampler.batch();restored=Sampler(plan['seed']);restored.load_state_dict(torch.load(dest/'sampler-state.pt',weights_only=True))
                got=restored.batch();assert all(torch.equal(a,b) for a,b in zip(expected[:4],got[:4])) and expected[4]==got[4]
                assert torch.equal(global_before,torch.random.get_rng_state())
                corrupt=restored.state_dict();corrupt['binding']='0'*64
                try:restored.load_state_dict(corrupt)
                except AssertionError:rejected.append('wrong_inventory_state')
                else:raise AssertionError('accepted wrong inventory')
                counts=[];sources={};lengths=set()
                for _ in range(32):
                    bx,by,bp,bm,detail=restored.batch()
                    assert bx.shape==by.shape==bp.shape==bm.shape==(8,128)
                    assert torch.equal(bx[:,1:],by[:,:-1])
                    for row,d in enumerate(detail):
                        L=d['length'];P=d['prefix'];K=d['copies'];full=torch.cat([bx[row,:1],by[row]])
                        span=full[P:P+L]
                        for k in range(K):assert torch.equal(full[P+k*L:P+(k+1)*L],span)
                        assert bool((full[P+K*L:]==50256).all())
                        assert int(bm[row].sum())==(K-1)*L==d['penalty_targets']
                        assert not bool(bm[row,:P+L-1].any()) and not bool(bm[row,P+K*L-1:].any())
                        assert torch.equal(by[row].gather(0,bp[row])[bm[row]],by[row][bm[row]])
                        counts.append(d['penalty_targets']);lengths.add(L);s=docs[d['document']]['source'];sources[s]=sources.get(s,0)+1
                result=dict(passed=True,plan_sha256=sha(dest/'plan.json'),inventory_spans=len(a),oracle_loss=expected_loss,
                    max_double_gradient_error=error,max_fp32_gradient_error=fp_error,saturation_loss=float(le.detach()),invalid_cases_rejected=rejected,
                    sampler_roundtrip_exact=True,global_rng_unchanged=True,batches_audited=32,span_lengths_audited=sorted(lengths),
                    valid_targets_per_row=dict(min=min(counts),max=max(counts),mean=sum(counts)/len(counts)),sampled_sources=sources,
                    new_model_scores=0,test_opened=False,validation_opened=False,seconds=time.monotonic()-start)
                (dest/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
            except BaseException as e:
                (dest/'failure.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc()),indent=2)+'\n');raise


if __name__=='__main__':main()
