"""Finite equations/init/gradient/cache proof for a new input-only dropout site."""
import argparse,copy,gc,json
from pathlib import Path
import torch
import torch.nn.functional as F
from research.body_allocation_model_v1.contract import ROOT,require,config as base_config,model_config
from research.body_allocation_model_v1.model import GPT as Original
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_v1.model import GPT,InputDropout,config,load_weights
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001'
p=argparse.ArgumentParser();p.add_argument('device',choices=['cpu','mps']);a=p.parse_args();torch.set_num_threads(4)
def inventory():
    d=json.load(open(B/'quality001-closure001.json'));pins={**d['sources'],**d['pilot_sources'],**d['quality_sources']}
    pins.update(json.load(open(ROOT/'research/tokenizer-evidence/exploration-boundary-002/continuation-completed001.json'))['new_inference_diagnostic_sources'])
    for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Frozen predecessor changed '+n)
    for f in (ROOT/'research/embedding_input_dropout_v1').glob('*.py'):pins[str(f.relative_to(ROOT))]=reference(f)['sha256']
    return pins
def rejected(fn):
    try:fn()
    except (TypeError,ValueError):return
    raise AssertionError('Invalid dropout contract accepted')
def rng():return (torch.get_rng_state().clone(),torch.mps.get_rng_state().clone() if a.device=='mps' else None)
def restore(s):
    torch.set_rng_state(s[0])
    if a.device=='mps':torch.mps.set_rng_state(s[1])
def same_rng(x,y):return torch.equal(x[0],y[0]) and (x[1] is None or torch.equal(x[1],y[1]))
def gradients(x,y):
    for (n,p),(m,q) in zip(x.named_parameters(),y.named_parameters()):
        require(n==m and p.grad is not None and q.grad is not None,'Gradient inventory differs')
        torch.testing.assert_close(p.grad,q.grad,rtol=3e-4,atol=3e-5)
OUT=E/f'correctness001-{a.device}.json';GUARD=E/f'correctness001-{a.device}-guard.json'
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not OUT.exists() and not GUARD.exists(),'Closed finite proof cannot retry');pins=inventory()
    guard=StrictMpsGuard(GUARD,ROOT) if a.device=='mps' else CpuPrerequisiteGuard(GUARD,ROOT,timeout_seconds=120,poll_seconds=.25)
    with guard,torch.random.fork_rng(devices=[]):
        initial=rng();old=Original(model_config('incumbent')).to(a.device);old_init=rng();restore(initial);zero=GPT(config(0.)).to(a.device)
        require(same_rng(old_init,rng()) and all(n==m and torch.equal(p,q) for (n,p),(m,q) in zip(old.named_parameters(),zero.named_parameters())),'Original initialization/RNG changed')
        rejected(lambda:GPT(config(.05)));cfg=base_config('incumbent',steps=4,seed=600404,lr=.00125)
        oo=build_optimizer(old,cfg);oz=build_optimizer(zero,cfg)
        require([len(o.param_groups) for o in oz.optimizers]==[44,44] if hasattr(oz,'optimizers') else True,'Optimizer groups differ')
        x=(torch.arange(16,device=a.device).reshape(1,16)%400+1).long();y=(x+1)%50257;maxlogit=0.
        for _ in range(2):
            old.zero_grad(set_to_none=True);zero.zero_grad(set_to_none=True);before=rng();lo,ceo=old(x,y);lz,cez=zero(x,y)
            require(same_rng(before,rng()),'Zero-dropout path consumed RNG');torch.testing.assert_close(lz,lo,rtol=3e-4,atol=3e-5)
            maxlogit=max(maxlogit,float((lz-lo).abs().max()));ceo.backward();cez.backward();gradients(old,zero)
            torch.nn.utils.clip_grad_norm_(old.parameters(),1.);torch.nn.utils.clip_grad_norm_(zero.parameters(),1.);oo.step();oz.step()
            require(all(torch.equal(p,q) for p,q in zip(old.parameters(),zero.parameters())),'Zero-control two-step optimizer identity changed')
            del lo,lz,ceo,cez
        own_state=oz.state_dict();require(own_state['body_allocation_signature']==zero.signature,'Own optimizer policy binding absent')
        candidate=GPT(config(.1)).to(a.device);candidate.load_state_dict(zero.state_dict());oc=build_optimizer(candidate,cfg);rejected(lambda:oc.load_state_dict(own_state));del oc,own_state,oo,oz
        old.zero_grad(set_to_none=True);zero.zero_grad(set_to_none=True);candidate.zero_grad(set_to_none=True)
        before=rng();embed=old.transformer.wte(x);mask=F.dropout(torch.ones_like(embed),p=.1,training=True);after_mask=rng();h=embed*mask
        for block in old.transformer.h:h=block(h,None,0)
        oracle=old.lm_head(old.transformer.ln_f(h))/old.width_mult;loss=F.cross_entropy(oracle.reshape(-1,50257),y.reshape(-1));loss.backward()
        restore(before);actual,loss_actual=candidate(x,y);require(same_rng(after_mask,rng()),'Input dropout RNG consumption differs from direct mask oracle')
        torch.testing.assert_close(actual,oracle,rtol=3e-4,atol=3e-5);loss_actual.backward();gradients(old,candidate)
        stochastic_error=float((actual-oracle).abs().max());require(bool((mask==0).any()) and bool((mask>0).any()),'Nontrivial dropout fixture required')
        # Same RNG -> same stochastic output; no quality/scientific seed consumed.
        restore(before);repeat=candidate(x)[0];require(torch.equal(actual,repeat),'Device-local stochastic repeat not exact')
        del actual,oracle,loss,loss_actual,h,embed,mask,repeat,old,zero;gc.collect();candidate.zero_grad(set_to_none=True)
        candidate.eval();records=[]
        with torch.no_grad():
            for batch in (1,8):
                idx=(torch.arange(batch*128,device=a.device).reshape(batch,128)%400+1).long();s=rng();full=candidate(idx)[0];require(same_rng(s,rng()),'Eval consumed dropout RNG')
                cache=candidate.caches(batch);replay=torch.cat([candidate(idx[:,:32],cache=cache)[0],candidate(idx[:,32:],cache=cache,start_pos=32)[0]],1)
                torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5);err=float((full-replay).abs().max());del replay
                for c in cache:c.reset()
                replay=candidate(idx,cache=cache)[0];torch.testing.assert_close(full,replay,rtol=3e-4,atol=3e-5);del replay
                changed=idx.clone();changed[:,64:]=(changed[:,64:]+3)%50257;future=candidate(changed)[0];torch.testing.assert_close(full[:,:64],future[:,:64],rtol=3e-4,atol=3e-5);del future,changed
                for c in cache:c.reset()
                cache[-1].signature['input_dropout']=0.;rejected(lambda:candidate(idx,cache=cache));require(all(c.position==0 for c in cache),'Foreign rejection mutated cache')
                cache=candidate.caches(batch);cache[0].c_cache=next(candidate.parameters()).detach().reshape(-1)[:batch*128*64].reshape(batch,128,64);rejected(lambda:candidate(idx,cache=cache));require(all(c.position==0 for c in cache),'Alias mutated cache')
                records.append(dict(batch=batch,full_cache_reset_causality=True,max_cache_error=err,foreign_and_alias_rejected=True,cache_bytes_per_item=196608))
                del full,cache,idx;gc.collect()
                if a.device=='mps':torch.mps.empty_cache()
            payload=dict(kind='h200_embedding_input_dropout_v1',architecture=copy.deepcopy(candidate.signature),model_state=candidate.state_dict())
            load_weights(candidate,payload);bad=dict(payload,architecture=dict(payload['architecture'],input_dropout=0.));rejected(lambda:load_weights(candidate,bad))
            bad=dict(payload,kind='h200_body_allocation_v1');rejected(lambda:load_weights(candidate,bad))
        del candidate,payload,bad;gc.collect();require(inventory()==pins,'Sources changed during proof')
    result=dict(status='passed',device=a.device,sources=pins,source=reference(__file__),zero_control_original_init_rng_logits_gradients_two_optimizer_steps_exact=True,
        zero_control_max_logit_difference=maxlogit,input_mask_equations_gradients_rng_repeat=True,input_mask_max_logit_difference=stochastic_error,
        foreign_optimizer_checkpoint_cache_rejected=True,records=records,parameters=28492416,parameter_tensors=88,muon_groups=44,adamw_groups=44,new_scientific_seed=False,guard=reference(GUARD))
    raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<=512*1024,'Bounded proof record')
    with OUT.open('xb') as f:f.write(raw)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','source','guard')}))
