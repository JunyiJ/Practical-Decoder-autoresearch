"""Generated full-model/optimizer integration. No corpus, EMA/resume or training run."""
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback
import torch
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.body_allocation_model_v1.contract import config as native_training, model_config as native_config
from research.body_allocation_model_v1.model import GPT as Native
from research.body_allocation_model_v1.optimizer import build_optimizer as native_optimizer
from research.tokenizer_moe_model_v1.contract import model_config,ARMS
from research.tokenizer_moe_model_v1.model import GPT,load_weights
from research.tokenizer_moe_model_v1.optimizer import build_optimizer

BASE=Path(__file__).resolve().parents[3]
E=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return isinstance(b,dict) and a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b


def digest(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():
        h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def reject(call):
    try:call()
    except (ValueError,RuntimeError):return
    raise AssertionError('Expected foreign/invalid policy rejection')


def main():
    for n in ['model001-result.json','model001-failure.json','model001-guard.json']:
        assert not (E/n).exists(),'Single-attempt new proof'
    plan=json.loads((E/'model001-plan.json').read_text())
    for n,h in plan['sources'].items():assert sha(BASE/n)==h,n
    assert len(plan['sources'])==1014
    assert all(os.environ.get(k)==v for k,v in plan['flags'].items())
    torch.set_num_threads(4)
    assert __import__('shutil').disk_usage(BASE).free>=plan['required_free_bytes']
    cpu_rng=torch.get_rng_state().clone();rows=[];native_checks={}
    with (BASE/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with StrictMpsGuard(E/'model001-guard.json',BASE),torch.random.fork_rng(devices=[]):
            mps_rng=torch.mps.get_rng_state().clone()
            # Exact native construction, RNG, logits, gradients and two optimizer steps.
            torch.random.default_generator.manual_seed(600452);reference=Native(native_config('incumbent'));after_reference=torch.get_rng_state().clone()
            torch.random.default_generator.manual_seed(600452);control=GPT(model_config('dense'))
            assert torch.equal(after_reference,torch.get_rng_state()) and same(reference.state_dict(),control.state_dict())
            training=native_training('incumbent',4,600452).training
            first=native_optimizer(reference,native_training('incumbent',4,600452));second=build_optimizer(control,training)
            assert same([o.state_dict()['param_groups'] for o in first.optimizers],[o.state_dict()['param_groups'] for o in second.optimizers])
            x=torch.randint(50257,(1,8));y=torch.randint(50257,(1,8))
            for step in range(2):
                first.zero_grad();second.zero_grad()
                logits,a=reference(x,y);other,b=control(x,y,include_aux_loss=False)
                assert torch.equal(logits,other) and torch.equal(a,b)
                a.backward();b.backward()
                assert all(same(p.grad,q.grad) for p,q in zip(reference.parameters(),control.parameters()))
                torch.nn.utils.clip_grad_norm_(reference.parameters(),1.)
                torch.nn.utils.clip_grad_norm_(control.parameters(),1.)
                first.step();second.step()
                assert same(reference.state_dict(),control.state_dict())
                assert same([o.state_dict() for o in first.optimizers],[o.state_dict() for o in second.optimizers])
            assert second.completed_steps==2 and set(second.gradient_steps.values())=={2}
            native_checks=dict(initial_state=True,post_cpu_rng=True,logits_ce=True,gradients=True,two_optimizer_steps=True,children_state_exact=True)
            del reference,control,first,second,logits,other,a,b
            gc.collect()

            for arm in ARMS:
                torch.random.default_generator.manual_seed(600453)
                cpu=GPT(model_config(arm)).eval()
                # Same original non-FFN initialization across arms by construction.
                torch.random.default_generator.manual_seed(600453);baseline=Native(native_config('incumbent'))
                assert all(torch.equal(v,baseline.state_dict()[n]) for n,v in cpu.state_dict().items() if '.mlp.' not in n)
                del baseline
                torch.random.default_generator.manual_seed(600454);mps=GPT(model_config(arm));mps.load_state_dict(cpu.state_dict());mps=mps.to('mps').eval()
                optimizer=build_optimizer(cpu,training)
                router_groups=[g for g in optimizer.optimizers[1].param_groups if g['parameter_names'][0].endswith('.mlp.router.weight')]
                assert len(router_groups)==(4 if arm=='moe' else 0)
                assert all(g['width_lr_scale']==1. and g['lr']==training.learning_rate for g in router_groups)
                assert len(optimizer.optimizers[0].param_groups)==cpu.signature['muon_groups']
                assert len(optimizer.optimizers[1].param_groups)==cpu.signature['adam_groups']
                ids=torch.randint(50257,(1,8));targets=torch.randint(50257,(1,8));device_rows=[]
                c_logits,c_loss,c_break=cpu(ids,targets,include_aux_loss=True,return_loss_breakdown=True)
                m_logits,m_loss,m_break=mps(ids.to('mps'),targets.to('mps'),include_aux_loss=True,return_loss_breakdown=True)
                assert torch.allclose(c_logits,m_logits.cpu(),atol=5e-5,rtol=5e-4)
                assert abs(float(c_loss.detach()-m_loss.detach().cpu()))<5e-5
                if arm=='moe':
                    assert torch.equal(c_loss,c_break['ce_loss']+.01*c_break['aux_loss'])
                    _,ce=cpu(ids,targets,include_aux_loss=False)
                    assert torch.equal(ce,c_break['ce_loss'])
                c_loss.backward();m_loss.backward();max_grad=0.
                for (name,p),(other,q) in zip(cpu.named_parameters(),mps.named_parameters()):
                    assert name==other
                    if p.grad is None:assert q.grad is None
                    else:
                        assert q.grad is not None and torch.allclose(p.grad,q.grad.cpu(),atol=5e-5,rtol=5e-4),name
                        max_grad=max(max_grad,float((p.grad-q.grad.cpu()).abs().max()))
                cpu.zero_grad(set_to_none=True);mps.zero_grad(set_to_none=True)
                del c_logits,c_loss,c_break,m_logits,m_loss,m_break
                for device,model in [('cpu',cpu),('mps',mps)]:
                    t=ids.to(device);changed=t.clone();changed[:,3:]=(changed[:,3:]+1)%50257
                    with torch.no_grad():
                        a,_=model(t);b,_=model(changed)
                        assert torch.allclose(a[:,:3],b[:,:3],atol=3e-5,rtol=3e-5)
                    captured=[]
                    def capture(_module,_args,value):value.retain_grad();captured.append(value)
                    hook=model.transformer.wte.register_forward_hook(capture)
                    a,_=model(t);a[:,:3].sum().backward();hook.remove()
                    assert torch.count_nonzero(captured[0].grad[:,3:])==0
                    model.zero_grad(set_to_none=True);del captured,a,b
                    checks=[]
                    for batch in [1,8]:
                        t=torch.randint(50257,(batch,128)).to(device)
                        with torch.no_grad():
                            full,_=model(t);caches=model.caches(batch)
                            parts=[];start=0
                            for length in [17,53,58]:
                                part,_=model(t[:,start:start+length],cache=caches,start_pos=start);parts.append(part);start+=length
                            cached=torch.cat(parts,dim=1)
                            err=float((full-cached).abs().max())
                            assert torch.allclose(full,cached,atol=4e-5,rtol=4e-5),(arm,device,batch,err)
                            allocated=sum(v.numel()*v.element_size() for c in caches for v in [c.c_cache,c.r_cache])
                            assert allocated==196608*batch
                            for c in caches:c.reset()
                            reset,_=model(t,cache=caches,start_pos=0)
                            assert torch.allclose(full,reset,atol=4e-5,rtol=4e-5)
                        checks.append(dict(batch=batch,cache_bytes_per_item=allocated//batch,maximum_logit_error=err))
                        del full,cached,reset,parts,part,caches,t
                    with torch.no_grad():
                        caches=model.caches(1);caches[0].signature=dict(cpu.signature,policy='foreign')
                        reject(lambda:model(ids.to(device),cache=caches))
                        caches=model.caches(1);caches[1]=caches[0]
                        reject(lambda:model(ids.to(device),cache=caches))
                        reject(lambda:model(torch.full((1,1),50257,dtype=torch.int64,device=device)))
                    device_rows.append(dict(device=device,causality=True,future_embedding_gradient_zero=True,cache=checks,foreign_cache_id_rejection=True))
                # Weights-only transport; this explicitly does not claim raw optimizer/EMA resume.
                saved=cpu.state_dict();payload=dict(kind='tokenizer_moe_weights_v1',architecture=cpu.signature,model_state=saved)
                before=digest(cpu);load_weights(cpu,payload);assert digest(cpu)==before
                bad=[]
                q=dict(payload,kind='old');bad.append(q)
                q=dict(payload,architecture=dict(cpu.signature,arm='foreign'));bad.append(q)
                key='transformer.h.0.attn.dkv.bias'
                for mode in ['missing','shape','dtype','nonfinite']:
                    states=dict(saved)
                    if mode=='missing':states.pop(key)
                    elif mode=='shape':states[key]=states[key][:-1]
                    elif mode=='dtype':states[key]=states[key].double()
                    else:states[key]=states[key].clone();states[key][0]=float('nan')
                    bad.append(dict(payload,model_state=states))
                for q in bad:reject(lambda q=q:load_weights(cpu,q));assert digest(cpu)==before
                if arm=='moe':
                    cpu.train()
                    with torch.no_grad():
                        for block in cpu.transformer.h:block.mlp.router.weight.zero_()
                    optimizer.zero_grad();_,loss=cpu(ids,targets,include_training_objective=True);loss.backward()
                    dormant={n:p for n,p in cpu.named_parameters() if any(f'.experts.{i}.' in n for i in [2,3])}
                    assert len(dormant)==24 and all(p.grad is None for p in dormant.values())
                    old={n:p.detach().clone() for n,p in dormant.items()}
                    optimizer.step()
                    assert all(torch.equal(old[n],p) and optimizer.gradient_steps[n]==0 for n,p in dormant.items())
                    assert all(int(b.mlp.routed_tokens)==8 and b.mlp.assignment_counts.tolist()==[8,8,0,0] for b in cpu.transformer.h)
                    active_name='transformer.h.0.mlp.experts.0.gate.weight';active=dict(cpu.named_parameters())[active_name]
                    optimizer.zero_grad();old_active=active.detach().clone();active.grad=torch.zeros_like(active)
                    # Fresh optimizer isolates exact independent decay from old momentum.
                    zero_opt=build_optimizer(cpu,training);zero_opt.step()
                    assert torch.equal(active,old_active*(1-.00006)) and zero_opt.gradient_steps[active_name]==1
                    assert all(zero_opt.gradient_steps[n]==0 and torch.equal(old[n],p) for n,p in dormant.items())
                    del old,zero_opt,loss
                rows.append(dict(arm=arm,parameters=cpu.signature['parameters'],muon_groups=cpu.signature['muon_groups'],adam_groups=cpu.signature['adam_groups'],shared_native_non_ffn_init_exact=True,maximum_cpu_mps_gradient_error=max_grad,devices=device_rows,weights_only_forgeries_rejected=len(bad),dormant_expert_skip_decay_checked=arm=='moe'))
                del cpu,mps,optimizer,payload,saved,bad
                gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
            assert torch.equal(mps_rng,torch.mps.get_rng_state()),'Deterministic models consumed MPS RNG'
    assert torch.equal(cpu_rng,torch.get_rng_state())
    for n,h in plan['sources'].items():assert sha(BASE/n)==h,n
    result=dict(status='three_arm_model_optimizer_and_cache_integration_passed',plan_sha256=sha(E/'model001-plan.json'),native_exact=native_checks,rows=rows,source_count=1014,predecessor_sources_unchanged=1009,fixture_seed=600452,extra_generated_fixture_seeds=[600453,600454],scientific_seed=False,full_resume_ema=False,corpus_loaded=False,weights_written=0,quality_panel=False,global_cpu_mps_rng_unchanged=True)
    with (E/'model001-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],arms=[r['arm'] for r in rows],maximum_gradient_error=max(r['maximum_cpu_mps_gradient_error'] for r in rows))))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if not (E/'model001-failure.json').exists():
            with (E/'model001-failure.json').open('x') as f:json.dump(dict(status='CLOSED_FAILED_GENERATED_MODEL_PROOF',error=repr(exc),traceback=traceback.format_exc(),source_sha256=sha(__file__),never_rerun=True,no_quality_result=True),f,indent=2);f.write('\n')
        raise
