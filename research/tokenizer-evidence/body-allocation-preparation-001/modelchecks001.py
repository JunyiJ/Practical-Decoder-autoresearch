"""Generated CPU/MPS math, grouping, causality and cache checks; no training data."""
import copy,gc,hashlib,json,math,sys
from pathlib import Path
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import config,BASE,ARMS
from research.body_allocation_model_v1.model import GPT,load_weights,_Grouped
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.final_ln_scale_model_v1.runtime.models.gpt import GPT as Incumbent
from research.final_ln_scale_model_v1.integration import build_optimizer as incumbent_optimizer
from research.gradient_clip_model_v1.clipping import clip_gradients
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.block_norm_v1.mps_guard import StrictMpsGuard

B=Path(__file__).resolve().parent;ROOT=B.parents[2]
mode=sys.argv[1];assert mode in ('cpu','mps');device=mode
stem='modelchecks001-'+mode;out=B/(stem+'.json');guard=B/(stem+'-guard.json')
assert not out.exists() and not guard.exists()
torch.set_num_threads(4)

def reject(fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError):return
    raise AssertionError('Invalid configuration/artifact accepted')

def close(a,b):
    torch.testing.assert_close(a.detach().cpu(),b.detach().cpu(),rtol=3e-4,atol=3e-5)
    return float((a.detach().cpu()-b.detach().cpu()).abs().max())

def oracle(model,tokens):
    x=F.embedding(tokens,model.transformer.wte.weight);s=tokens.shape[1]
    def linear(x,m):return x@m.weight.T+(0 if m.bias is None else m.bias)
    def norm(x,m):return x*torch.rsqrt(x.square().mean(-1,keepdim=True)+2**-23)*m.weight
    def rotary(x):
        pair=x.reshape(*x.shape[:-1],16,2);pos=torch.arange(s,device=x.device,dtype=torch.float32)
        f=1/(10000.**(torch.arange(0,32,2,device=x.device,dtype=torch.float32)/32))
        ang=pos[:,None]*f[None,:];c=ang.cos()[None,None];z=ang.sin()[None,None]
        return torch.stack((pair[...,0]*c-pair[...,1]*z,pair[...,0]*z+pair[...,1]*c),-1).flatten(-2)
    for block in model.transformer.h:
        a=block.attn;z=norm(x,block.ln1);cq=linear(z,a.dq);ck=linear(z,a.dkv)
        q=linear(cq,a.uq).reshape(x.shape[0],s,a.n_heads,32).transpose(1,2)
        k=linear(ck,a.uk).reshape(x.shape[0],s,a.n_heads,32).transpose(1,2)
        v=linear(ck,a.uv).reshape(x.shape[0],s,a.n_heads,32).transpose(1,2)
        qr=rotary(linear(cq,a.qr).reshape(x.shape[0],s,a.n_heads,32).transpose(1,2))
        kr=rotary(linear(z,a.kr).unsqueeze(1)).expand(-1,a.n_heads,-1,-1)
        score=(torch.cat((q,qr),-1)@torch.cat((k,kr),-1).transpose(-1,-2))*(.5/math.sqrt(64))
        score=score.masked_fill(torch.arange(s,device=x.device)[None,:]>torch.arange(s,device=x.device)[:,None],float('-inf'))
        h=(score.softmax(-1)@v).transpose(1,2).contiguous().reshape(x.shape)
        x=x+linear(h,a.out);z=norm(x,block.ln2);m=block.mlp
        x=x+linear(F.silu(linear(z,m.linear_gate))*linear(z,m.linear_up),m.linear_down)
    z=x-x.mean(-1,keepdim=True);f=model.transformer.ln_f
    z=z*torch.rsqrt(z.square().mean(-1,keepdim=True)+1e-5)*f.weight+f.bias
    return (z@model.lm_head.weight.T)/model.width_mult

def group_inventory(opt,model):
    names={id(p):n for n,p in model.named_parameters()}
    return [[dict(names=[names[id(p)] for p in g['params']],**{k:v for k,v in g.items() if k!='params'})
             for g in child.param_groups] for child in opt.optimizers]

def run():
    rows=[]
    if mode=='cpu':
        torch.manual_seed(91041);oldcfg=OmegaConf.create(json.loads(BASE.read_text()));old=Incumbent(oldcfg.model);oldopt=incumbent_optimizer(old,oldcfg)
        rng=torch.get_rng_state().clone();torch.manual_seed(91041);cfg=config('incumbent');new=GPT(cfg.model);opt=build_optimizer(new,cfg)
        assert torch.equal(rng,torch.get_rng_state())
        assert group_inventory(opt,new)==group_inventory(oldopt,old)
        p=dict(old.named_parameters());q=dict(new.named_parameters());assert p.keys()==q.keys() and all(torch.equal(p[n],q[n]) for n in p)
        tokens=torch.arange(18).reshape(2,9)+1;targets=tokens+1
        for step in range(2):
            oldopt.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
            a,al=old(tokens,targets);b,bl=new(tokens,targets);assert torch.equal(a,b) and torch.equal(al,bl)
            al.backward();bl.backward();assert all(torch.equal(p[n].grad,q[n].grad) for n in p)
            assert clip_gradients(old.parameters(),1.)==clip_gradients(new.parameters(),1.)
            oldopt.step();opt.step();assert all(torch.equal(p[n],q[n]) for n in p)
        rows.append(dict(case='incumbent_exact_init_rng_logits_all_gradients_groups_two_updates',passed=True))
        del old,new,oldopt,opt,p,q,a,b,al,bl;gc.collect()
    for index,arm in enumerate(ARMS):
        cfg=config(arm);torch.manual_seed(91051+index);cpu=GPT(cfg.model)
        tokens=(torch.arange(18).reshape(2,9)+41).long();targets=tokens+1
        if mode=='mps':
            with torch.no_grad():expected,ce=cpu(tokens,targets)
            ce_cpu=ce.item();cpu.zero_grad(set_to_none=True);cpu(tokens,targets)[1].backward()
            gradients={n:p.grad.detach().clone() for n,p in cpu.named_parameters()}
            cpu.zero_grad(set_to_none=True);model=cpu.to('mps');tokens=tokens.to('mps');targets=targets.to('mps')
            actual,loss=model(tokens,targets);fd=close(actual,expected);assert abs(loss.item()-ce_cpu)<3e-5
            loss.backward();gd=max(close(p.grad,gradients[n]) for n,p in model.named_parameters());del gradients,expected
        else:
            model=cpu;actual,loss=model(tokens,targets);expected=oracle(model,tokens);fd=close(actual,expected)
            loss.backward();gradients={n:p.grad.detach().clone() for n,p in model.named_parameters()}
            model.zero_grad(set_to_none=True);F.cross_entropy(expected.reshape(-1,50257),targets.reshape(-1)).backward()
            gd=max(close(p.grad,gradients[n]) for n,p in model.named_parameters());del gradients,expected
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            for batch in (1,8):
                x=(torch.arange(batch*11).reshape(batch,11)%400+300).to(device)
                full=model(x)[0];cache=model.caches(batch);pieces=[]
                for start,end in ((0,4),(4,7),(7,11)):pieces.append(model(x[:,start:end],cache=cache,start_pos=start)[0])
                close(full,torch.cat(pieces,1))
                for c in cache:c.reset()
                close(full,model(x,cache=cache)[0]);assert all(c.position==11 for c in cache)
                for c in cache:c.reset()
                before=[(c.c_cache.clone(),c.r_cache.clone(),c.position) for c in cache]
                cache[-1].signature=dict(cache[-1].signature,width=1)
                reject(lambda:model(x[:,:1],cache=cache))
                assert all(torch.equal(c.c_cache,a) and torch.equal(c.r_cache,b) and c.position==pos for c,(a,b,pos) in zip(cache,before))
                cache=model.caches(batch);cache[-1].c_cache=cache[0].c_cache;reject(lambda:model(x[:,:1],cache=cache))
                cache=model.caches(batch);cache[-1].position=1;reject(lambda:model(x[:,:1],cache=cache))
            changed=tokens.clone();changed[:,5:]+=100
            close(model(tokens)[0][:,:5],model(changed)[0][:,:5])
        model(tokens)[0][:,:5].sum().backward()
        future=tokens[:,5:].reshape(-1).unique();assert torch.count_nonzero(model.transformer.wte.weight.grad[future]).item()==0
        model.zero_grad(set_to_none=True)
        payload=dict(architecture=copy.deepcopy(model.signature),model_state=model.state_dict())
        wrong=dict(payload,architecture=dict(model.signature,arm='foreign'))
        reject(lambda:load_weights(model,wrong))
        badcfg=copy.deepcopy(cfg.model);badcfg.dim=128;reject(lambda:GPT(badcfg))
        rows.append(dict(arm=arm,forward_error=fd,max_gradient_error=gd,cache_batches=[1,8],
            cache_reset=True,atomic_foreign_cache_rejection=True,alias_rejection=True,
            forward_and_gradient_causality=True,foreign_checkpoint_rejected=True,parameters=model.signature['parameters']))
        del model,cpu,payload,wrong,cache,actual,loss,tokens,targets,full,pieces,before,changed,x;gc.collect()
        if mode=='mps':torch.mps.synchronize();torch.mps.empty_cache()
    return rows

with session_lock(GLOBAL_LOCK.parent):
    idle();sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'research/body_allocation_model_v1').glob('*.py'))}
    snap=B/(stem+'-sources');snap.mkdir()
    for name in sources:
        p=ROOT/name;(snap/p.name).write_bytes(p.read_bytes())
    try:
        cls=CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=120) if mode=='cpu' else StrictMpsGuard(guard,ROOT)
        with cls,torch.random.fork_rng(devices=[]):rows=run()
        assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in sources.items())
        result=dict(status='passed',mode=mode,cases=rows,sources=sources,training_launched=False,scientific_seeds_assigned=False,holdout_scored=False)
    except Exception as exc:
        result=dict(status='failed',mode=mode,error=repr(exc),sources=sources,no_retry=True)
        with out.open('x') as f:json.dump(result,f,indent=2)
        raise
    with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],mode=mode,cases=rows)))
