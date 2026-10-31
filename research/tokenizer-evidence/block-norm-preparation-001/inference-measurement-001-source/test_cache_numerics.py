"""New CPU norm/cache prerequisite; independent norm math, shared MLA/FFN math."""
import json
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F
from omegaconf import OmegaConf
from research.block_norm_v1.test_model_policy import config
from research.block_norm_v1.spec import EPSILON
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.tokenizer_architecture import validate_architecture_model
from research.block_norm_v1.runtime.attention.cache import build_caches

LOGIT_ATOL=3e-5
LOGIT_RTOL=2e-4
GRAD_ATOL=3e-6
GRAD_RTOL=5e-4


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def subject(request):
    cfg=config(request.param)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(372401);model=GPT(cfg.model).eval()
    return cfg,model


def tokens(batch,length,seed=372402):
    return torch.randint(0,50257,(batch,length),generator=torch.Generator().manual_seed(seed))


def caches(cfg,batch):return build_caches(cfg,'cpu',batch_size=batch,dtype=torch.float32)


def close(actual,expected,record_property,name,atol=LOGIT_ATOL,rtol=LOGIT_RTOL):
    error=(actual-expected).abs().max().item()
    record_property(name,error)
    torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)


@pytest.mark.parametrize('batch,length,chunks',[(1,17,(5,1,4,7)),(2,17,(5,1,4,7)),(8,17,(5,1,4,7)),(1,128,(31,1,63,33))])
def test_full_cached_and_nonzero_offsets(subject,batch,length,chunks,record_property):
    cfg,model=subject;x=tokens(batch,length);cache=caches(cfg,batch)
    with torch.no_grad():
        full,_=model(x);parts=[];start=0
        for size in chunks:
            part,_=model(x[:,start:start+size],cache=cache,start_pos=start);parts.append(part);start+=size
            assert all(c.position==start for c in cache)
        close(torch.cat(parts,dim=1),full,record_property,'max_full_cached_error')
    assert sum(t.numel()*t.element_size() for c in cache for t in c.buffers())==batch*196608


def test_future_tokens_cannot_change_prefix(subject):
    _,model=subject;x=tokens(2,17);changed=x.clone();changed[:,7:]=(changed[:,7:]+113)%50257
    with torch.no_grad():a,_=model(x);b,_=model(changed)
    assert torch.equal(a[:,:7],b[:,:7])


def test_reset_and_interleaved_requests(subject,record_property):
    cfg,model=subject;a=tokens(2,9);b=tokens(2,9,372403);ca,cb=caches(cfg,2),caches(cfg,2)
    with torch.no_grad():
        fa,_=model(a);fb,_=model(b);pa=[];pb=[];start=0
        for size in (3,1,5):
            y,_=model(a[:,start:start+size],cache=ca,start_pos=start);pa.append(y)
            y,_=model(b[:,start:start+size],cache=cb,start_pos=start);pb.append(y);start+=size
        close(torch.cat(pa,1),fa,record_property,'interleaved_a_error')
        close(torch.cat(pb,1),fb,record_property,'interleaved_b_error')
        saved=[{k:v.clone() for k,v in c.state_dict().items()} for c in cb]
        for c in ca:
            c.reset();assert c.position==0 and all(torch.count_nonzero(v)==0 for v in c.buffers())
        c=tokens(2,5,372404);reset,_=model(c,cache=ca,start_pos=0);fresh,_=model(c,cache=caches(cfg,2),start_pos=0)
        assert torch.equal(reset,fresh)
        assert all(torch.equal(before[k],v) for before,cache in zip(saved,cb) for k,v in cache.state_dict().items())


@pytest.mark.parametrize('mutation',['other_arm','wrong_batch','dtype','layer_order','shared_cache','aliased_storage','stale_position','nonzero_empty','overflow'])
def test_invalid_cache_rejected_without_partial_mutation(subject,mutation):
    cfg,model=subject;cache=caches(cfg,1);start=0;x=tokens(1,3)
    if mutation=='other_arm':cache=caches(config('rms_incumbent' if model.block_norm_spec.bias else 'layer_matched_epsilon'),1)
    elif mutation=='wrong_batch':cache=caches(cfg,2)
    elif mutation=='dtype':cache[3].r_cache=cache[3].r_cache.double()
    elif mutation=='layer_order':cache[2],cache[3]=cache[3],cache[2]
    elif mutation=='shared_cache':cache[3]=cache[2]
    elif mutation=='aliased_storage':cache[3].c_cache=cache[2].c_cache
    elif mutation=='stale_position':cache[3].position=2
    elif mutation=='nonzero_empty':start=1
    else:x=tokens(1,129)
    before=[({k:v.clone() for k,v in c.state_dict().items()},c.position) for c in cache]
    with torch.no_grad(),pytest.raises(ValueError):model(x,cache=cache,start_pos=start)
    for c,(state,position) in zip(cache,before):
        assert c.position==position and all(torch.equal(state[k],v) for k,v in c.state_dict().items())


def reference(model,x,inputs):
    # Validate real modules; do not replace them or bypass production checks.
    # Only normalization arithmetic differs: independent FP64 mean/variance,
    # then cast to FP32 at the branch boundary. Attention/FFN/readout are shared.
    validate_architecture_model(model)
    hidden=model.transformer.wte(x)
    def normalize(value,module,name):
        value.retain_grad();inputs[name]=value
        z=value.double()
        if model.block_norm_spec.bias:z=z-z.mean(dim=-1,keepdim=True)
        z=z*torch.rsqrt((z*z).mean(dim=-1,keepdim=True)+EPSILON)
        z=z*module.weight.double()
        if model.block_norm_spec.bias:z=z+module.bias.double()
        return z.float()
    for i,block in enumerate(model.transformer.h):
        n=normalize(hidden,block.ln1,f'{i}.ln1')
        branch,_=block.attn(n,freq_cos=block.freq_cos,freq_sin=block.freq_sin)
        hidden=hidden+branch
        hidden=hidden+block.mlp(normalize(hidden,block.ln2,f'{i}.ln2'))
    return model.lm_head(model.transformer.ln_f(hidden))/model.width_mult


def test_explicit_normalization_reference_forward_and_gradients(subject,record_property):
    _,model=subject;x=tokens(2,7);target=tokens(2,7,372405);native_inputs={};hooks=[]
    def capture(name):
        def hook(module,args):args[0].retain_grad();native_inputs[name]=args[0]
        return hook
    for i,block in enumerate(model.transformer.h):
        for name in ('ln1','ln2'):hooks.append(getattr(block,name).register_forward_pre_hook(capture(f'{i}.{name}')))
    try:
        model.zero_grad(set_to_none=True);logits,loss=model(x,targets=target);loss.backward()
        native_logits=logits.detach();native_loss=loss.detach()
        gradients={name:p.grad.clone() for name,p in model.named_parameters()}
        input_gradients={name:v.grad.clone() for name,v in native_inputs.items()}
    finally:
        for hook in hooks:hook.remove()
    del logits,loss,native_inputs
    model.zero_grad(set_to_none=True);ref_inputs={};ref=reference(model,x,ref_inputs)
    ref_loss=F.cross_entropy(ref.reshape(-1,50257),target.reshape(-1));ref_loss.backward()
    close(native_logits,ref.detach(),record_property,'reference_logit_error')
    close(native_loss,ref_loss.detach(),record_property,'reference_ce_error',atol=3e-6,rtol=2e-5)
    max_parameter=max_input=0.
    for name,p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all()
        max_parameter=max(max_parameter,(gradients[name]-p.grad).abs().max().item())
        torch.testing.assert_close(gradients[name],p.grad,atol=GRAD_ATOL,rtol=GRAD_RTOL)
    for name,value in ref_inputs.items():
        max_input=max(max_input,(input_gradients[name]-value.grad).abs().max().item())
        torch.testing.assert_close(input_gradients[name],value.grad,atol=GRAD_ATOL,rtol=GRAD_RTOL)
    record_property('max_parameter_gradient_error',max_parameter);record_property('max_norm_input_gradient_error',max_input)
    model.zero_grad(set_to_none=True)


def test_rms_full_model_gradient_is_bit_exact_to_original():
    from src.models.gpt import GPT as OriginalGPT
    root=Path(__file__).resolve().parents[2]
    old_cfg=OmegaConf.create(json.loads((root/'research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json').read_text())['config'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(372406);old=OriginalGPT(old_cfg.model)
        torch.manual_seed(372406);new=GPT(config('rms_incumbent').model)
    x=tokens(1,5);y=tokens(1,5,372407)
    a,la=old(x,targets=y);b,lb=new(x,targets=y);la.backward();lb.backward()
    assert torch.equal(a,b) and torch.equal(la,lb)
    assert dict(old.named_parameters()).keys()==dict(new.named_parameters()).keys()
    assert all(torch.equal(p.grad,dict(new.named_parameters())[name].grad) for name,p in old.named_parameters())
