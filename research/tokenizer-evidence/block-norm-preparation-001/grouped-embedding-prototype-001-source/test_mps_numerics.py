"""Prospective strict-MPS synthetic checks; never a training/resource pilot."""
import copy
import gc
import json
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from research.block_norm_v1.test_model_policy import config
from research.block_norm_v1.test_cache_numerics import reference
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.attention.cache import build_caches

LOGIT_ATOL=3e-4
LOGIT_RTOL=3e-4
GRAD_ATOL=3e-5
GRAD_RTOL=3e-3


def compare(a,b,record_property,name,atol=LOGIT_ATOL,rtol=LOGIT_RTOL):
    a=a.detach().cpu();b=b.detach().cpu()
    record_property(name,(a-b).abs().max().item())
    torch.testing.assert_close(a,b,atol=atol,rtol=rtol)


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def subject(request):
    cfg=config(request.param)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(372501);cpu=GPT(cfg.model).eval()
    gpu=copy.deepcopy(cpu).to('mps').eval()
    assert all(p.device.type=='mps' and p.dtype==torch.float32 for p in gpu.parameters())
    yield cfg,cpu,gpu
    gpu.zero_grad(set_to_none=True);cpu.zero_grad(set_to_none=True)
    torch.mps.synchronize();del gpu,cpu;gc.collect();torch.mps.empty_cache()


def tokens(batch,length,shift=0):
    return ((torch.arange(batch*length).reshape(batch,length)*7+shift)%50257).long()


def cache(cfg,batch,device='mps'):
    return build_caches(cfg,device,batch_size=batch,dtype=torch.float32)


@pytest.mark.parametrize('batch,length,chunks',[(1,17,(5,1,4,7)),(8,17,(5,1,4,7)),(1,128,(31,1,63,33))])
def test_cpu_mps_full_and_mps_cached(subject,batch,length,chunks,record_property):
    cfg,cpu,gpu=subject;x=tokens(batch,length);xm=x.to('mps');c=cache(cfg,batch)
    with torch.no_grad():
        expected,_=cpu(x);actual,_=gpu(xm)
        compare(actual,expected,record_property,'cpu_mps_logit_error')
        start=0;parts=[]
        for size in chunks:
            part,_=gpu(xm[:,start:start+size],cache=c,start_pos=start);parts.append(part);start+=size
            assert all(value.position==start for value in c)
        compare(torch.cat(parts,1),actual,record_property,'mps_full_cached_error')
    assert sum(v.numel()*v.element_size() for a in c for v in a.buffers())==196608*batch
    torch.mps.synchronize()


def test_mps_causality_reset_interleaving_and_bad_cache(subject,record_property):
    cfg,_,gpu=subject;a=tokens(2,9).to('mps');b=tokens(2,9,11).to('mps')
    ca,cb=cache(cfg,2),cache(cfg,2)
    with torch.no_grad():
        fa,_=gpu(a);fb,_=gpu(b);changed=a.clone();changed[:,4:]+=2;fc,_=gpu(changed)
        assert torch.equal(fa[:,:4],fc[:,:4])
        pa=[];pb=[];start=0
        for size in (3,1,5):
            y,_=gpu(a[:,start:start+size],cache=ca,start_pos=start);pa.append(y)
            y,_=gpu(b[:,start:start+size],cache=cb,start_pos=start);pb.append(y);start+=size
        compare(torch.cat(pa,1),fa,record_property,'mps_interleaved_a_error')
        compare(torch.cat(pb,1),fb,record_property,'mps_interleaved_b_error')
        before=[{k:v.clone() for k,v in c.state_dict().items()} for c in cb]
        for c in ca:c.reset();assert c.position==0
        x=tokens(2,5,23).to('mps');reset,_=gpu(x,cache=ca);fresh,_=gpu(x,cache=cache(cfg,2))
        compare(reset,fresh,record_property,'mps_reset_error')
        assert all(torch.equal(state[k],v) for state,c in zip(before,cb) for k,v in c.state_dict().items())
        wrong=cache(cfg,2,'cpu')
        with pytest.raises(ValueError,match='device'):gpu(x,cache=wrong)
        assert all(c.position==0 and all(torch.count_nonzero(v)==0 for v in c.buffers()) for c in wrong)
    torch.mps.synchronize()


def test_mps_gradients_against_cpu_normalization_reference(subject,record_property):
    _,cpu,gpu=subject;x=tokens(1,5);target=tokens(1,5,1)
    cpu.zero_grad(set_to_none=True);inputs={};expected=reference(cpu,x,inputs)
    loss=torch.nn.functional.cross_entropy(expected.reshape(-1,50257),target.reshape(-1));loss.backward()
    gpu.zero_grad(set_to_none=True);actual,actual_loss=gpu(x.to('mps'),targets=target.to('mps'));actual_loss.backward()
    compare(actual,expected,record_property,'mps_reference_logit_error')
    compare(actual_loss,loss,record_property,'mps_reference_ce_error',atol=1e-4,rtol=1e-5)
    maximum=0.
    for (name,p),(other,q) in zip(cpu.named_parameters(),gpu.named_parameters()):
        assert name==other and q.grad is not None and q.grad.device.type=='mps'
        grad=q.grad.cpu();maximum=max(maximum,(grad-p.grad).abs().max().item())
        torch.testing.assert_close(grad,p.grad,atol=GRAD_ATOL,rtol=GRAD_RTOL)
    record_property('mps_reference_max_parameter_gradient_error',maximum)
    cpu.zero_grad(set_to_none=True);gpu.zero_grad(set_to_none=True);torch.mps.synchronize()


def test_original_mps_rms_control_forward_gradient():
    from src.models.gpt import GPT as OriginalGPT
    root=Path(__file__).resolve().parents[2]
    cfg=OmegaConf.create(json.loads((root/'research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json').read_text())['config'])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(372502);old=OriginalGPT(cfg.model).to('mps')
        torch.manual_seed(372502);new=GPT(config('rms_incumbent').model).to('mps')
    x=tokens(1,5).to('mps');y=tokens(1,5,1).to('mps')
    a,la=old(x,targets=y);b,lb=new(x,targets=y);assert torch.equal(a,b) and torch.equal(la,lb)
    la.backward();lb.backward()
    for (name,p),(other,q) in zip(old.named_parameters(),new.named_parameters()):
        assert name==other
        torch.testing.assert_close(p.grad,q.grad,atol=1e-6,rtol=1e-4)
    torch.mps.synchronize()
