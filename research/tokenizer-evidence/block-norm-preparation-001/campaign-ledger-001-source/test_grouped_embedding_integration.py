"""Versioned policy binding; no optimizer steps or quality data."""
import copy
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from src.research.protocol import ROOT
from research.block_norm_v1.test_cache_numerics import reference
from research.block_norm_v1.test_model_policy import replace
from research.block_norm_v1.runtime.models.block_norm import POLICY, GROUPED_POLICY, contract, checkpoint_signature
from research.block_norm_v1.runtime.models.grouped_embedding import EMBEDDING_SPEC, GroupedMpsEmbedding
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.tokenizer_architecture import validate_architecture_model
from research.block_norm_v1.runtime.attention.cache import build_caches
from research.block_norm_v1.runtime.utils.checkpoint import save_checkpoint, read_checkpoint, load_model_weights


def config(arm, grouped=True):
    cfg=OmegaConf.create(json.loads((ROOT/f'research_state/block-norm-trainer-fixture-002/{arm}/config.json').read_text()))
    assert cfg.model.tokenizer_architecture_policy==POLICY
    if grouped:
        cfg.model.tokenizer_architecture_policy=GROUPED_POLICY
        cfg.model.embedding_backward=dict(EMBEDDING_SPEC)
    return cfg


def sha_tensor(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def state_hash(model):
    return {name:sha_tensor(t) for name,t in model.state_dict().items()}


def emit(event,**fields):
    with Path(os.environ['BLOCK_NORM_TRACE_PATH']).open('a') as stream:
        stream.write(json.dumps(dict(event=event,**fields),sort_keys=True)+'\n')
        assert stream.tell()<2*1024**2


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def subject(request):
    configs=[config(request.param,False),config(request.param,True)]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(373001);old=GPT(configs[0].model).eval();rng_old=torch.get_rng_state().clone()
        torch.manual_seed(373001);new=GPT(configs[1].model).eval();rng_new=torch.get_rng_state().clone()
    yield request.param,configs,old,new,rng_old,rng_new
    old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def test_init_and_cpu_native_gradient_exact(subject,record_property):
    arm,cfgs,old,new,a,b=subject
    assert torch.equal(a,b) and state_hash(old)==state_hash(new)
    assert type(old.transformer.wte) is torch.nn.Embedding and type(new.transformer.wte) is GroupedMpsEmbedding
    assert sum(p.numel() for p in new.parameters())==28492416+(2048 if arm=='layer_matched_epsilon' else 0)
    before=checkpoint_signature(old)
    assert before['architecture']['policy']==POLICY and 'embedding_backward' not in before['architecture']
    assert checkpoint_signature(new)['architecture']['embedding_backward']==EMBEDDING_SPEC
    x=(torch.arange(34)%7).reshape(2,17);y=(x+19)%50257
    old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)
    a,la=old(x,targets=y);b,lb=new(x,targets=y)
    assert torch.equal(a,b) and torch.equal(la,lb)
    la.backward();lb.backward()
    assert all(torch.equal(p.grad,q.grad) for p,q in zip(old.parameters(),new.parameters()))
    record_property('old_new_cpu_forward_gradient_rng','bit_exact')
    emit('cpu_equivalence',arm=arm,signature_old=before,signature_new=checkpoint_signature(new),state_sha256=state_hash(new))
    old.zero_grad(set_to_none=True);new.zero_grad(set_to_none=True)


def test_raw_memory_transport_and_policy_rejection(subject,record_property):
    arm,cfgs,old,new,_,_=subject
    models=[old,new]
    initial=[state_hash(m) for m in models]
    for index,(cfg,model) in enumerate(zip(cfgs,models)):
        buffer=io.BytesIO()
        save_checkpoint('in-memory-only.pt',model,cfg=cfg,payload_writer=lambda path,payload:torch.save(payload,buffer))
        binary=buffer.getvalue();buffer.seek(0);payload=read_checkpoint(buffer)
        assert 'optimizer_state' not in payload and 'training_state' not in payload
        with torch.no_grad():next(model.parameters()).add_(1.)
        load_model_weights(model,payload)
        assert state_hash(model)==initial[index]
        other=models[1-index];before=state_hash(other)
        with pytest.raises(ValueError):load_model_weights(other,payload)
        assert state_hash(other)==before
        if index==1:
            for mutation in ('missing_signature','missing_spec','wrong_spec','signature_wrong_spec'):
                bad=dict(payload);bad['cfg']=copy.deepcopy(payload['cfg'])
                bad['block_normalization_signature']=copy.deepcopy(payload['block_normalization_signature'])
                if mutation=='missing_signature':del bad['block_normalization_signature']
                elif mutation=='missing_spec':del bad['cfg']['model']['embedding_backward']
                elif mutation=='wrong_spec':bad['cfg']['model']['embedding_backward']['version']=True
                else:bad['block_normalization_signature']['architecture']['embedding_backward']['cpu_backward']='unknown'
                with pytest.raises(ValueError):load_model_weights(new,bad)
                assert state_hash(new)==initial[1]
            with pytest.raises(ValueError):load_model_weights(new,payload,for_resume=True)
            assert state_hash(new)==initial[1]
        emit('raw_memory_transport',arm=arm,policy=cfg.model.tokenizer_architecture_policy,
             bytes=len(binary),sha256=hashlib.sha256(binary).hexdigest(),roundtrip_exact=True,cross_policy_rejected_before_mutation=True)
        del payload,binary,buffer
    record_property('raw_inference_transport','both_policies_exact_cross_policy_rejected')


@pytest.mark.parametrize('mutation',['native_module','spec','padding','max_norm','sparse','frequency'])
def test_actual_embedding_rejections(subject,mutation):
    _,_,_,new,_,_=subject;embedding=new.transformer.wte
    native=torch.nn.Embedding(50257,256,device='meta');native.weight=embedding.weight
    changes={'native_module':(new.transformer,'wte',native),
             'spec':(embedding,'embedding_backward_spec',dict(EMBEDDING_SPEC,version=True)),
             'padding':(embedding,'padding_idx',0),'max_norm':(embedding,'max_norm',1.),
             'sparse':(embedding,'sparse',True),'frequency':(embedding,'scale_grad_by_freq',True)}
    with replace(*changes[mutation]):
        with pytest.raises(ValueError):validate_architecture_model(new)
        with pytest.raises(ValueError):checkpoint_signature(new)
        with pytest.raises(ValueError):new(torch.zeros(1,2,dtype=torch.long))


@pytest.mark.parametrize('batch,length,chunks',[(1,128,(31,1,63,33)),(8,17,(5,1,4,7))])
def test_mps_forward_cache_equivalence_and_signature_rejection(subject,batch,length,chunks,record_property):
    arm,cfgs,old,new,_,_=subject
    gpu_old=copy.deepcopy(old).to('mps');gpu_new=copy.deepcopy(new).to('mps')
    x=((torch.arange(batch*length)*7)%13).reshape(batch,length).to('mps')
    caches=[build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32) for cfg in cfgs]
    maximum=0.
    with torch.no_grad():
        a,_=gpu_old(x);b,_=gpu_new(x);assert torch.equal(a,b)
        start=0;parts=[]
        for size in chunks:
            a,_=gpu_old(x[:,start:start+size],cache=caches[0],start_pos=start)
            b,_=gpu_new(x[:,start:start+size],cache=caches[1],start_pos=start)
            assert torch.equal(a,b);parts.append(b);start+=size
        full,_=gpu_new(x);actual=torch.cat(parts,1)
        maximum=(actual-full).abs().max().item()
        torch.testing.assert_close(actual,full,atol=3e-4,rtol=3e-4)
        for model,wrong in [(gpu_new,caches[0]),(gpu_old,caches[1])]:
            for c in wrong:c.reset()
            before=[state_hash(c) for c in wrong]
            with pytest.raises(ValueError):model(x[:,:1],cache=wrong,start_pos=0)
            assert before==[state_hash(c) for c in wrong] and all(c.position==0 for c in wrong)
    record_property('max_full_cached_error',maximum)
    emit('mps_forward_cache',arm=arm,shape=list(x.shape),native_equivalence_exact=True,max_full_cached_error=maximum)
    del gpu_old,gpu_new,caches,parts,a,b,actual,full
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


@pytest.mark.parametrize('batch,length',[(1,128),(8,17)])
def test_full_model_mps_gradient_repeats_and_reference(subject,batch,length,record_property):
    arm,_,_,cpu,_,_=subject
    cpu.zero_grad(set_to_none=True);gpu=copy.deepcopy(cpu).to('mps')
    x=((torch.arange(batch*length)*7)%13).reshape(batch,length);target=(x+19)%50257
    inputs={};expected=reference(cpu,x,inputs)
    loss=torch.nn.functional.cross_entropy(expected.reshape(-1,50257),target.reshape(-1));loss.backward()
    hashes=[];maximum=0.
    for repeat in range(4):
        gpu.zero_grad(set_to_none=True);actual,actual_loss=gpu(x.to('mps'),targets=target.to('mps'));actual_loss.backward()
        torch.testing.assert_close(actual.detach().cpu(),expected.detach(),atol=3e-4,rtol=3e-4)
        torch.testing.assert_close(actual_loss.detach().cpu(),loss.detach(),atol=1e-4,rtol=1e-5)
        current={};error=0.
        for (name,p),(other,q) in zip(cpu.named_parameters(),gpu.named_parameters()):
            assert name==other and q.grad is not None and q.grad.device.type=='mps'
            grad=q.grad.cpu();current[name]=sha_tensor(grad)
            error=max(error,(grad-p.grad).abs().max().item())
            torch.testing.assert_close(grad,p.grad,atol=3e-5,rtol=3e-3)
        emit('mps_full_gradient',arm=arm,shape=list(x.shape),repeat=repeat,gradients_sha256=current,max_reference_error=error)
        hashes.append(current);maximum=max(maximum,error)
        assert current==hashes[0]
        del actual,actual_loss
    record_property('max_reference_parameter_gradient_error',maximum)
    record_property('exact_full_gradient_repeats',4)
    cpu.zero_grad(set_to_none=True);gpu.zero_grad(set_to_none=True)
    del gpu,expected,loss,inputs
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


@pytest.mark.parametrize('mutation',['missing','extra','bool_version','wrong_dtype','wrong_backend','old_policy','unknown_policy'])
def test_config_rejections(mutation):
    cfg=config('rms_incumbent').model
    if mutation=='missing':del cfg.embedding_backward
    elif mutation=='extra':cfg.embedding_backward.extra=0
    elif mutation=='bool_version':cfg.embedding_backward.version=True
    elif mutation=='wrong_dtype':cfg.embedding_backward.dtype='bfloat16'
    elif mutation=='wrong_backend':cfg.embedding_backward.cpu_backward='grouped'
    elif mutation=='old_policy':cfg.tokenizer_architecture_policy=POLICY
    else:cfg.tokenizer_architecture_policy='unknown'
    with pytest.raises(ValueError):contract(cfg)


def test_both_arms_share_initialization_and_embedding_policy():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(373001);rms=GPT(config('rms_incumbent').model);a=torch.get_rng_state().clone()
        torch.manual_seed(373001);layer=GPT(config('layer_matched_epsilon').model);b=torch.get_rng_state().clone()
    assert torch.equal(a,b)
    sa,sb=rms.state_dict(),layer.state_dict()
    assert len(set(sb)-set(sa))==8 and all(torch.equal(v,sb[n]) for n,v in sa.items())
    assert rms.transformer.wte.embedding_backward_spec==layer.transformer.wte.embedding_backward_spec==EMBEDDING_SPEC


def test_gradient_budget_and_inference_cpu_reference():
    embedding=GroupedMpsEmbedding(50257,256)
    ids=torch.zeros(9,128,dtype=torch.long)
    with pytest.raises(ValueError,match='budget'):embedding(ids)
    with torch.no_grad():assert torch.equal(embedding(ids),torch.nn.functional.embedding(ids,embedding.weight))
