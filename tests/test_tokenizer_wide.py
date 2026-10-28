"""Wider geometry, math, lineage and in-memory continuation; no quality inference."""
import copy
import gc
import hashlib
import io
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf
from src.attention.cache import KVCache
from src.models.gpt import GPT
from src.models.rope import apply_rotary_emb
from src.models.tokenizer_architecture import architecture_contract, validate_architecture_model, POLICY, MLA_POLICY
from src.research.tokenizer_wide_config import config
from src.train.optimizers import build_optimizer
from src.utils.checkpoint import load_model_weights, read_checkpoint


@pytest.mark.parametrize('width',[512,1024])
@pytest.mark.parametrize('family',['gqa','mla'])
def test_geometry_storage_optimizer_and_contract(tmp_path,width,family):
    cfg=OmegaConf.create(config(tmp_path,family,width,device='cpu'))
    with torch.device('meta'):
        model=GPT(cfg.model);validate_architecture_model(model)
        opt=build_optimizer(model,cfg.training)
        caches=[KVCache(cfg,'meta',batch_size=8) for _ in range(4)]
    expected={(512,'gqa'):62877696,(512,'mla'):62424192,(1024,'gqa'):148561920,(1024,'mla'):146606208}
    assert sum(p.numel() for p in model.parameters())==expected[width,family]
    assert sum(b.numel()*b.element_size() for c in caches for b in c.buffers())==4*8*128*4*(width if family=='gqa' else width//4+32)
    for g in opt.param_groups:
        n,p=g['parameter_names'][0],g['params'][0]
        scale=1/math.sqrt(width/64) if n.startswith('transformer.h.') and p.ndim==2 else 1
        assert g['lr']==pytest.approx(cfg.training.learning_rate*scale)
    bad=copy.deepcopy(cfg.model);bad.tokenizer_architecture_policy=POLICY if family=='gqa' else MLA_POLICY
    with pytest.raises(ValueError):architecture_contract(bad)
    for k,v in [('dim',256),('dim',768),('dim',True),('n_heads',8),('n_layers',8),('n_kv_heads',1),('dimcq',128),('head_dim_r',32),('weight_tying_kind','scaled_tied'),('attn_type','mha')]:
        bad=copy.deepcopy(cfg.model);bad[k]=v
        with pytest.raises(ValueError):architecture_contract(bad)


@pytest.mark.parametrize('family',['gqa','mla'])
def test_width512_equation_gradients_cache_causality_and_load(tmp_path,family):
    torch.set_num_threads(2);torch.manual_seed(3381)
    cfg=OmegaConf.create(config(tmp_path,family,device='cpu'))
    model=GPT(cfg.model).eval();block=model.transformer.h[0];att=block.attn
    # Attention-only double precision reference keeps vocabulary storage FP32.
    att.double();x=torch.randn(2,7,512,dtype=torch.float64,requires_grad=True)
    a=att(x,freq_cos=block.freq_cos,freq_sin=block.freq_sin)[0]
    if family=='gqa':
        q,k,v=att.qkv(x).split([512,256,256],-1)
        q=q.reshape(2,7,16,32).transpose(1,2)
        k=k.reshape(2,7,8,32).transpose(1,2);v=v.reshape(2,7,8,32).transpose(1,2).repeat_interleave(2,1)
        q=apply_rotary_emb(q,block.freq_cos[:7],block.freq_sin[:7]);k=apply_rotary_emb(k,block.freq_cos[:7],block.freq_sin[:7]).repeat_interleave(2,1)
        score=q@k.transpose(-1,-2)*(.5/math.sqrt(32))
    else:
        cq,ckv=att.dq(x),att.dkv(x)
        q=att.uq(cq).reshape(2,7,16,32).transpose(1,2)
        k=att.uk(ckv).reshape(2,7,16,32).transpose(1,2);v=att.uv(ckv).reshape(2,7,16,32).transpose(1,2)
        qr=att.qr(cq).reshape(2,7,16,32).transpose(1,2);kr=att.kr(x).unsqueeze(1)
        qr=apply_rotary_emb(qr,block.freq_cos[:7],block.freq_sin[:7]);kr=apply_rotary_emb(kr,block.freq_cos[:7],block.freq_sin[:7])
        score=(q@k.transpose(-1,-2)+qr@kr.transpose(-1,-2))*(.5/math.sqrt(64))
    score=score.masked_fill(~torch.ones(7,7,dtype=torch.bool).tril(),float('-inf'))
    b=att.out((score.softmax(-1)@v).transpose(1,2).reshape(2,7,512))
    torch.testing.assert_close(a,b,rtol=1e-12,atol=1e-12)
    targets=[x,*att.parameters()]
    ga=torch.autograd.grad(a.square().sum(),targets,retain_graph=True);gb=torch.autograd.grad(b.square().sum(),targets)
    for av,bv in zip(ga,gb):torch.testing.assert_close(av,bv,rtol=1e-10,atol=1e-10)
    del a,b,ga,gb,targets,x;att.float()
    for batch in [1,8]:
        ids=torch.arange(12).unsqueeze(0).repeat(batch,1)+100
        cache=[KVCache(cfg,'cpu',batch_size=batch) for _ in range(4)]
        with torch.no_grad():
            full=model(ids)[0]
            pieces=[model(ids[:,lo:hi],cache=cache,start_pos=lo)[0] for lo,hi in [(0,3),(3,7),(7,12)]]
            torch.testing.assert_close(torch.cat(pieces,1),full,rtol=1e-5,atol=1e-6)
            changed=ids.clone();changed[:,7:]+=10
            torch.testing.assert_close(model(changed)[0][:,:7],full[:,:7],rtol=0,atol=0)
            for c in cache:c.reset()
            torch.testing.assert_close(model(ids,cache=cache)[0],full,rtol=1e-5,atol=1e-6)
    payload=dict(cfg=OmegaConf.to_container(cfg),model_state=model.state_dict())
    load_model_weights(model,payload)
    payload['cfg']['model']['tokenizer_architecture_policy']=POLICY if family=='gqa' else MLA_POLICY
    with pytest.raises(ValueError):load_model_weights(model,payload)
    if family=='gqa':att.n_kv_heads+=1
    else:att.dimc_kv+=1
    with pytest.raises(ValueError):build_optimizer(model,cfg.training)


def tree_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            h.update(str((v.dtype,tuple(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


@pytest.mark.parametrize('family',['gqa','mla'])
def test_width512_real_trainer_in_memory_resume_and_ema(tmp_path,monkeypatch,family):
    # Intercept disk I/O only; real trainer, tokenizer, sampler, schedule, optimizer,
    # serialization/read_checkpoint and EMA. MPS pilots must separately test files/export.
    import src.train.train as training
    torch.set_num_threads(2)
    cfg=OmegaConf.create(config(tmp_path,family,steps=4,seed=3381,device='cpu'))
    cfg.training.weight_averaging.interval=1;cfg.training.weight_averaging.start_step=1;cfg.training.eval_iters=1;cfg.training.eval_every=2
    full_hash=None;partial=io.BytesIO();mode='full';results=[]
    def save(path,model,optimizer,step,cfg,training_state):
        nonlocal full_hash
        payload=dict(model_state=model.state_dict(),optimizer_state=optimizer.state_dict(),step=step,
            cfg=OmegaConf.to_container(cfg),training_state=training_state,model_math_version=2,parameter_layout_version=2)
        if mode=='partial':
            assert step==1;torch.save(payload,partial);return
        current=tree_hash(dict(model=payload['model_state'],optimizer=payload['optimizer_state'],
            ema=training_state['weight_averaging_state'],rng=training_state['torch_rng'],loader=training_state['loader_rng']))
        if mode=='full':full_hash=current
        else:
            assert current==full_hash
            wa=training_state['weight_averaging_state'];assert wa['snapshots']==4
            derived=dict(cfg=payload['cfg'],model_state=dict(payload['model_state']),averaged_weights={'half_life':200},tokenizer_contract=training_state['contract']['tokenizer'])
            derived['model_state'].update(wa['averages']['200'])
            load_model_weights(model,derived)
            with pytest.raises(ValueError):load_model_weights(model,derived,for_resume=True)
            assert derived['tokenizer_contract']['name']=='gpt2'
        results.append(step)
    monkeypatch.setattr(training,'save_checkpoint',save)
    assert training.run_training(cfg)['completed_steps']==4;gc.collect()
    mode='partial';cfg.training.max_train_seconds=1e-9
    assert training.run_training(cfg)['completed_steps']==1;gc.collect()
    mode='resumed';cfg.training.max_train_seconds=None;cfg.training.resume_from=str(tmp_path/'memory.pt')
    def read(path,map_location='cpu'):
        partial.seek(0);return read_checkpoint(partial,map_location)
    monkeypatch.setattr(training,'read_checkpoint',read)
    assert training.run_training(cfg)['completed_steps']==4
    assert results==[4,4]
