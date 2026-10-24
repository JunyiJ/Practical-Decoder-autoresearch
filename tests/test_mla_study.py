import copy
import math
import pytest
import torch
from src.models.gpt import GPT
from src.train.optimizers import build_optimizer
from src.research import mla_study as study
from tests.test_gated_study import config,training


def mla_config(w):
    c=config(w,'swiglu');c.update(n_layers=4,depth_width_policy='depth4_unscaled',attn_type='mla',attention_width_policy='mla_quarter_head32');return c


@pytest.mark.parametrize('w',[64,128,256])
def test_mla_shapes_counts_initialization_optimizer_gradients_checkpoint_causality(w):
    torch.set_num_threads(1);torch.manual_seed(1011);model=GPT(mla_config(w));opt=build_optimizer(model,training())
    att=model.transformer.h[0].attn
    assert (att.dimc_q,att.dimc_kv,att.head_dim,att.head_dim_r)==(w//4,w//4,32,32)
    assert att.dkv.weight.shape==(w//4,w) and att.kr.weight.shape==(32,w)
    assert att.qr.weight.shape==(w,w//4)
    assert sum(p.numel() for p in model.parameters())=={64:196864,128:733568,256:2826880}[w] - 128*w
    # The fixed-row rotary projection retains fan-in weight initialization and
    # width-scaled bias. Unlike other matrices, its aspect ratio changes.
    assert att.kr.weight.abs().max() <= 1/math.sqrt(w)
    assert att.kr.bias.abs().max() <= math.sqrt(w/64)/math.sqrt(w)
    torch.manual_seed(1011);mha_cfg=mla_config(w);mha_cfg.attn_type='mha';del mha_cfg.attention_width_policy
    mha=GPT(mha_cfg)
    torch.testing.assert_close(model.transformer.wte.weight,mha.transformer.wte.weight,rtol=0,atol=0)
    assert not torch.equal(model.transformer.h[0].mlp.linear_gate.weight,mha.transformer.h[0].mlp.linear_gate.weight)
    for group in opt.param_groups:
        name=group['parameter_names'][0];hidden=name.startswith('transformer.h.') and group['params'][0].ndim==2
        assert group['lr']==pytest.approx(.002/math.sqrt(w/64) if hidden else .002)
        assert group['weight_decay']==0
    assert opt.decay==.00006
    x=torch.arange(8).reshape(1,8);out,loss=model(x,x);loss.backward();assert torch.isfinite(out).all()
    for block in model.transformer.h:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in block.parameters())
        for name,p in block.attn.named_parameters():
            if name.endswith('weight'):assert p.grad.abs().sum()>0
    opt.step();opt.zero_grad()
    clone=GPT(mla_config(w));clone.load_state_dict(copy.deepcopy(model.state_dict()));other=build_optimizer(clone,training());other.load_state_dict(copy.deepcopy(opt.state_dict()))
    for m,o in [(model,opt),(clone,other)]:m(x,x)[1].backward();o.step()
    for p,q in zip(model.parameters(),clone.parameters()):torch.testing.assert_close(p,q,rtol=0,atol=0)
    y=x.clone();y[:,4:]=9;torch.testing.assert_close(model(x)[0][:,:4],model(y)[0][:,:4])


@pytest.mark.parametrize('field,value',[('dimckv',16),('dimcq',16),('head_dim_r',32),('attention_width_policy','unknown'),('n_layers',2),('n_heads',4),('depth_width_policy','unknown'),('n_kv_heads',1),('base_width',32)])
def test_mla_contract_rejects_unsupported_combination(field,value):
    c=mla_config(64);c[field]=value
    with pytest.raises(ValueError):build_optimizer(GPT(c),training())


def plan():
    from tests.test_architecture_study import full_plan
    p=full_plan();p.update(study='stage2f-mla-v1',families=study.FAMILIES,seeds=study.SEEDS,head_dimensions=study.HEAD_DIMS,depths=study.DEPTHS,max_trials=57);return p


@pytest.mark.parametrize('field,value',[('max_trials',58),('depths',{'mla':2}),('seeds',[777,888,999])])
def test_frozen_protocol(field,value):
    p=plan();study.validate_plan(p);p[field]=value
    with pytest.raises(ValueError):study.validate_plan(p)


def test_selection_requires_all_four_comparators_and_all_seeds():
    p=plan();rs=[dict(family=f,lr=lr,width=w,seed=s,status='completed',metric=2+abs(lr-.0025)*100-(.02 if f=='mla' else 0)) for f,lr,w,s in study.local_jobs(p)]
    assert len(rs)==27 and study.freeze_selection(p,rs)['winner']=='mla'
    for r in rs:
        if r['family']=='shallow':r['metric']-=.03
    assert study.freeze_selection(p,rs)['winner']=='incumbent'
    for r in rs:
        if r['family']=='shallow':r['metric']+=.03
        if r['family']=='mla' and r['seed']==1033:r['metric']+=.021
    assert study.freeze_selection(p,rs)['winner']=='incumbent'


def test_serial_engine_propagates_mla_contract_all_57_trials(tmp_path,monkeypatch):
    import json
    from omegaconf import OmegaConf
    from src.research import optimizer_study as engine
    from src.research.runner import atomic_json
    p=plan();p.update(source_root=str(tmp_path),source_sha256='a'*64,data_sha256=None)
    p['config']['data']=dict(path=str(tmp_path/'corpus.txt'),train_fraction=.8,val_fraction=.1)
    runtime={'packages':{'torch':'test'}}
    monkeypatch.setattr(engine,'_preflight',lambda spec:None)
    monkeypatch.setattr(engine,'_fingerprint',lambda spec:{'runtime':runtime})
    monkeypatch.setattr(engine,'_verify_checkpoints',lambda *args:None)
    original_request=engine._request;seen=[]
    def request(spec,row,d):
        req=original_request(spec,row,d);req['job']=row.copy();return req
    monkeypatch.setattr(engine,'_request',request)
    def launch(path,output,*args,**kwargs):
        req=json.loads(path.read_text());job=req['job'];seen.append(job)
        cfg=OmegaConf.create(req['config']['model']);cfg.vocab_size=65
        model=GPT(cfg);att=model.transformer.h[0].attn
        build_optimizer(model,training())
        assert att.n_heads==job['width']//32
        assert len(model.transformer.h)==study.DEPTHS[job['family']]
        assert cfg.attn_type==('mla' if job['family']=='mla' else 'mha')
        if job['family']=='mla':assert att.dimc_kv==att.dimc_q==job['width']//4
        if job['width']>64:
            frozen=json.loads((output/'selection.json').read_text())
            assert frozen['winner']=='mla' and len([r for r in seen if r['width']==64])==27
            assert job['lr']==frozen['choices'][job['family']]['lr']
        metric=2+abs(job['lr']-.0025)*100-(.02 if job['family']=='mla' else 0)
        atomic_json(path.parent/'result.json',{'metric':metric,'training_time_sec':3,'rss_mb':100,
             'runtime':{'torch':'test','cpu_threads':req['cpu_threads'],'requested_device':'mps'}})
        return 'completed',None
    monkeypatch.setattr(engine,'_launch',launch)
    monkeypatch.setattr(engine,'validate_result',lambda raw,*args:{k:raw[k] for k in ['metric','training_time_sec','rss_mb']})
    args=dict(suite_builder=study.suite_spec,job_builder=study.local_jobs,
              selection_builder=study.freeze_selection,report_builder=study.summarize)
    state=engine.run(p,tmp_path/'run',**args)
    assert state['status']=='completed' and len(seen)==57
    engine.run(p,tmp_path/'run',**args);assert len(seen)==57


@pytest.mark.parametrize('w',[64,128,256])
def test_quarter_mla_cache_matches_full_and_independent_attention_formula(w):
    from omegaconf import OmegaConf
    from src.attention.cache import KVCache
    from src.models.rope import apply_rotary_emb
    torch.set_num_threads(1);torch.manual_seed(1011)
    c=mla_config(w);model=GPT(c).eval();x=torch.arange(8).reshape(1,8)
    cfg=OmegaConf.create({'model':c,'training':{'batch_size':1}})
    caches=[KVCache(cfg,'cpu') for _ in range(4)]
    assert caches[0].c_cache.shape==(1,c.block_size,w//4)
    with torch.no_grad():
        full=model(x)[0];cached=torch.cat([model(x[:,i:i+1],cache=caches,start_pos=i)[0] for i in range(8)],dim=1)
        torch.testing.assert_close(full,cached,rtol=1e-5,atol=1e-6)
        block=model.transformer.h[0];a=block.attn;h=block.ln1(model.transformer.wte(x));cq=a.dq(h);ckv=a.dkv(h)
        q=a.uq(cq).view(1,8,w//32,32).transpose(1,2);k=a.uk(ckv).view(1,8,w//32,32).transpose(1,2);v=a.uv(ckv).view(1,8,w//32,32).transpose(1,2)
        qr=apply_rotary_emb(a.qr(cq).view(1,8,w//32,32).transpose(1,2),block.freq_cos[:8],block.freq_sin[:8]);kr=apply_rotary_emb(a.kr(h).unsqueeze(1),block.freq_cos[:8],block.freq_sin[:8])
        # Independent sum of content and rotary scores equals concatenation.
        score=(q@k.transpose(-2,-1)+qr@kr.transpose(-2,-1))/math.sqrt(64)
        score=score.masked_fill(~torch.ones(8,8,dtype=torch.bool).tril(),float('-inf'))
        expected=a.out((score.softmax(-1)@v).transpose(1,2).reshape(1,8,w))
        actual=a(h,freq_cos=block.freq_cos,freq_sin=block.freq_sin)[0]
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
