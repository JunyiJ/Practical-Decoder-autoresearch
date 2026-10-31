"""Bounded synthetic CPU primitive replay, not a corpus training campaign."""
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from omegaconf import OmegaConf
from research.block_norm_v1.test_model_policy import config
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.block_norm import PLACEMENTS
from research.block_norm_v1.runtime.data.loaders import _configure_splits, _get_batch
from research.block_norm_v1.runtime.train.optimizers import build_optimizer
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.train import _training_contract
from research.block_norm_v1.runtime.utils.checkpoint import save_checkpoint, read_checkpoint, load_checkpoint, load_model_weights, capture_training_state, restore_training_state


def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
    if type(a) is not type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def loader():
    value=SimpleNamespace(batch_size=1,block_size=2,device='cpu',batch_layout='independent')
    _configure_splits(value,torch.arange(512,dtype=torch.long)%37,.8,.2,372301,372302)
    return value


def components(cfg):
    model=GPT(cfg.model);opt=build_optimizer(model,cfg.training)
    return model,opt,LearningRateSchedule(opt,cfg.training),TailWeightAverages(model,cfg.training),loader()


def update(parts,step):
    model,opt,schedule,ema,data=parts
    x,y=_get_batch(data,'train');draw=torch.rand(())
    _,loss=model(x,targets=y);loss=loss*(.9+.2*draw)
    opt.zero_grad(set_to_none=True);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    schedule.apply(step);opt.step();ema.update(step+1)
    return dict(x=x.clone(),y=y.clone(),draw=draw.clone(),loss=loss.detach().clone())


def write_exclusive(path,payload):
    with path.open('xb') as stream:torch.save(payload,stream)
    assert path.stat().st_size<=600*1024**2


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def replay(request):
    root=Path(os.environ['BLOCK_NORM_FIXTURE_ROOT']);before=torch.get_rng_state().clone()
    try:
        torch.manual_seed(372303)
        cfg=config(request.param);cfg.training.device='cpu';cfg.training.max_iters=3
        cfg.training.batch_size=1;cfg.training.batch_layout='independent'
        cfg.training.lr_schedule=dict(kind='linear_cooldown',total_steps=3,start_step=1,min_ratio=.1)
        cfg.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=3,start_step=1,interval=1,half_lives=[200])
        data_path=root/(request.param+'-synthetic-ids.json')
        with data_path.open('x') as stream:json.dump((torch.arange(512)%37).tolist(),stream)
        cfg.data.path=str(data_path);cfg.data.train_fraction=.8;cfg.data.val_fraction=.2
        a=components(cfg);update(a,0)
        contract=_training_contract(cfg,'cpu',a[4])
        state=capture_training_state(a[4],'cpu',contract=contract,training_seconds=0.,metrics=[],best_metric=None,best_checkpoint=None)
        state['weight_averaging_state']=a[3].state_dict()
        raw_path=root/(request.param+'-step1-raw.pt')
        save_checkpoint(raw_path,a[0],a[1],step=1,cfg=cfg,training_state=state,payload_writer=write_exclusive)
        del state
        expected=[update(a,i) for i in (1,2)]
        expected_rng=torch.get_rng_state().clone();expected_loader={k:g.get_state() for k,g in a[4].generators.items()}
        b=components(cfg)
        saved=load_checkpoint(raw_path,b[0],b[1],map_location='cpu')
        restore_training_state(saved['training_state'],b[4],'cpu')
        b[3].load_state_dict(saved['training_state']['weight_averaging_state'],1)
        actual=[update(b,i) for i in (1,2)]
        assert same(expected,actual)
        assert same(a[0].state_dict(),b[0].state_dict())
        assert same(a[1].state_dict(),b[1].state_dict())
        assert same(a[3].state_dict(),b[3].state_dict())
        assert torch.equal(expected_rng,torch.get_rng_state())
        assert all(torch.equal(value,b[4].generators[k].get_state()) for k,value in expected_loader.items())
        # Optimizer loading may share CPU state tensors with its input payload;
        # reload immutable disk bytes before corruption tests after replay updates.
        saved=read_checkpoint(raw_path,map_location='cpu')
        yield dict(cfg=cfg,model=b[0],saved=saved,raw_path=raw_path,parts=b)
    finally:
        torch.set_rng_state(before);gc.collect()


def test_serialized_populated_optimizer_and_interrupted_replay(replay,record_property):
    saved=replay['saved'];model=replay['model']
    panels=saved['optimizer_state']['optimizers']
    assert all(panel['state'] for panel in panels)
    assert sum(len(p['state']) for p in panels)==len(list(model.parameters()))
    names=[n for child in panels for group in child['param_groups'] for n in group['parameter_names']]
    assert len(names)==len(set(names))==len(list(model.parameters()))
    if model.block_norm_spec.bias:
        assert {p+'.bias' for p in PLACEMENTS} <= set(names)
    record_property('exact_model_optimizer_ema_rng_loader_replay',True)
    record_property('raw_fixture_bytes',replay['raw_path'].stat().st_size)
    record_property('synthetic_steps_uninterrupted',3)
    record_property('synthetic_steps_replayed',2)


@pytest.mark.parametrize('mutation',['missing_optimizer','missing_training','missing_rng','missing_loader_rng','bad_step',
    'model_contract','group_order','group_lr','missing_slot','slot_dtype','slot_nonfinite','adam_step','ema_state','ema_counter'])
def test_malformed_resume_rejected_before_model_mutation(replay,mutation):
    saved=replay['saved'];payload=dict(saved);state=dict(saved['training_state']);payload['training_state']=state
    if mutation=='missing_optimizer':del payload['optimizer_state']
    elif mutation=='missing_training':del payload['training_state']
    elif mutation=='missing_rng':del state['torch_rng']
    elif mutation=='missing_loader_rng':state['loader_rng']={'train':state['loader_rng']['train']}
    elif mutation=='bad_step':payload['step']=True
    elif mutation=='model_contract':
        state['contract']=copy.deepcopy(state['contract']);state['contract']['model']['block_normalization']['epsilon']=1e-5
    elif mutation in ('ema_state','ema_counter'):
        if mutation=='ema_state':del state['weight_averaging_state']
        else:state['weight_averaging_state']=dict(state['weight_averaging_state'],snapshots=999)
    else:
        opt=dict(saved['optimizer_state']);opt['optimizers']=[dict(p) for p in opt['optimizers']];payload['optimizer_state']=opt
        panel=opt['optimizers'][-1]  # AdamW
        if mutation in ('group_order','group_lr'):
            panel['param_groups']=copy.deepcopy(panel['param_groups'])
            if mutation=='group_order':panel['param_groups'].reverse()
            else:panel['param_groups'][0]['lr']*=2
        else:
            panel['state']=dict(panel['state']);key=next(iter(panel['state']))
            if mutation=='missing_slot':del panel['state'][key]
            else:
                panel['state'][key]=dict(panel['state'][key]);entry=panel['state'][key]
                if mutation=='slot_dtype':entry['exp_avg']=entry['exp_avg'].half()
                elif mutation=='slot_nonfinite':entry['step']=torch.tensor(float('nan'))
                else:entry['step']=torch.tensor(2.)
    model=replay['model'];name=PLACEMENTS[0]+'.weight';before=model.state_dict()[name].clone();rng=torch.get_rng_state().clone()
    with pytest.raises(ValueError):load_model_weights(model,payload,for_resume=True)
    assert torch.equal(model.state_dict()[name],before) and torch.equal(torch.get_rng_state(),rng)
