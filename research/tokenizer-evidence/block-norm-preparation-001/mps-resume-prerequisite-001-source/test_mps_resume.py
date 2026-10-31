"""Strict MPS primitive replay with real generated-text loader and RAM transport."""
import copy
import gc
import hashlib
import io
import json
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from src.research.protocol import ROOT
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.runtime.models.output_bias import validate_output_bias_training_data
from research.block_norm_v1.runtime.train.optimizers import build_optimizer
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.train import _make_loader, _training_contract
from research.block_norm_v1.runtime.utils.checkpoint import save_checkpoint, read_checkpoint, load_model_weights, capture_training_state, restore_training_state


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return type(value)((k,cpu(v)) for k,v in value.items())
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)


def compare(actual,expected,atol=2e-6,rtol=2e-4):
    if isinstance(expected,torch.Tensor):
        a=actual.detach().cpu()
        assert a.dtype==expected.dtype and a.shape==expected.shape
        if expected.is_floating_point():
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a,expected,atol=atol,rtol=rtol)
            return float((a-expected).abs().max()) if a.numel() else 0.
        assert torch.equal(a,expected);return 0.
    assert type(actual) is type(expected)
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys()
        return max((compare(actual[k],v,atol,rtol) for k,v in expected.items()),default=0.)
    if isinstance(expected,(list,tuple)):
        assert len(actual)==len(expected)
        return max((compare(a,b,atol,rtol) for a,b in zip(actual,expected)),default=0.)
    assert actual==expected;return 0.


def components(cfg):
    data=_make_loader(cfg,'mps')
    validate_output_bias_training_data(cfg.model,data,cfg.data.path)
    assert set(data.splits)=={'train','val'}
    model=GPT(cfg.model).to('mps');opt=build_optimizer(model,cfg.training)
    return model,opt,LearningRateSchedule(opt,cfg.training),TailWeightAverages(model,cfg.training),data


def update(parts,step):
    model,opt,schedule,ema,data=parts
    x,y=data.get_batch('train');mps_draw=torch.rand((),device='mps');cpu_draw=torch.rand(())
    _,loss=model(x,targets=y);loss=loss*(.8+.1*cpu_draw+.1*mps_draw)
    opt.zero_grad(set_to_none=True);loss.backward()
    assert all(p.device.type=='mps' and p.dtype==torch.float32 and p.grad is not None
        and p.grad.device.type=='mps' and p.grad.dtype==torch.float32 and torch.isfinite(p.grad).all()
        for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    schedule.apply(step);opt.step();ema.update(step+1);torch.mps.synchronize()
    # Adam's scalar step counter stays on CPU; all matrix/vector slots stay on MPS.
    for child in opt.optimizers:
        for slots in child.state.values():
            for name,value in slots.items():
                assert value.dtype==torch.float32 and torch.isfinite(value).all()
                assert value.device.type==('cpu' if name=='step' else 'mps')
    return cpu(dict(x=x,y=y,mps_draw=mps_draw,cpu_draw=cpu_draw,loss=loss))


@pytest.mark.parametrize('kind',['rms_incumbent','layer_matched_epsilon'])
def test_mps_populated_resume_with_rng_rejections(kind,record_property):
    before_cpu=torch.get_rng_state();before_mps=torch.mps.get_rng_state()
    fixture=ROOT/'research_state/block-norm-trainer-fixture-002'/kind
    cfg=OmegaConf.create(json.loads((fixture/'config.json').read_text()))
    cfg.training.device='mps';cfg.training.max_iters=3;cfg.training.seed=372601;cfg.training.eval_seed=372602
    cfg.training.lr_schedule=dict(kind='linear_cooldown',total_steps=3,start_step=1,min_ratio=.1)
    cfg.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=3,start_step=1,interval=1,half_lives=[200])
    buffer=io.BytesIO()
    try:
        torch.manual_seed(372603)
        a=components(cfg);update(a,0)
        state=capture_training_state(a[4],'mps',contract=_training_contract(cfg,'mps',a[4]),
            training_seconds=0.,metrics=[],best_metric=None,best_checkpoint=None)
        state['weight_averaging_state']=a[3].state_dict()
        def writer(path,payload):
            assert buffer.tell()==0
            torch.save(payload,buffer)
            assert buffer.tell()<600*1024**2
        virtual=fixture/'not-written-mps-resume.pt';assert not virtual.exists()
        save_checkpoint(virtual,a[0],a[1],step=1,cfg=cfg,training_state=state,payload_writer=writer)
        del state
        buffer_hash=hashlib.sha256(buffer.getbuffer()).hexdigest()
        expected_trace=[update(a,i) for i in (1,2)]
        expected=cpu(dict(model=a[0].state_dict(),optimizer=a[1].state_dict(),ema=a[3].state_dict()))
        expected_cpu=torch.get_rng_state();expected_mps=torch.mps.get_rng_state()
        expected_loader={k:g.get_state() for k,g in a[4].generators.items()}
        del a;gc.collect();torch.mps.empty_cache()
        b=components(cfg);buffer.seek(0);saved=read_checkpoint(buffer,map_location='cpu')
        original_model=cpu(b[0].state_dict());rng_cpu=torch.get_rng_state();rng_mps=torch.mps.get_rng_state()
        for case in ('missing','empty','dtype','short','long'):
            bad=dict(saved);bad['training_state']=dict(saved['training_state'])
            if case=='missing':del bad['training_state']['mps_rng']
            else:
                valid=saved['training_state']['mps_rng']
                bad['training_state']['mps_rng']={'empty':torch.empty(0,dtype=torch.uint8),
                    'dtype':valid.float(),'short':valid[:-1],'long':torch.cat([valid,valid[:1]])}[case]
            with pytest.raises(ValueError,match='MPS RNG'):load_model_weights(b[0],bad,for_resume=True)
            compare(b[0].state_dict(),original_model,0.,0.)
            assert torch.equal(rng_cpu,torch.get_rng_state()) and torch.equal(rng_mps,torch.mps.get_rng_state())
        del bad,original_model
        load_model_weights(b[0],saved,for_resume=True);b[1].load_state_dict(saved['optimizer_state'])
        restore_training_state(saved['training_state'],b[4],'mps')
        b[3].load_state_dict(saved['training_state']['weight_averaging_state'],1)
        del saved
        actual_trace=[update(b,i) for i in (1,2)]
        errors={name:compare(actual,expected[name]) for name,actual in
            [('model',b[0].state_dict()),('optimizer',b[1].state_dict()),('ema',b[3].state_dict())]}
        for actual,wanted in zip(actual_trace,expected_trace):
            for name in ('x','y','mps_draw','cpu_draw'):compare(actual[name],wanted[name],0.,0.)
        errors['loss']=compare([x['loss'] for x in actual_trace],[x['loss'] for x in expected_trace],1e-5,1e-5)
        assert torch.equal(expected_cpu,torch.get_rng_state()) and torch.equal(expected_mps,torch.mps.get_rng_state())
        assert all(torch.equal(v,b[4].generators[k].get_state()) for k,v in expected_loader.items())
        assert hashlib.sha256(buffer.getbuffer()).hexdigest()==buffer_hash and not virtual.exists()
        record_property('maximum_absolute_replay_errors',errors)
        record_property('exact_batches_cpu_mps_draws_and_rng',True)
        record_property('mps_rng_rejections_before_any_model_change',5)
        record_property('in_memory_serialized_bytes',buffer.getbuffer().nbytes)
        record_property('in_memory_serialized_sha256',buffer_hash)
        record_property('synthetic_updates_including_replay',5)
        del b,expected
    finally:
        buffer.close();torch.set_rng_state(before_cpu);torch.mps.set_rng_state(before_mps)
        gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
