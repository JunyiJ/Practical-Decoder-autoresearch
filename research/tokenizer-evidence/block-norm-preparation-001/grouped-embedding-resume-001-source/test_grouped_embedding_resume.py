"""New v2 both-arm primitive MPS repeat/resume gate; closed v1 tests are not run."""
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
from research.block_norm_v1.test_grouped_embedding_integration import config
from research.block_norm_v1.test_mps_resume import components, cpu, compare
from research.block_norm_v1.test_mps_resume_trace import state_info, rng_info, tensor_info
from research.block_norm_v1.runtime.train.train import _training_contract
from research.block_norm_v1.runtime.utils.checkpoint import (
    save_checkpoint,read_checkpoint,load_model_weights,capture_training_state,restore_training_state)
from research.block_norm_v1.runtime.utils.block_norm_resume import validate_resume


@pytest.mark.parametrize('kind',['rms_incumbent','layer_matched_epsilon'])
def test_v2_mps_repeat_and_populated_resume(kind,record_property):
    path=Path(os.environ['BLOCK_NORM_TRACE_PATH']);buffer=io.BytesIO()
    before_cpu=torch.get_rng_state();before_mps=torch.mps.get_rng_state()
    cfg=config(kind)
    cfg.training.device='mps';cfg.training.max_iters=3;cfg.training.seed=373101;cfg.training.eval_seed=373102
    assert cfg.training.batch_size==1 and cfg.model.block_size==128
    cfg.training.lr_schedule=dict(kind='linear_cooldown',total_steps=3,start_step=1,min_ratio=.1)
    cfg.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=3,start_step=1,interval=1,half_lives=[200])
    frozen=json.loads(Path(os.environ['BLOCK_NORM_PLAN_PATH']).read_text())
    assert OmegaConf.to_container(cfg,resolve=True)==frozen['configs'][kind]
    refs={};states={};initial=None;completed=0
    virtual=ROOT/f'research_state/block-norm-trainer-fixture-002/{kind}/not-written-v2-resume.pt'
    assert not virtual.exists()
    with path.open('a') as stream:
        def emit(event,**value):
            stream.write(json.dumps(dict(event=event,normalization=kind,**value),sort_keys=True)+'\n');stream.flush()
            assert stream.tell()<8*1024**2
        def step(parts,index,label):
            nonlocal completed
            model,opt,schedule,ema,data=parts
            pre_rng=rng_info(data)
            x,y=data.get_batch('train');mps_draw=torch.rand((),device='mps');cpu_draw=torch.rand(())
            _,loss=model(x,targets=y);loss=loss*(.8+.1*cpu_draw+.1*mps_draw)
            opt.zero_grad(set_to_none=True);loss.backward();torch.mps.synchronize()
            gradients={n:tensor_info(p.grad) for n,p in model.named_parameters()}
            assert all(v['finite'] and v['device']=='mps:0' and v['dtype']=='torch.float32' for v in gradients.values())
            observation=cpu(dict(x=x,y=y,mps_draw=mps_draw,cpu_draw=cpu_draw,loss=loss))
            pre_clip=dict(pre_rng=pre_rng,post_draw_rng=rng_info(data),gradients=gradients)
            differing=[] if label=='reference' else [n for n,v in gradients.items() if v!=refs[index]['pre_clip']['gradients'][n]]
            emit('before_clip',trajectory=label,step=index+1,observation={k:tensor_info(v) for k,v in observation.items()},
                 **pre_clip,differing_gradient_names=differing)
            if label!='reference':
                assert pre_rng==refs[index]['pre_clip']['pre_rng']
                assert pre_clip['post_draw_rng']==refs[index]['pre_clip']['post_draw_rng']
                for name in ('x','y','mps_draw','cpu_draw'):compare(observation[name],refs[index]['observation'][name],0.,0.)
                compare(observation['loss'],refs[index]['observation']['loss'],1e-5,1e-5)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            ratio=schedule.apply(index);opt.step();ema.update(index+1);torch.mps.synchronize();completed+=1
            for child in opt.optimizers:
                for slots in child.state.values():
                    for name,value in slots.items():
                        assert value.dtype==torch.float32 and torch.isfinite(value).all()
                        assert value.device.type==('cpu' if name=='step' else 'mps')
            after=state_info(parts)
            tensor_diffs=[] if label=='reference' else [n for n,v in after['tensors'].items() if v!=states[index]['tensors'][n]]
            emit('after_update',trajectory=label,step=index+1,clip_norm=float(norm.cpu()),lr_multiplier=ratio,
                 state=after,differing_state_tensor_names=tensor_diffs)
            if label=='reference':refs[index]=dict(pre_clip=pre_clip,observation=observation);states[index]=after
            else:
                assert {k:v for k,v in after.items() if k!='tensors'}=={k:v for k,v in states[index].items() if k!='tensors'}
                assert after['tensors'].keys()==states[index]['tensors'].keys()
                for n,v in after['tensors'].items():
                    assert {k:x for k,x in v.items() if k!='sha256'}=={k:x for k,x in states[index]['tensors'][n].items() if k!='sha256'}
            return after
        def final_compare(parts,expected,label):
            errors={name:compare(actual,expected[name],2e-6,2e-4) for name,actual in [
                ('model',parts[0].state_dict()),('optimizer',parts[1].state_dict()),('ema',parts[3].state_dict())]}
            assert rng_info(parts[4])==states[2]['rng']
            emit('final_acceptance',trajectory=label,max_absolute_errors=errors,atol=2e-6,rtol=2e-4,final_rng_exact=True)
            return errors
        try:
            torch.manual_seed(373103);a=components(cfg);initial=state_info(a)
            emit('initialized',trajectory='reference',state=initial)
            step(a,0,'reference')
            state=capture_training_state(a[4],'mps',contract=_training_contract(cfg,'mps',a[4]),
                training_seconds=0.,metrics=[],best_metric=None,best_checkpoint=None)
            state['weight_averaging_state']=a[3].state_dict()
            def writer(_,payload):
                assert buffer.tell()==0;torch.save(payload,buffer);assert buffer.tell()<600*1024**2
            save_checkpoint(virtual,a[0],a[1],step=1,cfg=cfg,training_state=state,payload_writer=writer)
            del state
            archive_sha=hashlib.sha256(buffer.getbuffer()).hexdigest()
            emit('serialized_in_memory',bytes=buffer.getbuffer().nbytes,sha256=archive_sha,rng_unchanged=rng_info(a[4])==states[0]['rng'])
            assert rng_info(a[4])==states[0]['rng']
            step(a,1,'reference');step(a,2,'reference')
            expected=cpu(dict(model=a[0].state_dict(),optimizer=a[1].state_dict(),ema=a[3].state_dict()))
            del a;gc.collect();torch.mps.empty_cache()
            torch.manual_seed(373103);b=components(cfg);start=state_info(b)
            emit('initialized',trajectory='fresh_repeat',state=start,exact_reference=start==initial)
            assert start==initial
            for i in range(3):step(b,i,'fresh_repeat')
            repeat_errors=final_compare(b,expected,'fresh_repeat')
            del b;gc.collect();torch.mps.empty_cache()
            c=components(cfg);buffer.seek(0);saved=read_checkpoint(buffer,map_location='cpu')
            unmodified=state_info(c)
            for case in ('missing','empty','dtype','short','long','cpu_device','other_mps','torch_version'):
                bad=dict(saved);bad['training_state']=dict(saved['training_state'])
                if case in ('cpu_device','other_mps','torch_version'):
                    bound=dict(saved['training_state']['contract']);bad['training_state']['contract']=bound
                    if case=='torch_version':bound['torch_version']='unmatched-version'
                    else:bound['device']='cpu' if case=='cpu_device' else 'mps:1'
                elif case=='missing':del bad['training_state']['mps_rng']
                else:
                    valid=saved['training_state']['mps_rng']
                    bad['training_state']['mps_rng']={'empty':torch.empty(0,dtype=torch.uint8),
                        'dtype':valid.float(),'short':valid[:-1],'long':torch.cat([valid,valid[:1]])}[case]
                with pytest.raises(ValueError):load_model_weights(c[0],bad,for_resume=True)
                assert state_info(c)==unmodified
                emit('rejected_before_mutation',case=case)
            del bad,unmodified
            alias=dict(saved);alias['training_state']=dict(saved['training_state'])
            alias['training_state']['contract']=dict(saved['training_state']['contract'],device='mps:0')
            validate_resume(c[0],alias);del alias
            load_model_weights(c[0],saved,for_resume=True);c[1].load_state_dict(saved['optimizer_state'])
            restore_training_state(saved['training_state'],c[4],'mps')
            c[3].load_state_dict(saved['training_state']['weight_averaging_state'],1)
            restored=state_info(c)
            emit('restored_before_update',state=restored,exact_reference=restored==states[0],
                differing_tensor_names=[n for n,v in restored['tensors'].items() if v!=states[0]['tensors'][n]],
                differing_metadata=[k for k in restored if k!='tensors' and restored[k]!=states[0][k]])
            assert restored==states[0]
            del saved
            for i in (1,2):step(c,i,'resumed')
            resume_errors=final_compare(c,expected,'resumed')
            assert hashlib.sha256(buffer.getbuffer()).hexdigest()==archive_sha and not virtual.exists()
            emit('finished',updates=completed,acceptance_gate_passed=True,binary_files=0)
            record_property('repeat_errors',repeat_errors);record_property('resume_errors',resume_errors)
            record_property('restored_model_optimizer_ema_rng_metadata_exact',True)
            record_property('rng_device_version_rejections_before_mutation',8)
            record_property('synthetic_updates_including_repeat_resume',completed)
            record_property('archive_bytes',buffer.getbuffer().nbytes);record_property('archive_sha256',archive_sha)
            del c,expected,refs,states
        finally:
            buffer.close();torch.set_rng_state(before_cpu);torch.mps.set_rng_state(before_mps)
            gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
