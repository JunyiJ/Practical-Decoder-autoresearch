"""One RMS diagnostic, not a retry or acceptance of the failed resume gate."""
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import torch
from omegaconf import OmegaConf
from src.research.protocol import ROOT
from research.block_norm_v1.test_mps_resume import components, cpu
from research.block_norm_v1.runtime.train.train import _training_contract
from research.block_norm_v1.runtime.utils.checkpoint import save_checkpoint, read_checkpoint, load_model_weights, capture_training_state, restore_training_state


def tensor_info(value):
    x=value.detach().cpu().contiguous()
    return dict(sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),shape=list(x.shape),
        dtype=str(x.dtype),device=str(value.device),stride=list(value.stride()),
        finite=bool(torch.isfinite(x).all()))


def rng_info(data):
    return dict(cpu=tensor_info(torch.get_rng_state()),mps=tensor_info(torch.mps.get_rng_state()),
        loader={k:tensor_info(g.get_state()) for k,g in data.generators.items()})


def state_info(parts):
    model,opt,schedule,ema,data=parts
    tensors={'model/'+n:tensor_info(v) for n,v in model.state_dict().items()}
    groups=[]
    for index,child in enumerate(opt.optimizers):
        for group in child.param_groups:
            groups.append(dict(optimizer=index,**{k:v for k,v in group.items() if k!='params'}))
            for name,param in zip(group['parameter_names'],group['params']):
                for key,value in child.state.get(param,{}).items():
                    tensors[f'optimizer/{index}/{name}/{key}']=tensor_info(value)
    for h,panel in ema.averages.items():
        for n,v in panel.items():tensors[f'ema/{h}/{n}']=tensor_info(v)
    return dict(tensors=tensors,groups=groups,schedule_base_lrs=list(schedule.base_lrs),
        ema_spec=ema.spec,ema_completed=ema.completed_steps,ema_snapshots=ema.snapshots,rng=rng_info(data))


def comparison(actual,expected):
    result={}
    for name,value in actual.items():
        a=value.detach().cpu();b=expected[name]
        delta=(a-b).abs();threshold=2e-6+2e-4*b.abs()
        result[name]=dict(max_abs=float(delta.max()),different_elements=int((a!=b).sum()),
            outside_failed_gate_tolerance=int((delta>threshold).sum()),elements=a.numel())
    return result


def test_rms_resume_diagnostic_with_fresh_repeat_control(record_property):
    path=Path(os.environ['BLOCK_NORM_TRACE_PATH']);buffer=io.BytesIO()
    before_cpu=torch.get_rng_state();before_mps=torch.mps.get_rng_state()
    cfg=OmegaConf.create(json.loads((ROOT/'research_state/block-norm-trainer-fixture-002/rms_incumbent/config.json').read_text()))
    cfg.training.device='mps';cfg.training.max_iters=3;cfg.training.seed=372601;cfg.training.eval_seed=372602
    cfg.training.lr_schedule=dict(kind='linear_cooldown',total_steps=3,start_step=1,min_ratio=.1)
    cfg.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=3,start_step=1,interval=1,half_lives=[200])
    refs={};reference_states={};initial=None
    with path.open('x') as stream:
        def emit(kind,**value):
            stream.write(json.dumps(dict(event=kind,**value),sort_keys=True)+'\n');stream.flush()
            assert stream.tell()<8*1024**2
        def step(parts,index,label):
            model,opt,schedule,ema,data=parts
            pre_rng=rng_info(data)
            x,y=data.get_batch('train');mps_draw=torch.rand((),device='mps');cpu_draw=torch.rand(())
            _,loss=model(x,targets=y);loss=loss*(.8+.1*cpu_draw+.1*mps_draw)
            opt.zero_grad(set_to_none=True);loss.backward();torch.mps.synchronize()
            gradients={n:p.grad for n,p in model.named_parameters()}
            info={n:tensor_info(g) for n,g in gradients.items()}
            grad_compare=comparison(gradients,refs[index]['gradients']) if label!='reference' else None
            emit('before_clip',arm=label,step=index+1,pre_rng=pre_rng,post_draw_rng=rng_info(data),
                x=tensor_info(x),y=tensor_info(y),cpu_draw=float(cpu_draw),mps_draw=float(mps_draw.cpu()),
                loss=float(loss.detach().cpu()),gradients=info,versus_reference=grad_compare)
            assert all(g.device.type=='mps' and g.dtype==torch.float32 for g in gradients.values())
            assert all(v['finite'] for v in info.values())
            if label=='reference':refs[index]=dict(gradients=cpu(gradients))
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            ratio=schedule.apply(index);opt.step();ema.update(index+1);torch.mps.synchronize()
            after=state_info(parts)
            parameter_compare=comparison(dict(model.named_parameters()),refs[index]['parameters']) if label!='reference' else None
            emit('after_update',arm=label,step=index+1,clip_norm=float(norm.cpu()),lr_multiplier=ratio,
                state=after,versus_reference=parameter_compare)
            if label=='reference':
                refs[index]['parameters']=cpu(dict(model.named_parameters()));reference_states[index]=after
            return after
        try:
            torch.manual_seed(372603);a=components(cfg);initial=state_info(a)
            emit('initialized',arm='reference',state=initial)
            step(a,0,'reference')
            state=capture_training_state(a[4],'mps',contract=_training_contract(cfg,'mps',a[4]),
                training_seconds=0.,metrics=[],best_metric=None,best_checkpoint=None)
            state['weight_averaging_state']=a[3].state_dict()
            def writer(_,payload):
                assert buffer.tell()==0;torch.save(payload,buffer);assert buffer.tell()<600*1024**2
            virtual=ROOT/'research_state/block-norm-trainer-fixture-002/rms_incumbent/not-written-trace.pt'
            assert not virtual.exists()
            save_checkpoint(virtual,a[0],a[1],step=1,cfg=cfg,training_state=state,payload_writer=writer)
            del state
            archive_sha=hashlib.sha256(buffer.getbuffer()).hexdigest()
            emit('serialized_in_memory',bytes=buffer.getbuffer().nbytes,sha256=archive_sha,
                rng_unchanged=rng_info(a[4])==reference_states[0]['rng'])
            step(a,1,'reference');step(a,2,'reference')
            del a;gc.collect();torch.mps.empty_cache()
            torch.manual_seed(372603);b=components(cfg)
            start=state_info(b);emit('initialized',arm='fresh_repeat',state=start,exact_reference=start==initial)
            for i in range(3):step(b,i,'fresh_repeat')
            del b;gc.collect();torch.mps.empty_cache()
            c=components(cfg);buffer.seek(0);saved=read_checkpoint(buffer,map_location='cpu')
            load_model_weights(c[0],saved,for_resume=True);c[1].load_state_dict(saved['optimizer_state'])
            restore_training_state(saved['training_state'],c[4],'mps')
            c[3].load_state_dict(saved['training_state']['weight_averaging_state'],1)
            restored=state_info(c)
            emit('restored_before_update',state=restored,exact_reference=restored==reference_states[0],
                differing_tensor_names=[n for n,v in restored['tensors'].items() if v!=reference_states[0]['tensors'][n]],
                differing_metadata=[k for k in restored if k!='tensors' and restored[k]!=reference_states[0][k]])
            del saved
            for i in (1,2):step(c,i,'resumed')
            assert hashlib.sha256(buffer.getbuffer()).hexdigest()==archive_sha and not virtual.exists()
            emit('diagnostic_finished',updates=8,normalization='rms_incumbent',acceptance_gate_passed=False,
                scope='Diagnostic observation only; closed failed gate unchanged; no numerical tolerance widened.')
            record_property('diagnostic_completed',True);record_property('trace_path',str(path.relative_to(ROOT)))
            record_property('trace_sha256',hashlib.sha256(path.read_bytes()).hexdigest())
            del c,refs
        finally:
            buffer.close();torch.set_rng_state(before_cpu);torch.mps.set_rng_state(before_mps)
            gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
