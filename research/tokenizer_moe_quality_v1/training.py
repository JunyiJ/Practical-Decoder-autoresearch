import gc
import math
import time
import torch
from research.body_allocation_model_v1.contract import require,plain
from research.tokenizer_moe_model_v1.model import GPT
from research.tokenizer_moe_model_v1.optimizer import build_optimizer
from research.tokenizer_moe_resume_v1.data import load,architecture
from research.tokenizer_moe_resume_v1.checkpoint import raw_payload,h200_payload
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages,averaging_contract
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule,schedule_contract
from research.gradient_clip_model_v1.clipping import clip_gradients


def run(cfg,sink,emit):
    require(type(cfg.training.max_iters) is int and cfg.training.max_iters==1200,'Only complete frozen scientific horizons admitted')
    require(cfg.training.device=='mps' and cfg.training.strict_device is True and cfg.training.resume_from is None,'Fresh strict MPS worker required')
    architecture(cfg);steps=cfg.training.max_iters;start=800
    require(averaging_contract(cfg.training)==dict(kind='spaced_tail_ema',total_steps=steps,start_step=start,interval=10,half_lives=[200]),'Exact scientific H200 tail required')
    require(schedule_contract(cfg.training)==dict(kind='linear_cooldown',start_step=start,total_steps=steps,min_ratio=.1),'Exact scientific cooldown required')
    require(type(cfg.training.grad_clip) is float and cfg.training.grad_clip==1. and type(cfg.training.eval_every) is int and cfg.training.eval_every>=1
        and type(cfg.training.eval_iters) is int and cfg.training.eval_iters>=1,'Original clipping and finite sampled evaluation required')
    data=load(cfg,'mps');torch.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    model=GPT(plain(cfg.moe_model)).to('mps');opt=build_optimizer(model,cfg.training)
    ema=TailWeightAverages(model,cfg.training);schedule=LearningRateSchedule(opt,cfg.training)
    history=[];updates=[];metrics=[];began=time.monotonic()
    for i in range(steps):
        model.train();torch.mps.synchronize();started=time.monotonic();x,y=data.get_batch('train');opt.zero_grad(set_to_none=True)
        before=[b.mlp.assignment_counts.clone() for b in model.transformer.h] if model.signature['arm']=='moe' else None
        _,loss,parts=model(x,y,include_training_objective=True,return_loss_breakdown=True)
        require(bool(torch.isfinite(loss)),'Finite training objective required');loss.backward()
        active=[n for n,p in model.named_parameters() if p.grad is not None]
        routes=[(b.mlp.assignment_counts-old).cpu().tolist() for b,old in zip(model.transformer.h,before)] if before is not None else None
        history.append(dict(step=i+1,**clip_gradients(model.parameters(),1.),active_parameter_names=active,route_counts=routes,
                            ce_loss=float(parts['ce_loss'].detach()),aux_loss=float(parts['aux_loss'].detach()) if parts['aux_loss'] is not None else None,total_loss=float(loss.detach())))
        ratio=schedule.apply(i);opt.step();ema.update(i+1);torch.mps.synchronize();seconds=time.monotonic()-started;updates.append(seconds)
        require(time.monotonic()-began<7200,'Original scientific runtime exceeded')
        if i%cfg.training.eval_every==0 or i==steps-1:
            model.eval();scores=[]
            with torch.no_grad():
                for _ in range(cfg.training.eval_iters):
                    vx,vy=data.get_batch('val');value=float(model(vx,vy,include_aux_loss=False)[1]);require(math.isfinite(value),'Finite sampled CE required');scores.append(value)
            row=dict(step=i+1,targets=(i+1)*1024,train_ce=history[-1]['ce_loss'],train_aux_loss=history[-1]['aux_loss'],train_total_loss=history[-1]['total_loss'],
                     validation_ce=sum(scores)/len(scores),lr_multiplier=ratio,update_seconds=seconds)
            emit(row);metrics.append(row)
    torch.mps.synchronize();training_seconds=time.monotonic()-began
    raw=raw_payload(model,opt,ema,cfg,steps,history,{n:g.get_state() for n,g in data.generators.items()})
    parent=sink.save('parent.pt',raw);del raw;gc.collect()
    exported=sink.save('h200.pt',h200_payload(model,ema,cfg,parent))
    summary=dict(kind='tokenizer_moe_quality_worker_v1',config=plain(cfg),parent=parent,h200=exported,step=steps,targets=steps*1024,
                 ema_snapshots=ema.snapshots,parameter_tensors=len(ema.parameters),architecture=model.signature,data=data.data_identity,
                 metrics=metrics,training_seconds=training_seconds,update_seconds=updates,holdout_scored=False,
                 clipped_updates=sum(r['coefficient']<1 for r in history),final_online_train_ce=history[-1]['ce_loss'],
                 final_online_aux_loss=history[-1]['aux_loss'],final_online_total_loss=history[-1]['total_loss'],
                 parameter_gradient_steps=opt.gradient_steps,route_counts=[b.mlp.assignment_counts.cpu().tolist() for b in model.transformer.h] if model.signature['arm']=='moe' else None)
    del model,opt,ema,schedule,data;gc.collect();torch.mps.empty_cache()
    return summary
