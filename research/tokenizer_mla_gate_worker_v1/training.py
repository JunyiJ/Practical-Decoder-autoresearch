import gc
import math
import time
import torch
from research.body_allocation_model_v1.contract import require,plain
from research.tokenizer_mla_gate_model_v1.model import GPT
from research.tokenizer_mla_gate_model_v1.optimizer import build_optimizer
from research.tokenizer_mla_gate_resume_v1.data import load,architecture
from research.tokenizer_mla_gate_resume_v1.checkpoint import raw_payload,h200_payload
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages,averaging_contract
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule,schedule_contract
from research.gradient_clip_model_v1.clipping import clip_gradients


def run(cfg,sink,emit):
    require(type(cfg.training.max_iters) is int and cfg.training.max_iters in (4,20),'Only four/twenty-update prerequisites admitted')
    require(cfg.training.device=='mps' and cfg.training.strict_device is True and cfg.training.resume_from is None,'Fresh strict MPS worker required')
    architecture(cfg);steps=cfg.training.max_iters;start={4:2,20:13}[steps]
    require(averaging_contract(cfg.training)==dict(kind='spaced_tail_ema',total_steps=steps,start_step=start,interval=1,half_lives=[200]),'Exact prerequisite H200 tail required')
    require(schedule_contract(cfg.training)==dict(kind='linear_cooldown',start_step=start,total_steps=steps,min_ratio=.1),'Exact prerequisite cooldown required')
    require(type(cfg.training.grad_clip) is float and cfg.training.grad_clip==1. and type(cfg.training.eval_every) is int and cfg.training.eval_every>=1
        and type(cfg.training.eval_iters) is int and cfg.training.eval_iters>=1,'Original clipping and finite sampled evaluation required')
    data=load(cfg,'mps');torch.default_generator.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    model=GPT(plain(cfg.gate_model)).to('mps');opt=build_optimizer(model,cfg.training)
    ema=TailWeightAverages(model,cfg.training);schedule=LearningRateSchedule(opt,cfg.training)
    history=[];updates=[];metrics=[];began=time.monotonic()
    for i in range(steps):
        model.train();torch.mps.synchronize();started=time.monotonic();x,y=data.get_batch('train');opt.zero_grad(set_to_none=True)
        _,loss=model(x,y,include_aux_loss=False)
        require(bool(torch.isfinite(loss)),'Finite native training CE required');loss.backward()
        active=[n for n,p in model.named_parameters() if p.grad is not None]
        require(active==[n for n,_ in model.named_parameters()],'Every gate/control learned parameter must receive a gradient before update')
        history.append(dict(step=i+1,**clip_gradients(model.parameters(),1.),active_parameter_names=active,ce_loss=float(loss.detach())))
        ratio=schedule.apply(i);opt.step();ema.update(i+1);torch.mps.synchronize();seconds=time.monotonic()-started;updates.append(seconds)
        require(time.monotonic()-began<600,'Original prerequisite runtime exceeded')
        if i%cfg.training.eval_every==0 or i==steps-1:
            model.eval();scores=[]
            with torch.no_grad():
                for _ in range(cfg.training.eval_iters):
                    vx,vy=data.get_batch('val');value=float(model(vx,vy,include_aux_loss=False)[1]);require(math.isfinite(value),'Finite sampled CE required');scores.append(value)
            row=dict(step=i+1,targets=(i+1)*1024,train_ce=history[-1]['ce_loss'],
                     validation_ce=sum(scores)/len(scores),lr_multiplier=ratio,update_seconds=seconds)
            emit(row);metrics.append(row)
    torch.mps.synchronize();training_seconds=time.monotonic()-began
    raw=raw_payload(model,opt,ema,cfg,steps,history,{n:g.get_state() for n,g in data.generators.items()})
    parent=sink.save('parent.pt',raw);del raw;gc.collect()
    # Own immutable H200 schema uses the path/hash ancestry; writer size is checked separately.
    parent_identity={k:parent[k] for k in ('path','sha256')}
    exported=sink.save('h200.pt',h200_payload(model,ema,cfg,parent_identity))
    summary=dict(kind='tokenizer_mla_gate_prerequisite_worker_v1',config=plain(cfg),parent=parent,h200=exported,step=steps,targets=steps*1024,
                 ema_snapshots=ema.snapshots,parameter_tensors=len(ema.parameters),architecture=model.signature,data=data.data_identity,
                 metrics=metrics,training_seconds=training_seconds,update_seconds=updates,holdout_scored=False,
                 clipped_updates=sum(r['coefficient']<1 for r in history),final_online_train_ce=history[-1]['ce_loss'],
                 parameter_gradient_steps=opt.gradient_steps)
    del model,opt,ema,schedule,data;gc.collect();torch.mps.empty_cache()
    return summary
