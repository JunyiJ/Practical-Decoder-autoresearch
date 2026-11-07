"""Finite real-data readout prerequisites using frozen own loss/state equations."""
import gc
import math
import time
import torch
from research.body_allocation_model_v1.contract import require,plain
from research.readout_mixture_model_v1 import GPT,build_optimizer
from research.readout_mixture_resume_v1.data import load,architecture,model_spec
from research.readout_mixture_resume_v1 import checkpoint as C
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages,averaging_contract
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule,schedule_contract
from research.gradient_clip_model_v1.clipping import clip_gradients


def validate_config(cfg):
    architecture(cfg)
    require(type(cfg.training.max_iters) is int and cfg.training.max_iters in (4,20),'Only four/twenty-update prerequisites admitted')
    require(cfg.training.device=='mps' and cfg.training.strict_device is True and cfg.training.resume_from is None,'Fresh strict MPS prerequisite required')
    steps=cfg.training.max_iters;start={4:2,20:13}[steps]
    require(averaging_contract(cfg.training)==dict(kind='spaced_tail_ema',total_steps=steps,start_step=start,interval=1,half_lives=[200]),'Exact finite H200 tail required')
    require(schedule_contract(cfg.training)==dict(kind='linear_cooldown',start_step=start,total_steps=steps,min_ratio=.1),'Exact finite cooldown required')
    require(type(cfg.training.grad_clip) is float and cfg.training.grad_clip==1. and type(cfg.training.eval_every) is int and cfg.training.eval_every>=1
        and type(cfg.training.eval_iters) is int and cfg.training.eval_iters>=1,'Full clipping and finite evaluation required')
    binding=plain(cfg.readout_execution)
    require(type(binding) is dict and set(binding)=={'kind','sources_sha256','runner_sha256'} and binding['kind']=='readout_prerequisite_execution_v1'
        and all(type(binding[k]) is str and len(binding[k])==64 and set(binding[k])<=set('0123456789abcdef') for k in ['sources_sha256','runner_sha256']),'Full worker/runner binding required')
    return steps


class Counter:
    def __init__(self,cap):self.size=0;self.cap=cap
    def write(self,value):
        n=memoryview(value).nbytes;require(self.size+n<=self.cap,'Actual serialized layout exceeds pair cap');self.size+=n;return n
    def flush(self):pass
    def tell(self):return self.size


def save(sink,name,payload,used):
    counter=Counter(sink.allowance-used);torch.save(payload,counter)
    ref=sink.save(name,payload);require(ref['size']==counter.size,'Counted and actual archive sizes differ')
    return ref,counter.size


def run(cfg,sink,emit):
    steps=validate_config(cfg);data=load(cfg,'mps')
    torch.default_generator.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    model=GPT(model_spec(cfg),initializer_seed=cfg.initializer_seed).to('mps');opt=build_optimizer(model,cfg.training)
    ema=TailWeightAverages(model,cfg.training);schedule=LearningRateSchedule(opt,cfg.training)
    history=[];updates=[];metrics=[];began=time.monotonic()
    for i in range(steps):
        model.train();torch.mps.synchronize();started=time.monotonic();x,y=data.get_batch('train');opt.zero_grad()
        loss=model.loss(x,y);require(bool(torch.isfinite(loss)),'Finite own native-distribution NLL required');loss.backward()
        active=[n for n,p in model.named_parameters() if p.grad is not None]
        require(active==[n for n,_ in model.named_parameters()],'Every learned parameter must receive a gradient')
        history.append(dict(step=i+1,**clip_gradients(model.parameters(),1.),active_parameter_names=active,ce_loss=float(loss.detach())))
        ratio=schedule.apply(i);opt.step();ema.update(i+1);torch.mps.synchronize();seconds=time.monotonic()-started;updates.append(seconds)
        require(time.monotonic()-began<600,'Finite prerequisite time exceeded')
        if i%cfg.training.eval_every==0 or i==steps-1:
            model.eval();scores=[]
            with torch.no_grad():
                for _ in range(cfg.training.eval_iters):
                    vx,vy=data.get_batch('val');value=float(model.loss(vx,vy));require(math.isfinite(value),'Finite sampled NLL required');scores.append(value)
            row=dict(step=i+1,targets=(i+1)*1024,train_ce=history[-1]['ce_loss'],validation_ce=sum(scores)/len(scores),lr_multiplier=ratio,update_seconds=seconds)
            emit(row);metrics.append(row)
    torch.mps.synchronize();training_seconds=time.monotonic()-began;opt.zero_grad()
    raw=C.raw_payload(model,opt,ema,cfg,steps,history,data);parent,raw_bytes=save(sink,'parent.pt',raw,0);del raw;gc.collect()
    parent_identity=dict(kind=C.RAW,path=parent['path'],sha256=parent['sha256'],bytes=parent['size'])
    exported=C.h200_payload(model,ema,cfg,parent_identity);h200,h200_bytes=save(sink,'h200.pt',exported,raw_bytes);del exported
    summary=dict(kind='readout_mixture_prerequisite_worker_v1',config=plain(cfg),parent=parent,h200=h200,step=steps,targets=steps*1024,
        ema_snapshots=ema.snapshots,parameter_tensors=len(ema.parameters),architecture=model.signature,data=data.data_identity,metrics=metrics,
        training_seconds=training_seconds,update_seconds=updates,holdout_scored=False,clipped_updates=sum(r['coefficient']<1 for r in history),
        final_online_train_ce=history[-1]['ce_loss'],parameter_gradient_steps=opt.gradient_steps,serialized_bytes=dict(raw=raw_bytes,h200=h200_bytes,pair=raw_bytes+h200_bytes))
    del model,opt,ema,schedule,data;gc.collect();torch.mps.empty_cache();return summary
