"""Own full-horizon matched-matrix FFN scientific worker using frozen own loss/state equations."""
import gc
import math
import time
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.body_allocation_model_v1.contract import ROOT
import torch
from research.body_allocation_model_v1.contract import require,plain
from research.relu_squared_model_v1.model import GPT
from research.relu_squared_model_v1.optimizer import build_optimizer
from research.relu_squared_resume_v1.data import load,architecture,model_spec
from research.relu_squared_resume_v1 import checkpoint as C
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages,averaging_contract
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule,schedule_contract
from research.gradient_clip_model_v1.clipping import clip_gradients


def validate_config(cfg):
    architecture(cfg)
    require(type(cfg.training.max_iters) is int and cfg.training.max_iters==1200,'Only complete1200 horizon supported')
    require(cfg.training.device=='mps' and cfg.training.strict_device is True and cfg.training.resume_from is None,'Fresh strict MPS scientific worker required')
    require(averaging_contract(cfg.training)==dict(kind='spaced_tail_ema',total_steps=1200,start_step=800,interval=10,half_lives=[200]),'Exact41snapshot H200 required')
    require(schedule_contract(cfg.training)==dict(kind='linear_cooldown',start_step=800,total_steps=1200,min_ratio=.1),'Exact full cooldown required')
    require(type(cfg.training.grad_clip) is float and cfg.training.grad_clip==1.
        and type(cfg.training.eval_every) is int and cfg.training.eval_every==300
        and type(cfg.training.eval_iters) is int and cfg.training.eval_iters==1,'Full clipping and five sampled evaluations required')
    from . import contract as Q
    from research.body_allocation_model_v1.contract import fingerprint
    require(type(cfg.training.learning_rate) is float and cfg.training.learning_rate in Q.LRS,'Exact finite three-rate allowance required')
    require('relu_squared_execution' not in plain(cfg),'Prerequisite execution cannot masquerade as quality')
    require(plain(cfg.relu_squared_quality_execution)==dict(kind='ffn_full_quality_execution_v1',sources_sha256=fingerprint(Q.sources()),target_contract_sha256=Q.DESIGN['target_contract_sha256']),'Exact own scientific execution binding required')
    return 1200


class Counter:
    def __init__(self,cap):self.size=0;self.cap=cap
    def write(self,value):
        n=memoryview(value).nbytes;require(self.size+n<=self.cap,'Actual serialized layout exceeds pair cap');self.size+=n;return n
    def flush(self):pass
    def tell(self):return self.size


def save(sink,name,payload,used):
    counter=Counter(min(sink.allowance-used,C.FILE_CAPS[payload['kind']]));torch.save(payload,counter)
    ref=sink.save(name,payload);require(ref['size']==counter.size,'Counted and actual archive sizes differ')
    return ref,counter.size


def run(cfg,sink,emit):
    steps=validate_config(cfg)
    with CpuPrerequisiteGuard(Path(cfg.training.summary_file).parent/'cpu-data-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        data=load(cfg,'mps')
    torch.default_generator.manual_seed(cfg.rng_identity.global_cpu);torch.mps.manual_seed(cfg.rng_identity.global_mps)
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
        require(time.monotonic()-began<7200,'Finite scientific time exceeded')
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
    summary=dict(kind='relu_squared_quality_worker_v1',config=plain(cfg),parent=parent,h200=h200,step=steps,targets=steps*1024,
        ema_snapshots=ema.snapshots,parameter_tensors=len(ema.parameters),architecture=model.signature,data=data.data_identity,metrics=metrics,
        training_seconds=training_seconds,update_seconds=updates,holdout_scored=False,clipped_updates=sum(r['coefficient']<1 for r in history),
        final_online_train_ce=history[-1]['ce_loss'],parameter_gradient_steps=opt.gradient_steps,serialized_bytes=dict(raw=raw_bytes,h200=h200_bytes,pair=raw_bytes+h200_bytes))
    del model,opt,ema,schedule,data;gc.collect();torch.mps.empty_cache();return summary
