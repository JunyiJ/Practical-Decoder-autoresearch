"""One fixed-horizon real-data trajectory; bounded IO is supplied by the caller."""
import gc,time,math
import torch
from research.body_allocation_model_v1.contract import require,plain
from research.body_allocation_model_v1.model import GPT
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.body_allocation_model_v1.checkpoint import raw_payload,h200_payload
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.gradient_clip_model_v1.clipping import clip_gradients
from .data import load

def run(cfg,sink,emit):
    require(cfg.training.device=='mps' and cfg.training.resume_from is None,'Fresh strict MPS trajectory required')
    data=load(cfg,'mps');torch.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    m=GPT(cfg.model).to('mps');o=build_optimizer(m,cfg);e=TailWeightAverages(m,cfg.training);s=LearningRateSchedule(o,cfg.training)
    history=[];updates=[];metrics=[];began=time.monotonic()
    for i in range(cfg.training.max_iters):
        m.train();torch.mps.synchronize();start=time.monotonic();x,y=data.get_batch('train')
        _,loss=m(x,y);require(bool(torch.isfinite(loss)),'Nonfinite training loss')
        o.zero_grad(set_to_none=True);loss.backward();history.append(dict(step=i+1,**clip_gradients(m.parameters(),1.)))
        multiplier=s.apply(i);o.step();e.update(i+1);torch.mps.synchronize();seconds=time.monotonic()-start;updates.append(seconds)
        require(time.monotonic()-began<7200,'Original training time limit exceeded')
        if i%cfg.training.eval_every==0 or i==cfg.training.max_iters-1:
            m.eval();scores=[]
            with torch.no_grad():
                for _ in range(cfg.training.eval_iters):
                    vx,vy=data.get_batch('val');value=float(m(vx,vy)[1]);require(math.isfinite(value),'Nonfinite sampled validation');scores.append(value)
            row=dict(step=i+1,targets=(i+1)*1024,train_ce=float(loss),validation_ce=sum(scores)/len(scores),lr_multiplier=multiplier,update_seconds=seconds)
            emit(row);metrics.append(row)
    torch.mps.synchronize();train_seconds=time.monotonic()-began
    raw=raw_payload(m,o,e,cfg,cfg.training.max_iters,history,{n:g.get_state() for n,g in data.generators.items()})
    parent=sink.save('parent.pt',raw);del raw;gc.collect()
    exported=sink.save('h200.pt',h200_payload(m,e,cfg,parent))
    result=dict(kind='body_allocation_worker_v1',config=plain(cfg),parent=parent,h200=exported,
        step=cfg.training.max_iters,targets=cfg.training.max_iters*1024,ema_snapshots=e.snapshots,
        parameter_tensors=len(e.parameters),architecture=m.signature,metrics=metrics,
        training_seconds=train_seconds,update_seconds=updates,data=data.data_identity,holdout_scored=False,
        clipped_updates=sum(r['coefficient']<1 for r in history),final_online_train_ce=float(loss))
    del m,o,e,s,data;gc.collect();torch.mps.empty_cache()
    return result
