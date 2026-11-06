"""Finite generated MPS repeat/resume/export proof using bounded streaming IO."""
import argparse,copy,gc,hashlib,json,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import config,ROOT,require
from research.body_allocation_model_v1.model import GPT,load_weights
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.body_allocation_model_v1.checkpoint import raw_payload,resume,h200_payload,validate_resume
from research.body_allocation_model_v1.evidence import source_inventory,environment,reference
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter

B=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-body-allocation-resume-v1-001'
PLAN=B/'resume001-plan.json';torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','run']);parser.add_argument('--slot',type=int,choices=range(3));a=parser.parse_args()
CAPS=[1008*1024**2,768*1024**2,768*1024**2];FLOOR=8*1024**3+sum(CAPS)+32*1024**2

def digest(value):
    if isinstance(value,torch.Tensor):
        t=value.detach().cpu().contiguous();return dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(t.numpy().tobytes()).hexdigest())
    if isinstance(value,dict):return {str(k):digest(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [digest(v) for v in value]
    return value

def setup(arm,seed):
    cfg=config(arm,4,seed);torch.manual_seed(seed);torch.mps.manual_seed(seed)
    model=GPT(cfg.model).to('mps');opt=build_optimizer(model,cfg)
    ema=TailWeightAverages(model,cfg.training);schedule=LearningRateSchedule(opt,cfg.training)
    gens={name:torch.Generator().manual_seed(seed+offset) for name,offset in [('train',1),('val',2)]}
    return cfg,model,opt,ema,schedule,gens

def advance(model,opt,ema,schedule,gens,history,start,end):
    for step in range(start,end):
        x=torch.randint(11,400,(2,9),generator=gens['train']).to('mps');y=x+1
        schedule.apply(step);opt.zero_grad(set_to_none=True);_,loss=model(x,y);loss.backward()
        history.append(dict(step=step+1,**clip_gradients(model.parameters(),1.)))
        opt.step();ema.update(step+1)
    torch.mps.synchronize()

def identity(model,opt,ema,history,gens):
    return digest(dict(model=model.state_dict(),optimizer=opt.state_dict(),ema=ema.state_dict(),history=history,
        sampler={n:g.get_state() for n,g in gens.items()},cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state()))

def dispose():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()

def rejected(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Foreign/corrupt resume accepted')

with session_lock(GLOBAL_LOCK.parent) as g:
    idle();environment()
    if a.mode=='prepare':
        require(not PLAN.exists() and not RUN.exists(),'Existing attempt cannot be replaced')
        free=shutil.disk_usage(ROOT).free;require(free>=FLOOR,'Complete generated transport capacity required')
        RUN.mkdir()
        plan=dict(kind='body_allocation_generated_resume_v1',sources=source_inventory(),runner=reference(__file__),
            environment=environment(),arms=['incumbent','shallow','deeper'],fixture_seeds=[91161,91162,91163],
            steps=4,split_after=2,weight_caps=CAPS,required_free=FLOOR,cleanup_target=FLOOR+2*1024**3,free_bytes=free,
            reserve=8*1024**3,metadata=32*1024**2,retention='Keep all three mid/raw/H200 sets until downstream dependency closure and a new exact audit.',
            scientific_seeds_assigned=False,holdout_scored=False)
        with PLAN.open('x') as f:json.dump(plan,f,indent=2);f.write('\n')
        print(json.dumps(dict(status='prepared',required=FLOOR,free=free,cleanup_target=plan['cleanup_target'])))
    else:
        plan=json.loads(PLAN.read_text());require(a.slot is not None,'Exact fixture slot required')
        require(plan['sources']==source_inventory() and plan['runner']==reference(__file__) and plan['environment']==environment(),'Frozen fixture sources/environment changed')
        require(not (RUN/'STOP').exists(),'Closed fixture cannot rerun')
        slot=a.slot;arm=plan['arms'][slot];seed=plan['fixture_seeds'][slot]
        out=B/f'resume001-{arm}.json';guard=B/f'resume001-{arm}-guard.json'
        require(not out.exists() and not guard.exists(),'Existing fixture cannot retry')
        with session_lock(RUN) as r:
            writer=MetadataWriter(RUN,(g,r));sink=CheckpointWriter(writer,'pilots',slot,CAPS[slot])
            try:
                with StrictMpsGuard(guard,ROOT),torch.random.fork_rng(devices=[]):
                    cfg,m,o,e,s,gens=setup(arm,seed);initial=digest(m.state_dict());history=[]
                    advance(m,o,e,s,gens,history,0,4);expected=identity(m,o,e,history,gens)
                    del m,o,e,s,gens;dispose()
                    cfg,m,o,e,s,gens=setup(arm,seed);assert digest(m.state_dict())==initial;history=[]
                    advance(m,o,e,s,gens,history,0,2)
                    mid=sink.save('mid.pt',raw_payload(m,o,e,cfg,2,history,{n:g.get_state() for n,g in gens.items()}))
                    advance(m,o,e,s,gens,history,2,4);assert identity(m,o,e,history,gens)==expected
                    parent=sink.save('parent.pt',raw_payload(m,o,e,cfg,4,history,{n:g.get_state() for n,g in gens.items()}))
                    exported=sink.save('h200.pt',h200_payload(m,e,cfg,parent));assert e.snapshots==3
                    del m,o,e,s,gens;dispose()
                    cfg,m,o,e,s,gens=setup(arm,seed)
                    raw=torch.load(mid['path'],map_location='cpu',weights_only=True)
                    foreign=dict(raw,architecture=dict(raw['architecture'],arm='foreign'))
                    rejected(lambda:resume(m,o,e,foreign,cfg))
                    broken=dict(raw,clipping_history=[]);rejected(lambda:resume(m,o,e,broken,cfg))
                    bad=copy.copy(raw);state=dict(raw['optimizer_state']);children=list(state['optimizers']);child=dict(children[1]);groups=list(child['param_groups']);group=dict(groups[0]);group['eps']=1e-8;groups[0]=group;child['param_groups']=groups;children[1]=child;state['optimizers']=children;bad['optimizer_state']=state
                    rejected(lambda:resume(m,o,e,bad,cfg));del bad,broken,foreign,state,children,child,groups,group
                    assert digest(m.state_dict())==initial
                    step,history,saved=resume(m,o,e,raw,cfg);assert step==2
                    for n,generator in gens.items():generator.set_state(saved[n])
                    del raw,saved;dispose();advance(m,o,e,s,gens,history,2,4)
                    assert identity(m,o,e,history,gens)==expected
                    h=torch.load(exported['path'],map_location='cpu',weights_only=True)
                    rejected(lambda:resume(m,o,e,h,cfg));load_weights(m,h)
                    with torch.no_grad():
                        for batch in (1,8):
                            x=(torch.arange(batch*11).reshape(batch,11)%300+1).to('mps');full=m(x)[0];cache=m.caches(batch)
                            chunks=[m(x[:,:5],cache=cache)[0],m(x[:,5:],cache=cache,start_pos=5)[0]]
                            torch.testing.assert_close(full,torch.cat(chunks,1),rtol=3e-4,atol=3e-5)
                            for c in cache:c.reset()
                            torch.testing.assert_close(full,m(x,cache=cache)[0],rtol=3e-4,atol=3e-5)
                    assert source_inventory()==plan['sources']
                    result=dict(status='passed',arm=arm,plan=reference(PLAN),mid=mid,parent=parent,h200=exported,
                        exact_fresh_repeat=True,exact_serialized_resume=True,identity_fields=['dtype','shape','all tensor bytes','all scalar metadata','samplers','CPU/MPS RNG','complete clipping','optimizer','EMA'],
                        ema_tensors=len(e.parameters),ema_snapshots=e.snapshots,parameter_tensors=len(list(m.parameters())),
                        foreign_geometry_group_history_and_h200_resume_rejected=True,trained_cache_batches=[1,8],
                        reference_storage='Compact dtype/shape/SHA256 identities;no two retained full reference tensor trees.',
                        comparative_memory_saving_claim=False,scientific_seeds_assigned=False,holdout_scored=False)
                result['guard']=reference(guard)
                with out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
                print(json.dumps(result))
            except Exception as exc:
                writer.publish_json('STOP',dict(error=repr(exc),slot=slot,no_retry=True))
                with out.open('x') as f:json.dump(dict(status='failed',error=repr(exc),arm=arm,no_retry=True),f,indent=2)
                raise
