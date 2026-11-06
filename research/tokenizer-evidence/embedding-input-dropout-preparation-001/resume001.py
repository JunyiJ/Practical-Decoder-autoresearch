"""Four-update real-data stochastic repeat/serialized-resume identities, no weight files."""
import copy,gc,hashlib,io,json
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.body_allocation_worker_v1.data import load as data_view
from research.embedding_input_dropout_v1.model import GPT,load_weights
from research.embedding_input_dropout_resume_v1.contract import config,sources
from research.embedding_input_dropout_resume_v1.checkpoint import raw_payload,h200_payload,validate_resume,resume
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;torch.set_num_threads(4)
def pack(value):
    if isinstance(value,torch.Tensor):
        t=value.detach().cpu().contiguous();return dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value,dict):return [[type(k).__name__,repr(k),pack(value[k])] for k in sorted(value,key=lambda k:(type(k).__name__,repr(k))) ]
    if isinstance(value,(list,tuple)):return [pack(v) for v in value]
    require(value is None or type(value) in (int,float,str,bool),'Unsupported scalar identity');return value
def digest(value):return hashlib.sha256(json.dumps(pack(value),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()
def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value
def publish(p,d):
    raw=(json.dumps(d,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<=512*1024,'Bounded result')
    with p.open('xb') as f:f.write(raw)
def start(cfg,seed=True):
    if seed:torch.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    d=data_view(cfg,'mps');m=GPT(plain(cfg.input_embedding_regularization)).to('mps');o=build_optimizer(m,cfg);e=TailWeightAverages(m,cfg.training)
    return m,o,e,d,LearningRateSchedule(o,cfg.training)
def advance(m,o,e,d,s,first,end,history):
    for i in range(first,end):
        m.train();x,y=d.get_batch('train');o.zero_grad(set_to_none=True);_,loss=m(x,y)
        require(bool(torch.isfinite(loss)),'Finite real-data loss required');loss.backward();history.append(dict(step=i+1,**clip_gradients(m.parameters(),1.)))
        s.apply(i);o.step();e.update(i+1)
        # Actual eval uses its independent sampler and consumes no dropout RNG.
        m.eval()
        with torch.no_grad():vx,vy=d.get_batch('val');require(bool(torch.isfinite(m(vx,vy)[1])),'Finite sampled eval required')
    torch.mps.synchronize()
def endpoint(m,o,e,d,cfg,h):
    raw=raw_payload(m,o,e,cfg,4,h,{n:g.get_state() for n,g in d.generators.items()});raw_id=digest(raw)
    hp=h200_payload(m,e,cfg,dict(kind='in_memory_fixture_no_parent_file'));h_id=digest(hp);load_weights(m,hp);m.eval()
    with torch.no_grad():logits=m(torch.arange(16,device='mps').reshape(1,16))[0];logit_id=digest(logits)
    return dict(raw_model_optimizer_ema_history_sampler_cpu_mps_rng=raw_id,h200=h_id,h200_logits=logit_id,
                parameters=len(e.parameters),snapshots=e.snapshots,clipping_updates=len(h))
def rejected(fn):
    try:fn()
    except (TypeError,ValueError):return
    raise AssertionError('Resume forgery accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not(E/'resume001-plan.json').exists(),'Closed resume fixture cannot retry');pins=sources()
    proofs=[json.load(open(E/f'correctness001-{d}.json')) for d in ('cpu','mps')]
    for proof in proofs:
        require(proof['status']=='passed' and all(pins[k]==v for k,v in proof['sources'].items()),'Exact earlier model proof plus explicit additive transport required')
    publish(E/'resume001-plan.json',dict(kind='input_dropout_stochastic_serialized_resume_fixture',source=reference(__file__),sources=pins,probabilities=[0.,.1],updates=4,split=2,
        fixture_seed=600404,scientific_seed=False,real_data=True,training_quality_claim=False,
        protocol='Per arm:4updates,fresh repeat4updates,serialized2+2. Compare exact dtype/shape/SHA256 state bytes and scalar/RNG/samplers/history/EMA/H200/logits. Retain only compact digests;release CPU tensor trees and serialized buffer after decode.',
        resources='SingleMPS worker/4threads,600s/RSS4GiB/MPS6GiB/reserve8GiB,512KiBmetadata per record,no filesystem tensor outputs. No measured comparative memory-saving claim.'))
    try:
        with StrictMpsGuard(E/'resume001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
            reports=[]
            for probability in (0.,.1):
                cfg=config(probability);runs=[]
                for attempt in range(2):
                    m,o,e,d,s=start(cfg);h=[];advance(m,o,e,d,s,0,4,h);runs.append(endpoint(m,o,e,d,cfg,h));del m,o,e,d,s,h;gc.collect();torch.mps.empty_cache()
                require(runs[0]==runs[1],'Fresh repeat state-byte identity failed')
                m,o,e,d,s=start(cfg);h=[];advance(m,o,e,d,s,0,2,h)
                payload=raw_payload(m,o,e,cfg,2,h,{n:g.get_state() for n,g in d.generators.items()})
                original_payload_id=digest(payload);buffer=io.BytesIO();torch.save(cpu(payload),buffer);serialized_bytes=buffer.tell()
                del payload,m,o,e,d,s,h;gc.collect();torch.mps.empty_cache();buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=True);buffer.close();del buffer
                require(digest(saved)==original_payload_id,'Serialization changed state bytes')
                m,o,e,d,s=start(cfg,seed=False)
                before=digest(dict(model=m.state_dict(),optimizer=o.state_dict(),ema=e.state_dict()))
                bad=dict(saved,architecture=dict(saved['architecture'],input_dropout=.1 if probability==0. else 0.));rejected(lambda:validate_resume(m,o,e,bad,cfg))
                bad=dict(saved,kind='h200_embedding_input_dropout_v1');rejected(lambda:validate_resume(m,o,e,bad,cfg))
                bad=dict(saved,mps_rng=None);rejected(lambda:validate_resume(m,o,e,bad,cfg))
                bad=dict(saved,clipping_history=saved['clipping_history'][:-1]);rejected(lambda:validate_resume(m,o,e,bad,cfg))
                bad=dict(saved,optimizer_state=dict(saved['optimizer_state'],body_allocation_signature={}));rejected(lambda:validate_resume(m,o,e,bad,cfg))
                require(digest(dict(model=m.state_dict(),optimizer=o.state_dict(),ema=e.state_dict()))==before,'Rejected metadata forgery mutated destination')
                step,h,sampler=resume(m,o,e,saved,cfg);require(step==2,'Split differs')
                for n,g in d.generators.items():g.set_state(sampler[n].cpu())
                del saved,bad,sampler;gc.collect();advance(m,o,e,d,s,2,4,h);resumed=endpoint(m,o,e,d,cfg,h)
                require(resumed==runs[0],'Serialized resume state-byte/scalar/RNG identity failed')
                reports.append(dict(probability=probability,fresh_repeat_exact=True,serialized_resume_exact=True,serialized_bytes=serialized_bytes,endpoint=resumed,
                    five_resume_forgeries_rejected_before_mutation=True,raw_h200_ema_history_sampler_cpu_mps_rng=True))
                del m,o,e,d,s,h;gc.collect();torch.mps.empty_cache()
            require(sources()==pins,'Transport/runtime sources changed')
        result=dict(status='passed_exact_stochastic_repeat_and_serialized_resume',source=reference(__file__),plan=reference(E/'resume001-plan.json'),sources=pins,reports=reports,
            training_quality_result=False,new_scientific_seed=False,weight_files_written=0,guard=reference(E/'resume001-guard.json'))
        publish(E/'resume001.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('source','plan','sources','guard')}))
    except Exception as exc:publish(E/'STOP-resume001',dict(error=repr(exc),closed=True));raise
