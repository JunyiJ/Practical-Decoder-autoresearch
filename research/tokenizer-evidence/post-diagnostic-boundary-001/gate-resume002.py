"""New real native-data own-gate fresh/repeat/serialized-resume proof; no quality test."""
import copy,gc,hashlib,io,json,os,shutil,traceback
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_model_v1.contract import ARMS
from research.tokenizer_mla_gate_model_v1.model import GPT
from research.tokenizer_mla_gate_model_v1.optimizer import build_optimizer
from research.tokenizer_mla_gate_resume_v1.contract import sources,runtime
from research.tokenizer_mla_gate_resume_v1.data import config,load,DATA,identity
from research.tokenizer_mla_gate_resume_v1.checkpoint import raw_payload,h200_payload,validate_resume,resume,load_export
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.tokenizer_moe_worker_v1.execution import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock

E=Path(__file__).resolve().parent;R=E/'gate-resume002';META=2*1024**2
INITIALIZER=600523;TRAIN=600524;EVALUATION=600525
FLOOR=8*1024**3+META+1267904512+1458176;TARGET=FLOOR+2*1024**3
BUFFER_CAPS={'none':480*1024**2,'static_half':480*1024**2,'sigmoid_elementwise_zero':512*1024**2}

def publish(p,v):
    raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(f.stat().st_size for f in R.glob('*') if f.is_file())+len(raw)<=META,'Complete2MiB proof metadata cap required')
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def pack(v):
    if isinstance(v,torch.Tensor):
        t=v.detach().cpu().contiguous();return dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(v,dict):return [[type(k).__name__,repr(k),pack(v[k])] for k in sorted(v,key=lambda k:(type(k).__name__,repr(k)))]
    if isinstance(v,(list,tuple)):return [pack(x) for x in v]
    require(v is None or type(v) in (int,float,str,bool),'Supported state scalar required');return v

def digest(v):return hashlib.sha256(json.dumps(pack(v),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    if isinstance(v,tuple):return tuple(cpu(x) for x in v)
    return v

def start(cfg):
    d=load(cfg,'mps')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(INITIALIZER);m=GPT(plain(cfg.gate_model))
    m=m.to('mps');o=build_optimizer(m,cfg.training)
    return m,o,TailWeightAverages(m,cfg.training),d,LearningRateSchedule(o,cfg.training)

def advance(m,o,e,d,s,first,end,history):
    stream=[]
    for i in range(first,end):
        m.train();x,y=d.get_batch('train');o.zero_grad(set_to_none=True)
        _,loss=m(x,y,include_aux_loss=False);require(bool(torch.isfinite(loss)),'Finite real native training CE required');loss.backward()
        active=[n for n,p in m.named_parameters() if p.grad is not None]
        history.append(dict(step=i+1,**clip_gradients(m.parameters(),1.),active_parameter_names=active,ce_loss=float(loss.detach())))
        s.apply(i);o.step();e.update(i+1);m.eval()
        with torch.no_grad():
            vx,vy=d.get_batch('val');ce=m(vx,vy,include_aux_loss=False)[1];require(bool(torch.isfinite(ce)),'Finite sampled prerequisite validation CE required')
        stream.append(dict(train_input=digest(x),train_targets=digest(y),eval_input=digest(vx),eval_targets=digest(vy)))
    torch.mps.synchronize();return stream

def destination(m,o,e,d):return digest(dict(model=m.state_dict(),optimizer=o.state_dict(),ema=e.state_dict(),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),samplers={n:g.get_state() for n,g in d.generators.items()}))

def endpoint(m,o,e,d,cfg,history,stream):
    raw=raw_payload(m,o,e,cfg,4,history,{n:g.get_state() for n,g in d.generators.items()});validate_resume(m,o,e,raw,cfg);raw_id=digest(raw)
    hp=h200_payload(m,e,cfg,dict(fixture_kind='in_memory_no_weight_file',raw_state_sha256=raw_id));h_id=digest(hp)
    require(set(hp['model_state'])==set(m.state_dict()) and set(e.averages['200'])=={n for n,_ in m.named_parameters()},'EMA learned parameter/buffer distinction differs')
    for n,t in m.named_buffers():
        if n in hp['model_state']:require(torch.equal(hp['model_state'][n],t),'H200 changed persisted nonlearned buffer')
        else:require(n.endswith('.attn.causal_mask') and t.dtype==torch.bool and t.shape==(128,128) and torch.equal(t,torch.tril(torch.ones(128,128,dtype=torch.bool,device=t.device))),'Nonpersistent causal buffer must reconstruct exact native mask')
    # Separate H200 schema is inference-only; no derived weights may resume.
    reject(lambda:validate_resume(m,o,e,hp,cfg))
    load_export(m,hp,cfg);m.eval()
    with torch.no_grad():
        z=torch.arange(16,device='mps',dtype=torch.int64).reshape(1,16);logit_id=digest(m(z)[0])
    return dict(raw_model_optimizer_ema_history_sampler_cpu_mps_rng=raw_id,h200=h_id,h200_logits=logit_id,learned_parameter_tensors=len(e.parameters),snapshots=e.snapshots,clipping_updates=len(history),per_parameter_updates=copy.deepcopy(o.gradient_steps),streams=stream)

def reject(call):
    try:call()
    except (TypeError,ValueError,RuntimeError):return
    raise AssertionError('Invalid transport payload accepted')

def forgeries(saved):
    cfg=copy.deepcopy(saved['cfg']);cfg['training']['learning_rate']=.0025
    wrong_rope=dict(saved['model_state']);key=next(n for n in wrong_rope if n.endswith('freq_cos'));wrong_rope[key]=wrong_rope[key].clone();wrong_rope[key][0,0]+=1.
    counts=dict(saved['optimizer_state']['gradient_steps']);counts[next(iter(counts))]+=1
    opt=dict(saved['optimizer_state'],gradient_steps=counts)
    ema=dict(saved['ema'],parameter_names=saved['ema']['parameter_names']+['foreign_buffer'])
    panel=dict(saved['ema']['averages']['200']);panel.pop(next(iter(panel)))
    hist=copy.deepcopy(saved['clipping_history']);hist[0]['active_parameter_names']=hist[0]['active_parameter_names'][:-1]
    clip=copy.deepcopy(saved['clipping_history']);clip[0]['coefficient']=0.
    result=[dict(kind='raw_tokenizer_moe_v1'),dict(kind='h200_tokenizer_mla_gate_v1'),dict(architecture=dict(saved['architecture'],arm='foreign')),
        dict(cpu_rng=saved['cpu_rng'][:-1]),dict(mps_rng=saved['mps_rng'][:-1]),dict(mps_rng=saved['mps_rng'].to(torch.int64)),
        dict(sampler_state={'train':saved['sampler_state']['train']}),dict(sampler_state=dict(saved['sampler_state'],val=saved['sampler_state']['val'][:-1])),
        dict(cfg=cfg),dict(training_identity=dict(saved['training_identity'],sources_sha256='foreign')),dict(model_state=wrong_rope),dict(optimizer_state=opt),dict(ema=ema),
        dict(ema=dict(saved['ema'],averages={'200':panel})),dict(ema=dict(saved['ema'],snapshots=0)),dict(clipping_history=hist),dict(clipping_history=clip),dict(clipping_history=saved['clipping_history'][:-1]),dict(step=True)]
    for mode in ('lr','signature','adam_step','dtype','shape','nonfinite','missing_slot'):
        altered=dict(saved['optimizer_state']);children=list(altered['optimizers']);altered['optimizers']=children
        if mode=='signature':altered['tokenizer_mla_gate_signature']={}
        elif mode=='lr':
            child=dict(children[0]);groups=list(child['param_groups']);groups[0]=dict(groups[0],lr=0.);child['param_groups']=groups;children[0]=child
        else:
            child=dict(children[1]);states=dict(child['state']);child['state']=states;children[1]=child
            index=next(g['params'][0] for g in child['param_groups'] if g['parameter_names'][0].endswith('.bias'));slot=dict(states[index]);states[index]=slot
            if mode=='adam_step':slot['step']=slot['step']+1
            elif mode=='missing_slot':states.pop(index)
            else:
                value=slot['exp_avg']
                if mode=='dtype':value=value.double()
                elif mode=='shape':value=value[:-1]
                else:value=value.clone();value[0]=float('nan')
                slot['exp_avg']=value
        result.append(dict(optimizer_state=altered))
    return result

def main():
    with session_lock(GLOBAL_LOCK.parent):
        idle();torch.set_num_threads(4);env=runtime();require(not R.exists(),'Fresh own resume proof only;no attempted/closed retry')
        roles=json.loads((E/'fixture-role-census003.json').read_text());require(roles['no_prior_matches'] and roles['roles']==dict(initializer=INITIALIZER,private_train=TRAIN,private_evaluation=EVALUATION) and not roles['scientific_seeds'],'Fresh audited unused fixture roles required')
        pins=sources();pins[str((E/'gate-resume001.py').relative_to(ROOT))]=reference(E/'gate-resume001.py')['sha256'];pins[str(Path(__file__).resolve().relative_to(ROOT))]=reference(__file__)['sha256'];require(len(pins)==1068,'Preserve failed proof plus new additive corrected harness')
        failure=json.loads((E/'gate-resume-failure001.json').read_text())
        for k in ('source','plan','guard','terminal'):require(reference(failure[k]['path'])==failure[k],'Original failed proof evidence changed')
        require(failure['no_retry'] and failure['exception']=="KeyError('transformer.h.0.attn.causal_mask')",'Exact closed harness failure required')
        design=json.loads((E/'quality-design001.json').read_text());free=shutil.disk_usage(ROOT).free;require(free>=max(TARGET,design['complete_retained_budget']['target']),'Full prerequisite and future retained budget required')
        forecast=dict(rss=3186769920+256*1024**2,mps=2719825920+512*1024**2,seconds=360);require(forecast['rss']<4*1024**3 and forecast['mps']<6*1024**3 and forecast['seconds']<600,'Complete transport forecast must fit original guard')
        cfgs={a:config(a,4,TRAIN,EVALUATION) for a in ARMS};data=identity(next(iter(cfgs.values())))
        R.mkdir();publish(R/'plan.json',dict(kind='own_mla_gate_real_data_repeat_and_serialized_resume_v1',sources=pins,roles=reference(E/'fixture-role-census003.json'),prerequisite_review=reference(E/'review001.json'),original_failed_proof=reference(E/'gate-resume-failure001.json'),correction='Only persistent versus reconstructible nonpersistent buffer assertion; unchanged runtime, fresh roles',design=reference(E/'quality-design001.json'),environment=env,configs={a:plain(c) for a,c in cfgs.items()},data=data,data_source=reference(DATA),corpus_sha256=data['corpus_sha256'],initializer=INITIALIZER,private_train=TRAIN,private_evaluation=EVALUATION,buffer_caps=BUFFER_CAPS,metadata_cap=META,forecast=forecast,full_floor=FLOOR,target=TARGET,complete_prospective_target=design['complete_retained_budget']['target'],free_bytes=free,required_cases='all3arms,fresh4updates twice,serialized2+2,26rawforgeriesperarm+H200notresume+H200metadataforgeries,private samplers,fullstate/RNG invariant',weight_files_written=0,quality_result=False))
        try:
            reports=[];cpu_before=torch.get_rng_state().clone();mps_before=torch.mps.get_rng_state().clone()
            with StrictMpsGuard(R/'guard.json',ROOT):
                for arm,cfg in cfgs.items():
                    repeats=[]
                    for _ in range(2):
                        m,o,e,d,s=start(cfg);history=[];stream=advance(m,o,e,d,s,0,4,history);repeats.append(endpoint(m,o,e,d,cfg,history,stream))
                        del m,o,e,d,s,history,stream;gc.collect();torch.mps.empty_cache()
                    require(repeats[0]==repeats[1],'Exact fresh real-data repeat failed')
                    m,o,e,d,s=start(cfg);history=[];prefix=advance(m,o,e,d,s,0,2,history)
                    raw=raw_payload(m,o,e,cfg,2,history,{n:g.get_state() for n,g in d.generators.items()});validate_resume(m,o,e,raw,cfg);raw_id=digest(raw);tree=cpu(raw)
                    del raw,m,o,e,d,s,history;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
                    buffer=io.BytesIO();torch.save(tree,buffer);serialized=buffer.tell();require(serialized<=BUFFER_CAPS[arm],'Original serialized buffer cap exceeded')
                    del tree;gc.collect();buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=True);buffer.close();del buffer
                    require(digest(saved)==raw_id,'Serialized fullstate bytes changed')
                    m,o,e,d,s=start(cfg);before=destination(m,o,e,d);bad_cases=forgeries(saved)
                    for fields in bad_cases:
                        reject(lambda:resume(m,o,e,dict(saved,**fields),cfg,d));require(destination(m,o,e,d)==before,'Rejected resume mutated destination/RNG/samplers')
                    count=len(bad_cases);del fields,bad_cases
                    step,history=resume(m,o,e,saved,cfg,d);require(step==2,'Split step differs');del saved;gc.collect()
                    suffix=advance(m,o,e,d,s,2,4,history);resumed=endpoint(m,o,e,d,cfg,history,prefix+suffix);require(resumed==repeats[0],'Serialized2+2 endpoint,history,samplers or RNG differs')
                    hp=h200_payload(m,e,cfg,dict(fixture_kind='in_memory_no_weight_file',raw_state_sha256=resumed['raw_model_optimizer_ema_history_sampler_cpu_mps_rng']));before=destination(m,o,e,d)
                    for fields in [dict(kind='h200_tokenizer_moe_v1'),dict(parent={}),dict(averaged_weights=dict(hp['averaged_weights'],half_life=100)),dict(model_state={})]:
                        reject(lambda:load_export(m,dict(hp,**fields),cfg));require(destination(m,o,e,d)==before,'Rejected H200 export mutated destination')
                    reports.append(dict(arm=arm,parameters=m.signature['parameters'],learned_tensors=len(e.parameters),fresh_repeat_exact=True,serialized_2_plus_2_exact=True,serialized_bytes=serialized,raw_transactional_rejections=count,h200_transactional_rejections=4,endpoint=resumed))
                    publish(R/('arm-'+arm+'.json'),reports[-1]);print(json.dumps(dict(status='arm_real_repeat_resume_complete',arm=arm,serialized_bytes=serialized)),flush=True)
                    del m,o,e,d,s,history,repeats,resumed,prefix,suffix,hp;gc.collect();torch.mps.empty_cache()
                require(all(r['endpoint']['streams']==reports[0]['endpoint']['streams'] for r in reports),'Same private batch streams across arms required')
                require(torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state()),'Global CPU/MPS RNG changed')
                for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Frozen source changed during transport proof')
            result=dict(status='all_three_native_gate_real_repeat_and_serialized_resume_passed',plan=reference(R/'plan.json'),source_bindings=1068,reports=reports,global_cpu_mps_rng_preserved=True,same_inputs_across_arms=True,roles_used=roles['roles'],training_target_evaluations=3*3*4*1024,sampled_validation_target_evaluations=3*3*4*1024,full_validation_scored=False,holdout_encoded_or_scored=False,weight_files_written=0,serialization='bounded_in_memory_only_released_per_arm',quality_result=False,adopted=False)
            publish(R/'result.json',result);publish(R/'STOP',dict(status='complete_own_gate_real_data_resume_proof',result=reference(R/'result.json'),never_rerun=True));print(json.dumps(dict(status=result['status'],source_bindings=1068)),flush=True)
        except BaseException as exc:
            if not (R/'STOP').exists():publish(R/'STOP',dict(status='closed_failed',error=repr(exc),traceback=traceback.format_exc(),no_retry=True,quality_result=False))
            raise

if __name__=='__main__':main()
