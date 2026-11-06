"""Complete real-data three-arm repeat/serialized2+2 transport proof, no quality."""
import copy
import fcntl
import gc
import hashlib
import io
import json
from pathlib import Path
import traceback
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.tokenizer_moe_model_v1.model import GPT
from research.tokenizer_moe_model_v1.optimizer import build_optimizer
from research.tokenizer_moe_resume_v1.data import config,load,restore_samplers
from research.tokenizer_moe_resume_v1.contract import sources,runtime
from research.tokenizer_moe_resume_v1.checkpoint import raw_payload,h200_payload,validate_resume,resume,load_export
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard

E=Path(__file__).resolve().parent


def pack(value):
    if isinstance(value,torch.Tensor):
        t=value.detach().cpu().contiguous()
        return dict(dtype=str(t.dtype),shape=list(t.shape),sha256=hashlib.sha256(t.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest())
    if isinstance(value,dict):return [[type(k).__name__,repr(k),pack(value[k])] for k in sorted(value,key=lambda k:(type(k).__name__,repr(k)))]
    if isinstance(value,(list,tuple)):return [pack(v) for v in value]
    require(value is None or type(value) in (int,float,str,bool),'Supported state scalar required');return value


def digest(value):return hashlib.sha256(json.dumps(pack(value),sort_keys=True,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu()
    if isinstance(value,dict):return {k:cpu(v) for k,v in value.items()}
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return value


def publish(name,value):
    raw=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<1024**2,'OneMiB receipt cap')
    with (E/name).open('xb') as f:f.write(raw)


def start(cfg,seed=True):
    if seed:torch.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed)
    d=load(cfg,'mps');m=GPT(plain(cfg.moe_model)).to('mps');o=build_optimizer(m,cfg.training)
    return m,o,TailWeightAverages(m,cfg.training),d,LearningRateSchedule(o,cfg.training)


def fixture_config(arm):
    cfg=config(arm)
    cfg.training.eval_every=1;cfg.training.eval_iters=1
    return cfg


def advance(m,o,e,d,s,first,end,history):
    for i in range(first,end):
        m.train();x,y=d.get_batch('train');o.zero_grad(set_to_none=True)
        before=[b.mlp.assignment_counts.clone() for b in m.transformer.h] if m.signature['arm']=='moe' else None
        _,loss,parts=m(x,y,include_training_objective=True,return_loss_breakdown=True)
        require(bool(torch.isfinite(loss)),'Finite training objective required');loss.backward()
        active=[n for n,p in m.named_parameters() if p.grad is not None]
        routes=[(b.mlp.assignment_counts-old).cpu().tolist() for b,old in zip(m.transformer.h,before)] if before is not None else None
        history.append(dict(step=i+1,**clip_gradients(m.parameters(),1.),active_parameter_names=active,route_counts=routes,
                            ce_loss=float(parts['ce_loss'].detach()),aux_loss=float(parts['aux_loss'].detach()) if parts['aux_loss'] is not None else None,total_loss=float(loss.detach())))
        s.apply(i);o.step();e.update(i+1)
        m.eval()
        with torch.no_grad():
            vx,vy=d.get_batch('val');_,ce=m(vx,vy,include_aux_loss=False)
            require(bool(torch.isfinite(ce)),'Finite sampled validation CE required')
    torch.mps.synchronize()


def endpoint(m,o,e,d,cfg,history):
    raw=raw_payload(m,o,e,cfg,4,history,{n:g.get_state() for n,g in d.generators.items()})
    validate_resume(m,o,e,raw,cfg);raw_id=digest(raw)
    hp=h200_payload(m,e,cfg,dict(kind='in_memory_fixture_no_weight_file',raw_state_digest=raw_id))
    h_id=digest(hp);load_export(m,hp,cfg);m.eval()
    with torch.no_grad():
        logits=m(torch.arange(16,device='mps').reshape(1,16),include_aux_loss=False)[0]
        logit_id=digest(logits)
    return dict(raw_model_optimizer_ema_history_sampler_cpu_mps_rng=raw_id,h200=h_id,h200_logits=logit_id,
                learned_parameter_tensors=len(e.parameters),snapshots=e.snapshots,clipping_updates=len(history),
                per_parameter_updates=o.gradient_steps,route_counts=[b.mlp.assignment_counts.cpu().tolist() for b in m.transformer.h] if m.signature['arm']=='moe' else None)


def destination(m,o,e,d):
    return digest(dict(model=m.state_dict(),optimizer=o.state_dict(),ema=e.state_dict(),cpu_rng=torch.get_rng_state(),mps_rng=torch.mps.get_rng_state(),samplers={n:g.get_state() for n,g in d.generators.items()}))


def reject(call):
    try:call()
    except (TypeError,ValueError,RuntimeError):return
    raise AssertionError('Invalid resume accepted')


def forgeries(saved):
    cfg=copy.deepcopy(saved['cfg']);cfg['training']['learning_rate']=.0025
    wrong_rope=dict(saved['model_state']);key=next(n for n in wrong_rope if n.endswith('freq_cos'))
    wrong_rope[key]=wrong_rope[key].clone();wrong_rope[key][0,0]+=1.
    exp=copy.deepcopy(saved['optimizer_state']['gradient_steps']);name=next(iter(exp));exp[name]+=1
    opt=copy.copy(saved['optimizer_state']);opt['gradient_steps']=exp
    lr_opt=copy.copy(saved['optimizer_state']);lr_opt['optimizers']=list(saved['optimizer_state']['optimizers'])
    child=copy.copy(lr_opt['optimizers'][0]);groups=list(child['param_groups']);groups[0]=dict(groups[0],lr=0.);child['param_groups']=groups;lr_opt['optimizers'][0]=child
    wrong_step=copy.copy(saved['optimizer_state']);wrong_step['optimizers']=list(saved['optimizer_state']['optimizers'])
    child=copy.copy(wrong_step['optimizers'][1]);states=dict(child['state']);index=next(iter(states));states[index]=dict(states[index],step=states[index]['step']+1)
    child['state']=states;wrong_step['optimizers'][1]=child
    ema=copy.copy(saved['ema']);ema['parameter_names']=list(saved['ema']['parameter_names'])+['integer_route_buffer']
    history=copy.deepcopy(saved['clipping_history']);history[0]['active_parameter_names']=history[0]['active_parameter_names'][:-1]
    result=[dict(kind='h200_tokenizer_moe_v1'),dict(architecture=dict(saved['architecture'],policy='foreign')),
            dict(cpu_rng=saved['cpu_rng'][:-1]),dict(mps_rng=saved['mps_rng'][:-1]),dict(mps_rng=saved['mps_rng'].to(torch.int64)),
            dict(sampler_state={'train':saved['sampler_state']['train']}),dict(sampler_state=dict(saved['sampler_state'],val=saved['sampler_state']['val'][:-1])),
            dict(cfg=cfg),dict(training_identity=dict(saved['training_identity'],data_identity={})),dict(model_state=wrong_rope),
            dict(optimizer_state=opt),dict(optimizer_state=lr_opt),dict(optimizer_state=wrong_step),dict(ema=ema),dict(clipping_history=history),dict(clipping_history=saved['clipping_history'][:-1])]
    adam=saved['optimizer_state']['optimizers'][1]
    small_index=next(g['params'][0] for g in adam['param_groups'] if g['parameter_names'][0].endswith('.bias'))
    for mode in ['dtype','shape','nonfinite']:
        altered=copy.copy(saved['optimizer_state']);altered['optimizers']=list(saved['optimizer_state']['optimizers'])
        child=copy.copy(adam);states=dict(adam['state']);slot=dict(states[small_index]);value=slot['exp_avg']
        if mode=='dtype':value=value.double()
        elif mode=='shape':value=value[:-1]
        else:value=value.clone();value[0]=float('nan')
        slot['exp_avg']=value;states[small_index]=slot;child['state']=states;altered['optimizers'][1]=child
        result.append(dict(optimizer_state=altered))
    ema=copy.copy(saved['ema']);panel=dict(ema['averages']['200']);panel.pop(next(iter(panel)))
    ema['averages']={'200':panel};result.append(dict(ema=ema))
    if saved['architecture']['arm']=='moe':
        bad=dict(saved['model_state']);key='transformer.h.0.mlp.assignment_counts';bad[key]=bad[key].clone();bad[key][0]+=1
        result.append(dict(model_state=bad))
        history=copy.deepcopy(saved['clipping_history']);history[0]['route_counts'][0][0]+=1
        result.append(dict(clipping_history=history))
    return result


def main():
    require(not any((E/n).exists() for n in ['resume001-guard.json','resume001-result.json','STOP-resume001']),'Closed proof cannot retry')
    torch.set_num_threads(4);runtime();pins=sources();plan=json.loads((E/'resume001-plan.json').read_text())
    require(len(plan['sources'])==1019 and all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in plan['sources'].items()),'Exact preflight source pins required')
    require(__import__('shutil').disk_usage(ROOT).free>=plan['cleanup_target_bytes'],'Complete prerequisite requirement plus2GiB required')
    reports=[];before_cpu=torch.get_rng_state().clone()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with StrictMpsGuard(E/'resume001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
            before_mps=torch.mps.get_rng_state().clone()
            try:
                for arm in ['dense','wide','moe']:
                    cfg=fixture_config(arm);repeat=[]
                    for _ in range(2):
                        m,o,e,d,s=start(cfg);history=[];advance(m,o,e,d,s,0,4,history)
                        repeat.append(endpoint(m,o,e,d,cfg,history))
                        del m,o,e,d,s,history;gc.collect();torch.mps.empty_cache()
                    require(repeat[0]==repeat[1],'Fresh exact repeat failed')
                    m,o,e,d,s=start(cfg);history=[];advance(m,o,e,d,s,0,2,history)
                    raw=raw_payload(m,o,e,cfg,2,history,{n:g.get_state() for n,g in d.generators.items()})
                    raw_id=digest(raw);tree=cpu(raw)
                    # Release ALL MPS training state before encoding the CPU tree.
                    del raw,m,o,e,d,s,history;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
                    buffer=io.BytesIO();torch.save(tree,buffer);serialized=buffer.tell()
                    require(serialized<=plan['serialized_buffer_cap_bytes'],'Declared complete serialization cap exceeded')
                    del tree;gc.collect();buffer.seek(0);saved=torch.load(buffer,map_location='cpu',weights_only=True);buffer.close();del buffer
                    require(digest(saved)==raw_id,'Serialized dtype/shape/state bytes changed')
                    m,o,e,d,s=start(cfg,seed=False);before=destination(m,o,e,d)
                    bad_cases=forgeries(saved)
                    for fields in bad_cases:
                        bad=dict(saved,**fields);reject(lambda:validate_resume(m,o,e,bad,cfg))
                        require(destination(m,o,e,d)==before,'Rejected payload mutated destination or global RNG')
                    count=len(bad_cases);del bad,bad_cases
                    step,history=resume(m,o,e,saved,cfg,d);require(step==2,'Split step differs')
                    del saved;gc.collect();advance(m,o,e,d,s,2,4,history)
                    resumed=endpoint(m,o,e,d,cfg,history);require(resumed==repeat[0],'Serialized2+2 exact endpoint differs')
                    report=dict(arm=arm,parameters=m.signature['parameters'],per_trajectory_targets=4096,
                                fresh_repeat_exact=True,serialized_resume_exact=True,serialized_bytes=serialized,forgeries_rejected_before_mutation=count,
                                endpoint=resumed,corpus_quality_result=False)
                    publish(f'resume001-arm-{arm}.json',report);reports.append(report)
                    del m,o,e,d,s,history;gc.collect();torch.mps.empty_cache()
                # An additional declared real-data one-update tie fixture exercises
                # legitimate missing sparse slots at a NONZERO global step.
                cfg=fixture_config('moe');m,o,e,d,s=start(cfg)
                with torch.no_grad():
                    for b in m.transformer.h:b.mlp.router.weight.zero_()
                history=[];advance(m,o,e,d,s,0,1,history)
                raw=raw_payload(m,o,e,cfg,1,history,{n:g.get_state() for n,g in d.generators.items()})
                validate_resume(m,o,e,raw,cfg)
                zero_names=[n for n,v in o.gradient_steps.items() if v==0]
                require(len(zero_names)==24 and all(any(f'.experts.{i}.' in n for i in [2,3]) for n in zero_names),'Declared dormant expert coverage differs')
                dormant=dict(global_step=1,zero_gradient_update_parameters=len(zero_names),absent_optimizer_slots_accepted=True,real_targets=1024,quality_result=False)
                del raw,m,o,e,d,s,history;gc.collect();torch.mps.empty_cache()
            finally:torch.mps.set_rng_state(before_mps)
            require(torch.equal(torch.mps.get_rng_state(),before_mps),'Proof changed global MPS RNG')
    require(torch.equal(torch.get_rng_state(),before_cpu) and sources()==pins,'Global CPU RNG or immutable sources changed')
    for n,h in plan['sources'].items():require(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,'Frozen proof source changed')
    result=dict(status='all_three_real_fresh_repeat_and_serialized_resume_passed',reports=reports,dormant_expert_fixture=dormant,
                sources=pins,proof_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),total_source_bindings=1019,
                fixture_seed=600460,evaluation_fixture_seed=600461,new_scientific_seed=False,real_training_targets=37888,
                fresh_snapshots_per_trajectory=3,weight_files_written=0,serialized_buffers='in_memory_only_released_per_arm',
                corpus_quality_result=False,holdout_encoded_or_scored=False,global_cpu_mps_rng_preserved=True)
    require(sum(p.stat().st_size for p in E.rglob('*') if p.is_file())+1024**2<=plan['metadata_cap_bytes'],'Complete metadata cap exceeded')
    publish('resume001-result.json',result);publish('STOP-resume001',dict(status='complete_closed_prerequisite',never_rerun=True))
    print(json.dumps(dict(status=result['status'],arms=[r['arm'] for r in reports],serialized_bytes=[r['serialized_bytes'] for r in reports],dormant=dormant)))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        if not (E/'STOP-resume001').exists():publish('STOP-resume001',dict(status='CLOSED_FAILED_REAL_RESUME_PROOF',error=repr(exc),traceback=traceback.format_exc(),never_rerun=True,no_quality_result=True))
        raise
