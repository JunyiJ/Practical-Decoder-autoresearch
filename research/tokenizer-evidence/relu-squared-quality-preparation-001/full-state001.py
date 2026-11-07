"""One NEW arm's genuine1200 generated updates/fullstate/EMA41 proof, not corpus quality."""
import argparse
import copy
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.relu_squared_model_v1.model import GPT
from research.relu_squared_model_v1.optimizer import build_optimizer
from research.relu_squared_resume_v1.data import View,model_spec
from research.relu_squared_resume_v1 import checkpoint as C
from research.relu_squared_resume_v1 import scoring as S
from research.relu_squared_quality_v1 import contract as Q
from research.relu_squared_quality_v1.training import validate_config,Counter
from research.relu_squared_quality_v1.validation import cache_check
from research.block_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

B=Path(__file__).resolve().parent
R=Path(__file__).with_suffix('')
A=ROOT/'research_runs/tokenizer-relu-squared-full-state-v1-001'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(path,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.rglob('*') if p.is_file())+len(raw)<=16*1024**2,'Complete full-state metadata cap')
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def tree_hash(value):
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            t=v.detach().cpu().contiguous();h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        elif isinstance(v,dict):
            h.update(b'dict')
            for k in sorted(v,key=lambda x:(type(x).__name__,str(x))):visit(k);visit(v[k])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for x in v:visit(x)
        else:h.update(json.dumps([type(v).__name__,v],sort_keys=True,allow_nan=False).encode())
    visit(value);return h.hexdigest()


def replace(value,keys,new):
    out=value.copy() if isinstance(value,dict) else list(value)
    out[keys[0]]=new if len(keys)==1 else replace(value[keys[0]],keys[1:],new)
    return out


def view(cfg):
    tensors={n:(torch.arange(length,dtype=torch.int64)*factor+137)%50257 for n,length,factor in [('train',16384,97),('val',8192,101)]}
    return View(train_data=tensors['train'],val_data=tensors['val'],splits=tensors,
        generators={n:torch.Generator(device='cpu').manual_seed(seed) for n,seed in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]},
        batch_size=8,block_size=128,batch_layout='paired_blocks512',vocab_size=50257,device='mps',
        data_identity=dict(kind='NEW_generated_1200_FFN_state_fixture_NOT_corpus',ids_sha256=tree_hash(tensors)),tokenizer_contract=dict(kind='literal_native_ID_fixture_NOT_physical_codec'))


def snapshot(model,opt,ema,data):
    return tree_hash(dict(model=model.state_dict(),optimizer=opt.state_dict(),ema=ema.state_dict(),samplers={n:g.get_state() for n,g in data.generators.items()},
        cpu=torch.get_rng_state(),mps=torch.mps.get_rng_state(),weight_epoch=model.weight_epoch,grads={n:p.grad for n,p in model.named_parameters()}))


def save(path,payload):
    counter=Counter(C.FILE_CAPS[payload['kind']]);torch.save(payload,counter)
    ref=C.publish_payload(path,payload);require(ref['bytes']==counter.size,'Actual counted/published archive bytes differ');return ref


def serving_core(model):
    rows=[]
    with torch.no_grad():
        for batch in (1,8):
            ids=((torch.arange(batch*128).reshape(batch,128)*19+61)%50257).to('mps');cache=model.caches(batch)
            size=sum(t.numel()*t.element_size() for item in cache.layers for t in item.buffers())//batch;require(size==196608,'Own cache bytes differ')
            ft=[];dt=[];maximum=0.
            for repeat in range(7):
                cache.reset();torch.mps.synchronize();start=time.perf_counter();full=model(ids).log_probs();torch.mps.synchronize();a=time.perf_counter()-start
                prefix=model(ids[:,:83],cache=cache).log_probs();del prefix;torch.mps.synchronize();start=time.perf_counter()
                for index in range(83,128):
                    last=model(ids[:,index:index+1],cache=cache,start_pos=index).log_probs()
                    if repeat==0:
                        torch.testing.assert_close(last,full[:,index:index+1],atol=3e-5,rtol=3e-5)
                        maximum=max(maximum,float((last-full[:,index:index+1]).abs().max().cpu()))
                torch.mps.synchronize();d=time.perf_counter()-start
                if repeat>=2:ft.append(a);dt.append(d)
                del full,last
            rows.append(dict(batch=batch,cache_bytes_per_item=size,full128_seconds=ft,decode45_after_prefill83_seconds=dt,
                median_full128_seconds=statistics.median(ft),median_decode45_seconds=statistics.median(dt),max_logprob_error=maximum))
            del ids,cache
    return rows


def arm_proof(arm,plan):
    cohort=plan['roles']['screen'];cfg=Q.spec(Q.ARMS.index(arm),arm,cohort,.00125)['config']
    from omegaconf import OmegaConf
    cfg=OmegaConf.create(cfg);cfg.training.checkpoint_dir=str(A/arm);cfg.training.metrics_file=str(R/(arm+'-metrics.jsonl'));cfg.training.summary_file=str(R/(arm+'-summary.json'))
    require(validate_config(cfg)==1200,'Genuine own full-horizon config required')
    torch.manual_seed(cfg.rng_identity.global_cpu);torch.mps.manual_seed(cfg.rng_identity.global_mps)
    with CpuPrerequisiteGuard(R/(arm+'-cpu-construction.json'),ROOT,timeout_seconds=120,poll_seconds=.25):
        model=GPT(model_spec(cfg),initializer_seed=cfg.initializer_seed);data=view(cfg)
    model=model.to('mps');opt=build_optimizer(model,cfg.training);ema=TailWeightAverages(model,cfg.training);schedule=LearningRateSchedule(opt,cfg.training)
    initial_cpu=torch.get_rng_state().clone();initial_mps=torch.mps.get_rng_state().clone();history=[];traces=[];ema_steps=[];targets=0;started=time.monotonic()
    for i in range(1200):
        model.train();x,y=data.get_batch('train');require(x.shape==y.shape==(8,128),'Actual full generated batch required');targets+=y.numel();opt.zero_grad();loss=model.loss(x,y);require(bool(torch.isfinite(loss)),'Finite generated NLL required');loss.backward()
        names=[n for n,p in model.named_parameters() if p.grad is not None];require(names==[n for n,_ in model.named_parameters()],'Every learned parameter exposed')
        history.append(dict(step=i+1,**clip_gradients(model.parameters(),1.),active_parameter_names=names,ce_loss=float(loss.detach())))
        ratio=schedule.apply(i);expected=1. if i<=800 else (.1 if i==1199 else 1-.9*(i-800)/399)
        require(ratio==expected,'Independent literal absolute LR differs');opt.step();before=ema.snapshots;ema.update(i+1)
        if ema.snapshots!=before:ema_steps.append(i+1)
        if i%300==0 or i==1199:
            model.eval()
            with torch.no_grad():
                vx,vy=data.get_batch('val');value=float(model.loss(vx,vy));require(math.isfinite(value),'Finite sampled generated NLL required')
            traces.append(dict(step=i+1,lr_multiplier=ratio,validation_ce=value,train_ce=history[-1]['ce_loss'],ema_snapshots=ema.snapshots))
            print(json.dumps(dict(arm=arm,step=i+1,ema_snapshots=ema.snapshots)),flush=True)
    opt.zero_grad();require(ema_steps==list(range(800,1201,10)) and ema.snapshots==41 and opt.completed_steps==1200,'Actual1200steps/41snapshot cadence required')
    require(targets==1228800,'Complete actual generated target count required')
    exposure=copy.deepcopy(opt.gradient_steps)
    require(all(v==1200 for v in opt.gradient_steps.values()) and len(opt.gradient_steps)==(88 if arm=='native_swiglu' else 84),'Actual all-parameter1200exposure required')
    require(torch.equal(initial_cpu,torch.get_rng_state()) and torch.equal(initial_mps,torch.mps.get_rng_state()),'Deterministic full worker global streams changed')
    for name,seed,count,high in [('train',cfg.training.seed,2400,16384-512),('val',cfg.training.eval_seed,40,8192-128)]:
        gen=torch.Generator(device='cpu').manual_seed(seed);torch.randint(high,(count,),generator=gen)
        require(torch.equal(gen.get_state(),data.generators[name].get_state()),'Actual generated private sampler counters differ')
    hashes=dict(model=tree_hash(model.state_dict()),optimizer=tree_hash(opt.state_dict()),ema=tree_hash(ema.state_dict()))
    A.joinpath(arm).mkdir();raw=C.raw_payload(model,opt,ema,cfg,1200,history,data);raw_ref=save(A/arm/'raw.pt',raw);del raw;gc.collect()
    exported=C.h200_payload(model,ema,cfg,raw_ref);h200_ref=save(A/arm/'h200.pt',exported);expected_h200=tree_hash(exported['model_state']);del exported
    training_seconds=time.monotonic()-started;del model,opt,ema,schedule;gc.collect();torch.mps.empty_cache()
    with CpuPrerequisiteGuard(R/(arm+'-cpu-archive-read.json'),ROOT,timeout_seconds=120,poll_seconds=.25):raw=C.read_payload(raw_ref);h200=C.read_payload(h200_ref)
    with CpuPrerequisiteGuard(R/(arm+'-cpu-destination.json'),ROOT,timeout_seconds=120,poll_seconds=.25):model=GPT(model_spec(cfg),initializer_seed=cfg.initializer_seed)
    model=model.to('mps');opt=build_optimizer(model,cfg.training);ema=TailWeightAverages(model,cfg.training);data=view(cfg)
    step,loaded_history=C.resume(model,opt,ema,raw,cfg,data)
    require(step==1200 and loaded_history==history and tree_hash(model.state_dict())==hashes['model'] and tree_hash(opt.state_dict())==hashes['optimizer'] and tree_hash(ema.state_dict())==hashes['ema'],'Actual full RAW roundtrip differs')
    before=snapshot(model,opt,ema,data);negative=[]
    state_cases=[('step1199',['step'],1199),('history1199',['clipping_history'],raw['clipping_history'][:-1]),('EMA40',['ema','snapshots'],40),
        ('final_lr_wrong',['optimizer_state','optimizers',0,'param_groups',0,'lr'],.00125)]
    for name,path,value in state_cases:
        bad=replace(raw,path,value)
        try:C.resume(model,opt,ema,bad,cfg,data)
        except ValueError:negative.append(name)
        else:raise AssertionError('Invalid full-state boundary accepted: '+name)
        require(snapshot(model,opt,ema,data)==before,'Rejected full-state mutated destination')
    try:C.resume(model,opt,ema,h200,cfg,data)
    except ValueError:negative.append('H200_not_optimizer_resume')
    else:raise AssertionError('H200 resumed optimizer')
    require(snapshot(model,opt,ema,data)==before,'Rejected H200resume mutated destination')
    del raw;gc.collect();C.load_export(model,h200,cfg);require(tree_hash(model.state_dict())==expected_h200,'H200 exact retained RAWparent EMA differs')
    for name,path,value in [('H200step1199',['step'],1199),('H200EMA40',['averaged_weights','snapshots'],40)]:
        before=snapshot(model,opt,ema,data)
        try:C.load_export(model,replace(h200,path,value),cfg)
        except ValueError:negative.append(name)
        else:raise AssertionError('Invalid full H200 boundary accepted')
        require(snapshot(model,opt,ema,data)==before,'Rejected H200load mutated destination')
    del h200,opt,ema;gc.collect();model.eval()
    caches=dict(mps=cache_check(model));serving=serving_core(model)
    with CpuPrerequisiteGuard(R/(arm+'-cpu-inference.json'),ROOT,timeout_seconds=120,poll_seconds=.25):
        with torch.random.fork_rng(devices=[]):cpu=GPT(model_spec(cfg),initializer_seed=cfg.initializer_seed).eval()
        payload=C.read_payload(h200_ref);C.load_export(cpu,payload,cfg);del payload;caches['cpu']=cache_check(cpu)
        ids=(torch.arange(282,dtype=torch.int64)*109+149)%50257
        cpu_losses=torch.cat(S.window_losses(cpu,ids))
    mps_losses=torch.cat(S.window_losses(model,ids));torch.testing.assert_close(cpu_losses,mps_losses,atol=3e-5,rtol=0.)
    gap=float((cpu_losses-mps_losses).abs().max());del cpu,cpu_losses,mps_losses
    x=ids[:25].unsqueeze(0).to('mps');y=ids[1:26].unsqueeze(0).to('mps');model.zero_grad(set_to_none=True)
    ce=model.loss(x,y);ce.backward();grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters()};model.zero_grad(set_to_none=True)
    lp=model(x).log_probs();gather=-lp.gather(-1,y.unsqueeze(-1)).squeeze(-1).mean();gather.backward()
    require(abs(float(ce.detach())-float(gather.detach()))<=3e-5,'Generated CE/fullgather differs')
    maximum=0.
    for name,p in model.named_parameters():
        torch.testing.assert_close(grads[name],p.grad.detach().cpu(),atol=4e-5,rtol=4e-4);maximum=max(maximum,float((grads[name]-p.grad.detach().cpu()).abs().max()))
    result=dict(status='accepted_genuine_generated_full_horizon_arm',arm=arm,config=plain(cfg),step=1200,actual_targets=targets,gradient_exposure=exposure,
        ema_snapshots=41,ema_steps=ema_steps,metrics=traces,data_identity=data.data_identity,raw=raw_ref,h200=h200_ref,hashes=hashes,
        training_and_publication_seconds=training_seconds,full_state_roundtrip_exact=True,transactional_boundary_rejections=negative,cache=caches,serving_core=serving,
        generated_target_count=281,cpu_mps_target_ce_max=gap,ce_gather_gradient_max=maximum,
        not_corpus_physical_scoring_or_quality=True,serving_scope='Three-arm per-model actual generated H200 protocol;not the full nine-fresh main consumer success',
        no_old_fixture_replay_or_CPU_update_MPS_update_equivalence_claim=True)
    del model,grads,lp,x,y,data;gc.collect();torch.mps.empty_cache();return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);ap.add_argument('--arm',choices=list(Q.ARMS),required=True);ap.add_argument('--lock-fd',type=int,required=True);arg=ap.parse_args()
    plan=json.loads((R/'plan.json').read_text());require(sha(R/'plan.json')==arg.plan_sha and not (R/(arg.arm+'-attempt.json')).exists() and not (A/'STOP').exists(),'Unattempted exact new arm plan required')
    require(os.fstat(arg.lock_fd).st_ino==(ROOT/'research_state/tokenizer-worker-lock/.lock').stat().st_ino,'Inherited shared worker lock required')
    census=Q.checked(plan['census']);require(census['status']=='new_full_horizon_fixture_roles_unused_before_construction' and not census['prior_matches'] and plan['roles']==census['roles'],'Original new fixture census required')
    for p,h in plan['sources'].items():require(sha(ROOT/p)==h,'Pinned source differs')
    torch.set_num_threads(4);Q.environment();publish(R/(arg.arm+'-attempt.json'),dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
    with StrictMpsGuard(R/(arg.arm+'-mps-guard.json'),ROOT):
        saved_cpu=torch.get_rng_state().clone();saved_mps=torch.mps.get_rng_state().clone()
        try:result=arm_proof(arg.arm,plan)
        finally:torch.set_rng_state(saved_cpu);torch.mps.set_rng_state(saved_mps)
        require(torch.equal(saved_cpu,torch.get_rng_state()) and torch.equal(saved_mps,torch.mps.get_rng_state()),'Caller CPU/MPS RNG changed')
        for p,h in plan['sources'].items():require(sha(ROOT/p)==h,'Source changed')
        result['caller_cpu_mps_rng_unchanged']=True;publish(R/(arg.arm+'-result.json'),result)
    publish(R/(arg.arm+'-STOP'),dict(status='complete_original_generated1200_arm',no_replay=True,no_retry=True))

if __name__=='__main__':main()
