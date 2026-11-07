"""Generated own native model integration;no corpus,checkpoint file or quality trial."""
import copy,gc,hashlib,json,os,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,model_config as native_config,config as native_training
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_model_v1.model import GPT as Native
from research.body_allocation_model_v1.optimizer import build_optimizer as native_optimizer
from research.tokenizer_mla_gate_model_v1.contract import ARMS,model_config
from research.tokenizer_mla_gate_model_v1.model import GPT,load_weights
from research.tokenizer_mla_gate_model_v1.optimizer import build_optimizer
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_worker_v1.execution import idle
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;R=E/'gate-model001';META=2*1024**2;INITIALIZER=600511
FLOOR=8*1024**3+META+1267904512+1458176;TARGET=FLOOR+2*1024**3
TOL=dict(logits_atol=4e-5,logits_rtol=4e-4,ce_atol=3e-5,gradient_atol=4e-5,gradient_rtol=4e-4)
def checked(r):
    p=Path(r['path']);require(reference(p)=={k:r[k] for k in ('path','sha256')},'Immutable evidence differs');return json.loads(p.read_text())
def publish(p,v):
    raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(sum(f.stat().st_size for f in R.glob('*') if f.is_file())+len(raw)<=META,'Complete2MiB metadata cap required')
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def same(a,b):
    if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.cpu(),b.cpu())
    if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return type(a) is type(b) and a==b
def digest(model):
    h=hashlib.sha256()
    for n,t in model.state_dict().items():h.update(n.encode());h.update(str(tuple(t.shape)).encode());h.update(str(t.dtype).encode());h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def construct(arm,device='cpu'):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(INITIALIZER);m=GPT(model_config(arm));after=torch.get_rng_state().clone()
    return m.to(device),after
def tokens(batch=1,length=8):return ((torch.arange(batch*length,dtype=torch.int64).reshape(batch,length)*97+31)%50257)
def gradients(model,x,y):
    model.train();model.zero_grad(set_to_none=True);logits,loss=model(x,y,include_aux_loss=False);loss.backward();g={}
    for n,p in model.named_parameters():require(p.grad is not None and bool(torch.isfinite(p.grad).all()),'Complete finite generated learned gradient required '+n);g[n]=p.grad.detach().cpu().clone()
    return logits.detach().cpu(),float(loss.detach()),g
def rejects(call):
    try:call()
    except (ValueError,RuntimeError):return
    raise AssertionError('Foreign/invalid gate policy accepted')

with session_lock(GLOBAL_LOCK.parent):
    idle();torch.set_num_threads(4);env=environment();require(not R.exists(),'Fresh own model proof only;closed failures cannot retry')
    component=checked(reference(E/'gate-component001/result.json'));component_plan=checked(component['source_plan']);component_guard=checked(reference(E/'gate-component001/guard.json'));component_stop=checked(reference(E/'gate-component001/STOP'));role=checked(reference(E/'fixture-role-census001.json'));design=checked(reference(E/'quality-design001.json'))
    require(component['status']=='elementwise_gate_scalar_output_gradient_and_native_cpu_mps_contracts_passed' and component_stop['never_rerun'] and component_guard['status']=='finished' and component_guard['reason'] is None,'Original complete gate component prerequisite required')
    require(role['no_prior_matches'] and role['generated_model_initializer']==INITIALIZER and not role['scientific_seed'],'Declared new previously unused CPU initializer required')
    pins=dict(component_plan['sources']);require(len(pins)==1057,'All1057 immutable predecessor bindings required')
    for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Immutable predecessor changed '+p)
    for p in [*sorted((ROOT/'research/tokenizer_mla_gate_model_v1').glob('*.py')),Path(__file__).resolve()]:pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==1062,'Four-file own model plus new integration proof required')
    free=shutil.disk_usage(ROOT).free;require(free>=TARGET and free>=design['complete_retained_budget']['target'],'Complete component and future retained-output budget must fit')
    forecast=dict(rss=3186769920+256*1024**2+4*1024**2,mps=2283585536+512*1024**2+8*1024**2,seconds=300);require(forecast['rss']<4*1024**3 and forecast['mps']<6*1024**3 and forecast['seconds']<600,'Complete generated model forecast must fit original guards')
    R.mkdir();publish(R/'plan.json',dict(kind='own_native_mla_gate_generated_model_integration_v1',sources=pins,component=reference(E/'gate-component001/result.json'),roles=reference(E/'fixture-role-census001.json'),design=reference(E/'quality-design001.json'),environment=env,tolerances=TOL,initializer=INITIALIZER,tokens='arange*97+31 modulo50257,targetsplus1mod50257',no_corpus=True,no_quality_seed=True,no_tensor_files=True,full_floor=FLOOR,target=TARGET,free_bytes=free,forecast=forecast,metadata_cap=META))
    try:
        rows=[];mps_before=torch.mps.get_rng_state().clone();cpu_before=torch.get_rng_state().clone()
        with StrictMpsGuard(R/'guard.json',ROOT):
            # Exact baseline on the SAME MPS backend;no cross-device optimizer equivalence claim.
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(INITIALIZER);baseline=Native(native_config('incumbent'));after_native=torch.get_rng_state().clone()
            control,after_control=construct('none');require(torch.equal(after_native,after_control) and same(baseline.state_dict(),control.state_dict()),'Genuine native initialization/RNG/state changed')
            baseline=baseline.to('mps');control=control.to('mps');training=native_training('incumbent',4,INITIALIZER,.00125);a=native_optimizer(baseline,training);b=build_optimizer(control,training.training)
            require(same([o.state_dict()['param_groups'] for o in a.optimizers],[o.state_dict()['param_groups'] for o in b.optimizers]),'Native optimizer grouping/LR/epsilon changed')
            x=tokens().to('mps');y=(x+1)%50257
            for step in range(2):
                a.zero_grad();b.zero_grad();la,ca=baseline(x,y);lb,cb=control(x,y);require(torch.equal(la,lb) and torch.equal(ca,cb),'Ungated native logits/loss changed');ca.backward();cb.backward()
                require(same({n:p.grad for n,p in baseline.named_parameters()},{n:p.grad for n,p in control.named_parameters()}),'Ungated native MPS gradients changed')
                clip_gradients(baseline.parameters(),1.);clip_gradients(control.parameters(),1.);a.step();b.step();require(same(baseline.state_dict(),control.state_dict()) and same([o.state_dict() for o in a.optimizers],[o.state_dict() for o in b.optimizers]),'Ungated native MPS optimizer updates/state changed')
            native_exact=dict(initialization=True,rng=True,logits=True,gradients=True,two_mps_optimizer_steps=True)
            del baseline,control,a,b,x,y,la,lb,ca,cb;gc.collect();torch.mps.empty_cache()
            # Every arm preserves baseline common construction and draws;gates alone addzeros.
            control,_=construct('none');shared={n:t.clone() for n,t in control.state_dict().items()};del control
            for arm in ARMS:
                model,after=construct(arm);require(torch.equal(after_native,after) and all(torch.equal(t,model.state_dict()[n]) for n,t in shared.items()),'Shared model construction/RNG changed')
                if arm=='sigmoid_elementwise_zero':require(all(bool((p==0).all()) for n,p in model.named_parameters() if n.endswith('output_gate.weight')),'Gate did not startzero')
                del model;gc.collect()
            del shared
            # Learned/static initial outputs and ALL common gradients exact on CPU.
            half,_=construct('static_half');gate,_=construct('sigmoid_elementwise_zero');x=tokens();y=(x+1)%50257;lh,ch,gh=gradients(half,x,y);lg,cg,gg=gradients(gate,x,y)
            require(torch.equal(lh,lg) and ch==cg and all(torch.equal(g,gg[n]) for n,g in gh.items()),'Zero-init learned gate and static-half common outputs/gradients differ')
            require(all(bool((g!=0).any()) for n,g in gg.items() if n.endswith('output_gate.weight')),'Zero-initialized gates received no gradient')
            del half,gate,lh,lg,gh,gg;gc.collect()
            for arm in ARMS:
                cpu,_=construct(arm);mps,_=construct(arm,'mps');x=tokens();y=(x+1)%50257;lc,cc,gc_cpu=gradients(cpu,x,y);lm,cm,gm=gradients(mps,x.to('mps'),y.to('mps'));err=float((lc-lm).abs().max())
                require(torch.allclose(lc,lm,atol=TOL['logits_atol'],rtol=TOL['logits_rtol']) and abs(cc-cm)<=TOL['ce_atol'] and gc_cpu.keys()==gm.keys(),'Generated CPU/MPS logits/CE/inventory differs')
                max_grad=max(float((gc_cpu[n]-gm[n]).abs().max()) for n in gm)
                require(all(torch.allclose(gc_cpu[n],gm[n],atol=TOL['gradient_atol'],rtol=TOL['gradient_rtol']) for n in gm),'Generated CPU/MPS learned gradient differs')
                del cpu,gc_cpu,gm,lc,lm;gc.collect()
                opt=build_optimizer(mps,training.training);require(sum(len(o.param_groups) for o in opt.optimizers)==len(list(mps.parameters())),'Complete optimizer roles required')
                gates=[g for g in opt.param_groups if g['parameter_names'][0].endswith('output_gate.weight')]
                require(len(gates)==(4 if arm=='sigmoid_elementwise_zero' else 0) and all(g['lr']==.000625 and g['width_lr_scale']==.5 for g in gates),'Explicit gate Muon/LR roles differ')
                clip_gradients(mps.parameters(),1.);opt.step();require(opt.completed_steps==1 and all(v==1 for v in opt.gradient_steps.values()),'Actual perparameter generated update exposure differs')
                mps.eval();cache_errors=[]
                with torch.no_grad():
                    for batch in (1,8):
                        z=tokens(batch,9).to('mps');full=mps(z)[0];caches=mps.caches(batch);parts=[mps(z[:,:4],cache=caches)[0],mps(z[:,4:],cache=caches,start_pos=4)[0]];cached=torch.cat(parts,dim=1);e=float((full-cached).abs().max());require(torch.allclose(full,cached,atol=TOL['logits_atol'],rtol=TOL['logits_rtol']),'Generated full/cache gate parity differs')
                        altered=z.clone();altered[:,-1]=(altered[:,-1]+17)%50257;changed=mps(altered)[0];require(torch.equal(full[:,:-1],changed[:,:-1]),'Future token changed prior generated logits')
                        for cache in caches:cache.reset();require(cache.position==0 and bool((cache.c_cache==0).all()) and bool((cache.r_cache==0).all()),'Cache reset differs')
                        again=torch.cat([mps(z[:,:4],cache=caches)[0],mps(z[:,4:],cache=caches,start_pos=4)[0]],dim=1);require(torch.equal(cached,again),'Reset generated cache replay differs')
                        cache_errors.append(dict(batch=batch,maxerror=e,bytes_per_item=sum(t.numel()*t.element_size() for c in caches for t in (c.c_cache,c.r_cache))//batch));del full,cached,changed,altered,again,parts,caches,z
                before=digest(mps);payload=dict(kind='tokenizer_mla_gate_weights_v1',architecture=mps.signature,model_state=mps.state_dict());load_weights(mps,payload);rejects(lambda:load_weights(mps,dict(payload,kind='tokenizer_moe_weights_v1')));foreign=copy.deepcopy(payload['architecture']);foreign['arm']='foreign';rejects(lambda:load_weights(mps,dict(payload,architecture=foreign)));missing=dict(payload['model_state']);missing.pop(next(iter(missing)));rejects(lambda:load_weights(mps,dict(payload,model_state=missing)));require(digest(mps)==before,'Rejected own/foreign weights mutated model')
                optimizer_state=opt.state_dict();bad=dict(optimizer_state,tokenizer_mla_gate_signature={});rejects(lambda:opt.load_state_dict(bad));require(same(opt.state_dict(),optimizer_state),'Foreign optimizer signature mutated state')
                rows.append(dict(arm=arm,parameters=mps.signature['parameters'],muon_groups=mps.signature['muon_groups'],adam_groups=mps.signature['adam_groups'],learned_gradients=len(list(mps.parameters())),cpu_mps_logit_maxerror=err,cpu_mps_ce_error=abs(cc-cm),cpu_mps_gradient_maxerror=max_grad,one_generated_mps_step=True,all_perparameter_exposures_one=True,gate_muon_groups=len(gates),cache=cache_errors,transactional_weights_optimizer_rejections=4))
                del mps,opt,payload,optimizer_state,bad,missing,foreign,gates;gc.collect();torch.mps.empty_cache()
            torch.mps.synchronize();require(torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state()),'Global CPU/MPS RNG changed')
            for p,h in pins.items():require(reference(ROOT/p)['sha256']==h,'Frozen source changed during model proof')
        result=dict(status='own_native_mla_gate_model_optimizer_gradient_and_cache_integration_passed',source_plan=reference(R/'plan.json'),source_bindings=1062,native_control_exact=native_exact,static_learned_zero_common_gradients_exact=True,common_initial_weights_and_rng_exact=True,rows=rows,initializer_role_used=INITIALIZER,global_cpu_mps_rng_unchanged=True,corpus_loaded=False,tensor_files_written=0,quality_seed_assigned=False,quality_result=False,adopted=False,scope='Generated token own-model proof only. Real native-data/RAW-H200/EMA/optimizer perparameter transport/replay,serialized resource layouts,workers,pilots and quality remain unproved.')
        publish(R/'result.json',result);publish(R/'STOP',dict(status='complete_generated_own_mla_gate_model_integration',result=reference(R/'result.json'),never_rerun=True));print(json.dumps(dict(status=result['status'],source_bindings=1062,rows=rows)),flush=True)
    except BaseException as exc:
        if not(R/'STOP').exists():publish(R/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
        raise
