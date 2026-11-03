"""Both final LayerNorm scale policies: generated four-update repeat, serialized resume and final H200."""
import copy, datetime, gc, hashlib, io, json, os, shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.warmup_quality_v1 import campaign_contract as c
from research.ema_horizon_v1 import execution as e
from research.gradient_clip_quality_v2.quality_execution import idle
from research.ffn_structure_v1 import pilot_contract as pilot
from research.block_norm_v1.fixture_dispatch import read, reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.final_ln_scale_model_v1.runtime.models.gpt import GPT
from research.final_ln_scale_model_v1.runtime.models.block_norm import READOUT_POLICY
from research.final_ln_scale_model_v1.runtime.models.output_bias import tokenizer_no_bias_prior, validate_output_bias_training_data
from research.final_ln_scale_model_v1.training_state import _make_loader, _training_contract, _contracts_match, validate_history
from research.gradient_clip_model_v1.clipping import clip_gradients
from research.final_ln_scale_model_v1.integration import build_optimizer, load_model_weights as checked_load
from research.final_ln_scale_model_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.final_ln_scale_model_v1.runtime.train.weight_averaging import TailWeightAverages, export_averaged_checkpoint
from research.final_ln_scale_model_v1.runtime.utils.checkpoint import save_checkpoint, read_checkpoint, load_model_weights, capture_training_state, restore_training_state, checkpoint_loader
from research.final_ln_scale_model_v1.runtime.attention.cache import build_caches
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
B=Path(__file__).resolve().parent
ROOT=c.ROOT/'research_runs/tokenizer-final-ln-scale-resume-002'
SERIALIZED_STEP2 = None
RAW_CAP=448*1024**2;EXPORT_CAP=112*1024**2

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def cpu(value):
    if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
    if isinstance(value,dict):return type(value)((k,cpu(v)) for k,v in value.items())
    if isinstance(value,list):return [cpu(v) for v in value]
    if isinstance(value,tuple):return tuple(cpu(v) for v in value)
    return copy.deepcopy(value)
def compare(actual,expected,atol=2e-6,rtol=2e-4):
    if isinstance(expected,torch.Tensor):
        actual=actual.detach().cpu();assert actual.shape==expected.shape and actual.dtype==expected.dtype
        torch.testing.assert_close(actual,expected,atol=atol,rtol=rtol)
        return float((actual-expected).abs().max()) if actual.numel() else 0.
    assert type(actual) is type(expected)
    if isinstance(expected,dict):
        assert actual.keys()==expected.keys();return max((compare(actual[k],v,atol,rtol) for k,v in expected.items()),default=0.)
    if isinstance(expected,(tuple,list)):
        assert len(actual)==len(expected);return max((compare(a,b,atol,rtol) for a,b in zip(actual,expected)),default=0.)
    assert actual==expected;return 0.
class CappedWriter:
    def __init__(self,stream,limit):self.stream,self.limit=stream,limit
    def write(self,value):
        assert self.tell()+memoryview(value).nbytes<=self.limit,'Checkpoint cap exceeded'
        return self.stream.write(value)
    def tell(self):return self.stream.tell()
    def flush(self):return self.stream.flush()
def write_payload(path,payload):
    path=Path(path);assert path.parent in (ROOT/'learned', ROOT/'unit') and not path.is_symlink()
    assert path.name in ('final.pt','h200.pt')
    with path.open('xb') as f:torch.save(payload,CappedWriter(f,EXPORT_CAP if path.name=='h200.pt' else RAW_CAP))
def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    return value
def emit(event,**fields):
    with (B/'model-resume002-trace.jsonl').open('a') as f:
        f.write(json.dumps(json_safe(dict(event=event,**fields)))+'\n');f.flush();os.fsync(f.fileno());assert f.tell()<2*1024**2

def components(cfg):
    data=_make_loader(cfg,'mps');validate_output_bias_training_data(cfg.model,data,cfg.data.path)
    assert set(data.splits)==set(data.generators)=={'train','val'}
    model=GPT(cfg.model).to('mps');opt=build_optimizer(model,cfg)
    return model,opt,LearningRateSchedule(opt,cfg.training),TailWeightAverages(model,cfg.training),data,[]

def digest_state(value):
    # Exact dtype/shape/SHA byte identities avoid keeping multiple full reference
    # model/optimizer/EMA tensor trees alive. RNG and scalar metadata stay exact.
    if isinstance(value,torch.Tensor):
        x=value.detach().cpu().contiguous()
        return dict(tensor_dtype=str(x.dtype),tensor_shape=list(x.shape),tensor_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest())
    if isinstance(value,dict):return type(value)((k,digest_state(v)) for k,v in value.items())
    if isinstance(value,list):return [digest_state(v) for v in value]
    if isinstance(value,tuple):return tuple(digest_state(v) for v in value)
    return copy.deepcopy(value)
def state(parts):
    model,opt,schedule,ema,data,history=parts
    return digest_state(dict(model=model.state_dict(),optimizer=opt.state_dict(),ema=ema.state_dict(),rng=capture_training_state(data,'mps'),clipping_observations=history))
def update(parts,i,threshold):
    model,opt,schedule,ema,data,history=parts
    x,y=data.get_batch('train');assert tuple(x.shape)==tuple(y.shape)==(8,128)
    a=torch.rand(());b=torch.rand((),device='mps')
    _,loss=model(x,targets=y);loss=loss*(.8+.1*a+.1*b)
    opt.zero_grad(set_to_none=True);loss.backward()
    assert all(p.grad is not None and p.grad.device.type=='mps' and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    clipping=dict(step=i+1,**clip_gradients(model.parameters(),threshold))
    ratio=schedule.apply(i)
    assert all(g['lr']==base*ratio for g,base in zip(opt.param_groups,schedule.base_lrs))
    opt.step();ema.update(i+1);history.append(clipping);torch.mps.synchronize()
    return cpu(dict(x=x,y=y,cpu_draw=a,mps_draw=b,loss=loss,clipping=clipping))
def save(parts,cfg,path,step):
    model,opt,_,ema,data,history=parts
    progress=capture_training_state(data,'mps',contract=_training_contract(cfg,'mps',data),training_seconds=0.,metrics=[],best_metric=None,best_checkpoint=None)
    progress['weight_averaging_state']=ema.state_dict()
    progress['clipping_observations']=validate_history(history,step,cfg)
    global SERIALIZED_STEP2
    if step == 2:
        stream=io.BytesIO()
        save_checkpoint(path,model,opt,step=step,cfg=cfg,training_state=progress,payload_writer=lambda p,payload:torch.save(payload,CappedWriter(stream,RAW_CAP)))
        SERIALIZED_STEP2=stream.getvalue()
        emit('serialized_step2_memory',bytes=len(SERIALIZED_STEP2),sha256=hashlib.sha256(SERIALIZED_STEP2).hexdigest(),step=step)
    else:
        assert step == 4 and path.name == 'final.pt'
        save_checkpoint(path,model,opt,step=step,cfg=cfg,training_state=progress,payload_writer=write_payload)
        emit('saved',path=str(path),bytes=path.stat().st_size,sha256=sha(path),step=step)
def rejected(fn):
    try:fn()
    except ValueError as exc:return str(exc)
    raise AssertionError('Malformed checkpoint accepted')
def clear():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle();assert not ROOT.exists() and not (B/'model-resume002-trace.jsonl').exists()
    plan=json.loads((B/'model-resume002-plan.json').read_text())
    for n,h in plan['sources'].items():assert sha(c.ROOT/n)==h
    frozen=plan['frozen_sources']
    for n,h in frozen.items():assert sha(c.ROOT/n)==h
    assert read(reference(B/'model001.json'))['status']=='passed'
    assert shutil.disk_usage(c.ROOT).free>=8*1024**3+2*(RAW_CAP+EXPORT_CAP)+16*1024**2+1267904512  # Additional observed filesystem headroom; original reserve unchanged.
    ROOT.mkdir();(B/'model-resume002-trace.jsonl').touch(exist_ok=False)
    outcome=None
    try:
        with StrictMpsGuard(B/'model-resume002-guard.json',c.ROOT):
            text=''.join(f'Example {i}: amber birds cross a quiet river. Red green blue; count {i%17}.\n' for i in range(800))
            corpus=ROOT/'generated.txt';corpus.write_text(text)
            seeds=dict(pilot=91671,screen=91672,fresh=[91681,91682,91683],evaluation=91673,bootstrap=91674)
            base=OmegaConf.create(pilot.config(0,seeds))
            base.model.tokenizer_architecture_policy=READOUT_POLICY
            base.model.residual_gain_width_policy='mla4_fixed_complete_branch_gain_v1'
            base.model.residual_gain_kind='unit'
            base.model.readout_norm_kind='layer_affine';base.model.readout_norm_width_policy='mla4_final_ln_scale_v1';base.model.readout_norm_epsilon=1e-5
            base.model.final_ln_bias_policy='mla4_final_ln_bias_v1';base.model.final_ln_bias_kind='learned'
            base.model.final_ln_scale_policy='mla4_final_ln_scale_v1';base.model.final_ln_scale_kind='learned'
            base.training.adamw_epsilon_policy='mla4_adamw_epsilon_fixed_v1';base.training.adamw_epsilon=1e-6
            base.data.path=str(corpus);base.data.train_fraction=.8;base.data.val_fraction=.2
            base.training.max_iters=4;base.training.learning_rate=.00125
            base.training.adamw_beta2_policy='mla4_adamw_beta2_fixed_v1';base.training.adamw_beta2=.999
            base.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=4,start_step=2,interval=1,half_lives=[200])
            base.training.lr_schedule=dict(kind='linear_cooldown',total_steps=4,start_step=2,min_ratio=.1)
            loader=_make_loader(base,'cpu');base.model.output_bias_prior=tokenizer_no_bias_prior(loader,str(corpus));del loader
            configs={};rows=[]
            for arm,scale in (('learned','learned'),('unit','fixed_unit')):
                root=ROOT/arm;root.mkdir();cfg=copy.deepcopy(base);cfg.training.weight_decay=.00006;cfg.training.grad_clip=1.;cfg.training.adamw_beta2_policy='mla4_adamw_beta2_fixed_v1';cfg.training.adamw_beta2=.999;cfg.model.final_ln_scale_kind=scale
                cfg.training.checkpoint_dir=str(root);cfg.training.metrics_file=str(root/'metrics.jsonl');cfg.training.summary_file=str(root/'summary.json')
                configs[arm]=OmegaConf.to_container(cfg,resolve=True)
                e.exclusive(root/'config.json',configs[arm]);emit('begin_arm',arm=arm)
                torch.manual_seed(91675);a=components(cfg)
                # The full model/group identity was independently checked in model001.
                assert sum(p.numel() for p in a[0].parameters())==28492416-(256 if scale=='fixed_unit' else 0)
                assert len(list(a[0].parameters()))==(87 if scale=='fixed_unit' else 88)
                groups=[{k:v for k,v in g.items() if k!='params'} for g in a[1].param_groups]
                obs=[update(a,0,cfg.training.grad_clip),update(a,1,cfg.training.grad_clip)]
                save(a,cfg,root/'step2.pt',2);initial=state(a)
                assert a[3].state_dict()['averages']['200']
                obs.extend(update(a,i,cfg.training.grad_clip) for i in (2,3));expected=state(a)
                del a;clear()
                # A new run must match before interrupted replay is interpreted.
                torch.manual_seed(91675);b=components(cfg)
                for i in range(4):
                    actual=update(b,i,cfg.training.grad_clip)
                    for key in ('x','y','cpu_draw','mps_draw','clipping'):compare(actual[key],obs[i][key],0.,0.)
                    compare(actual['loss'],obs[i]['loss'],1e-5,1e-5)
                repeated=state(b);repeat_error=compare(repeated,expected,0.,0.);compare(repeated['rng'],expected['rng'],0.,0.)
                del b,repeated,actual;clear();emit('fresh_repeat_passed',arm=arm,state_digest_equal=(repeat_error==0.))
                b=components(cfg);saved=torch.load(io.BytesIO(SERIALIZED_STEP2),map_location='cpu',weights_only=False)
                SERIALIZED_STEP2=None;clear()  # Loaded tensors own the decoded state; serialized bytes are no longer needed.
                # Reject malformed or inference-only resume before modifying model weights.
                before=cpu(b[0].state_dict());before_rng=cpu(capture_training_state(b[4],'mps'))
                errors={}
                own=_training_contract(cfg,'mps',b[4])
                assert _contracts_match(saved['training_state']['contract'],own)
                other=copy.deepcopy(cfg);other.model.final_ln_scale_kind='fixed_unit' if scale=='learned' else 'learned'
                assert not _contracts_match(saved['training_state']['contract'],_training_contract(other,'mps',b[4]))
                errors['other_arm_training_contract']='Rejected by actual trainer contract before loading'
                for kind in ('missing_optimizer','group_order','missing_mps_rng','scheduled_rate','foreign_beta2'):
                    bad=dict(saved)
                    if kind=='missing_optimizer':del bad['optimizer_state']
                    elif kind=='missing_mps_rng':
                        bad['training_state']=dict(saved['training_state']);del bad['training_state']['mps_rng']
                    elif kind=='scheduled_rate':
                        bad['optimizer_state']=copy.deepcopy(saved['optimizer_state'])
                        bad['optimizer_state']['optimizers'][0]['param_groups'][0]['lr']*=.5
                    elif kind=='foreign_beta2':
                        bad['optimizer_state']=copy.deepcopy(saved['optimizer_state'])
                        bad['optimizer_state']['optimizers'][1]['param_groups'][0]['betas']=(.9,.95)
                    else:
                        bad['optimizer_state']=dict(saved['optimizer_state']);bad['optimizer_state']['optimizers']=[dict(p) for p in saved['optimizer_state']['optimizers']]
                        panel=bad['optimizer_state']['optimizers'][0];panel['param_groups']=list(reversed(panel['param_groups']))
                    errors[kind]=rejected(lambda:checked_load(b[0],bad,cfg,for_resume=True))
                    compare(b[0].state_dict(),before,0.,0.);compare(capture_training_state(b[4],'mps'),before_rng,0.,0.)
                del before,before_rng,bad
                errors['cross_final_scale']=rejected(lambda:checked_load(b[0],saved,other,for_resume=True))
                checked_load(b[0],saved,cfg,for_resume=True);b[1].load_state_dict(saved['optimizer_state'])
                b[5].extend(validate_history(saved['training_state'].get('clipping_observations'),2,cfg))
                errors['missing_history']=rejected(lambda:validate_history(None,2,cfg))
                restore_training_state(saved['training_state'],b[4],'mps');b[3].load_state_dict(saved['training_state']['weight_averaging_state'],2)
                compare(state(b),initial,0.,0.);del saved,initial
                for i in (2,3):
                    actual=update(b,i,cfg.training.grad_clip)
                    for key in ('x','y','cpu_draw','mps_draw','clipping'):compare(actual[key],obs[i][key],0.,0.)
                    compare(actual['loss'],obs[i]['loss'],1e-5,1e-5)
                restored=state(b);resume_error=compare(restored,expected,0.,0.);compare(restored['rng'],expected['rng'],0.,0.)
                save(b,cfg,root/'final.pt',4)
                SERIALIZED_STEP2=None
                del b,restored,actual,expected,obs;clear();emit('resume_passed',arm=arm,state_digest_equal=(resume_error==0.))
                raw_hash=sha(root/'final.pt');cpu_rng=torch.get_rng_state();mps_rng=torch.mps.get_rng_state()
                export=export_averaged_checkpoint(root/'final.pt',root/'h200.pt',200,payload_writer=write_payload)
                assert sha(root/'final.pt')==raw_hash and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
                raw=read_checkpoint(root/'final.pt');avg=read_checkpoint(root/'h200.pt')
                averages=raw['training_state']['weight_averaging_state']['averages']['200']
                assert len(averages)==(87 if scale=='fixed_unit' else 88)
                assert ('transformer.ln_f.weight' in averages)==(scale=='learned')
                assert 'optimizer_state' not in avg and 'training_state' not in avg
                assert avg['parent_checkpoint']==dict(path=str(root/'final.pt'),sha256=raw_hash)
                assert avg['block_normalization_signature']==raw['block_normalization_signature']
                for n,v in avg['model_state'].items():assert torch.equal(v,averages[n] if n in averages else raw['model_state'][n])
                check_loader=checkpoint_loader(copy.deepcopy(cfg),avg,corpus,'cpu');assert set(check_loader.splits)=={'train','val'}
                cpu_model=GPT(cfg.model).eval();checked_load(cpu_model,avg,cfg)
                gpu_model=GPT(cfg.model).to('mps').eval();checked_load(gpu_model,avg,cfg)
                errors['h200_resume']=rejected(lambda:checked_load(gpu_model,avg,cfg,for_resume=True))
                inference=[]
                with torch.no_grad():
                    for batch in (1,8):
                        x=((torch.arange(batch*17)*7)%13).reshape(batch,17);full=gpu_model(x.to('mps'))[0];ref=cpu_model(x)[0]
                        cross=compare(full,ref,3e-4,3e-4);cache=build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32)
                        pieces=[];pos=0
                        for count in (5,1,4,7):pieces.append(gpu_model(x[:,pos:pos+count].to('mps'),cache=cache,start_pos=pos)[0]);pos+=count
                        cached=compare(torch.cat(pieces,1),full.cpu(),3e-4,3e-4)
                        from src.research.tokenizer_wide_hybrid_measurements import check_kernel
                        audit=check_kernel(gpu_model,((torch.arange(96)*7)%13).to('mps').unsqueeze(0).repeat(batch,1),lambda:build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32))
                        cpu_audit=check_kernel(cpu_model,((torch.arange(96)*7)%13).unsqueeze(0).repeat(batch,1),lambda:build_caches(cfg,'cpu',batch_size=batch,dtype=torch.float32))
                        inference.append(dict(batch=batch,cpu_mps_max_abs=cross,cached_max_abs=cached,reset_kernel=audit,cpu_reset_kernel=cpu_audit))
                row=dict(arm=arm,final_ln_scale=scale,parameter_count=28492416-(256 if scale=='fixed_unit' else 0),ema_tensors=len(averages),grad_clip=cfg.training.grad_clip,clipping_observations=raw['training_state']['clipping_observations'],optimizer_groups=groups,resume_checkpoint_step=2,export_checkpoint_step=4,final_step=4,repeat_state_digest_equal=(repeat_error==0.),resume_state_digest_equal=(resume_error==0.),restored_state_and_rng_exact=True,state_comparison='dtype/shape/SHA256 byte identities and exact scalar/RNG metadata; no full reference tensor trees retained',rejections=errors,
                    inference=inference,files={p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in root.iterdir()},export=export)
                rows.append(row);emit('arm_completed',**row)
                del raw,avg,averages,cpu_model,gpu_model,check_loader,cache,pieces,full,ref;clear()
            for n,h in frozen.items():assert sha(c.ROOT/n)==h
            for n,h in plan['sources'].items():assert sha(c.ROOT/n)==h
            outcome=dict(status='final_ln_scale_both_arms_serialized_resume_final_h200_passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner=reference(__file__),
                root=str(ROOT),source_binding=reference(B/'model-resume002-plan.json'),configs=configs,arms=rows,generated_corpus=reference(corpus),
                generated_training_updates=20,fixture_scope='Both final LayerNorm scale policies four-update generated repeat/resume from serialized in-memory step2; final-step4 raw and H200 exports, three snapshots each. Actual worker/parent full-validation remains required. No scientific trajectory.',scientific_training_launched=False,research_corpus_read=False,holdout_read=False,
                limits=dict(raw_cap_bytes=RAW_CAP,export_cap_bytes=EXPORT_CAP),resume_tolerance=dict(atol=2e-6,rtol=2e-4),
                remaining='Actual run_training/output sink plus serial worker and independent parent full-validation acceptance; scientific resource/storage and finite freezes.',
                retention='Keep new fixture files until downstream acceptance and exact cleanup audit; all failed attempts remain protected.')
    except Exception as exc:
        emit('failed',error=repr(exc))
        e.exclusive(B/'model-resume002-failure.json',dict(status='failed',error=repr(exc),runner=reference(__file__),root=str(ROOT),trace=reference(B/'model-resume002-trace.jsonl')))
        raise
    outcome['guard']=reference(B/'model-resume002-guard.json');assert read(outcome['guard'])['status']=='finished'
    e.exclusive(B/'model-resume002.json',json_safe(outcome));print(json.dumps(dict(status=outcome['status'],arms=[dict(arm=r['arm'],resume_max_abs=r['resume_max_abs'],repeat_max_abs=r['repeat_max_abs']) for r in rows])),flush=True)
