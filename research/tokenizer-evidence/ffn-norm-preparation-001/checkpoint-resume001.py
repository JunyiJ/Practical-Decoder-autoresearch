"""New strict-MPS generated-text file transport/repeat/resume prerequisite."""
import copy, datetime, gc, hashlib, json, os, shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.mla_score_quality_v1 import campaign_contract as c
from research.ema_horizon_v1 import execution as e
from research.mla_score_quality_v1.support import idle
from research.ffn_structure_v1 import pilot_contract as pilot
from research.block_norm_v1.fixture_dispatch import read, reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.ffn_norm_v1.runtime.models.gpt import GPT
from research.ffn_norm_v1.runtime.models.block_norm import FFN_NORM_POLICY
from research.ffn_norm_v1.runtime.models.output_bias import tokenizer_no_bias_prior, validate_output_bias_training_data
from research.ffn_norm_v1.runtime.train.train import _make_loader, _training_contract
from research.ffn_norm_v1.runtime.train.optimizers import build_optimizer
from research.ffn_norm_v1.runtime.train.lr_schedule import LearningRateSchedule
from research.ffn_norm_v1.runtime.train.weight_averaging import TailWeightAverages, export_averaged_checkpoint
from research.ffn_norm_v1.runtime.utils.checkpoint import save_checkpoint, read_checkpoint, load_model_weights, capture_training_state, restore_training_state, checkpoint_loader
from research.ffn_norm_v1.runtime.attention.cache import build_caches
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
B=Path(__file__).resolve().parent
ROOT=c.ROOT/'research_runs/tokenizer-ffn-norm-checkpoint-prerequisite-001'
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
    path=Path(path);assert path.parent in (ROOT/'none',ROOT/'rms') and not path.is_symlink()
    assert path.name in ('step1.pt','final.pt','h200.pt')
    with path.open('xb') as f:torch.save(payload,CappedWriter(f,EXPORT_CAP if path.name=='h200.pt' else RAW_CAP))
def emit(event,**fields):
    with (B/'checkpoint-resume001-trace.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=event,**fields))+'\n');f.flush();os.fsync(f.fileno());assert f.tell()<2*1024**2

def components(cfg):
    data=_make_loader(cfg,'mps');validate_output_bias_training_data(cfg.model,data,cfg.data.path)
    assert set(data.splits)==set(data.generators)=={'train','val'}
    model=GPT(cfg.model).to('mps');opt=build_optimizer(model,cfg.training)
    return model,opt,LearningRateSchedule(opt,cfg.training),TailWeightAverages(model,cfg.training),data

def state(parts):
    model,opt,schedule,ema,data=parts
    return cpu(dict(model=model.state_dict(),optimizer=opt.state_dict(),ema=ema.state_dict(),
        rng=capture_training_state(data,'mps')))
def update(parts,i):
    model,opt,schedule,ema,data=parts
    x,y=data.get_batch('train');assert tuple(x.shape)==tuple(y.shape)==(8,128)
    a=torch.rand(());b=torch.rand((),device='mps')
    _,loss=model(x,targets=y);loss=loss*(.8+.1*a+.1*b)
    opt.zero_grad(set_to_none=True);loss.backward()
    assert all(p.grad is not None and p.grad.device.type=='mps' and p.grad.dtype==torch.float32 and bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    schedule.apply(i);opt.step();ema.update(i+1);torch.mps.synchronize()
    return cpu(dict(x=x,y=y,cpu_draw=a,mps_draw=b,loss=loss))
def save(parts,cfg,path,step):
    model,opt,_,ema,data=parts
    progress=capture_training_state(data,'mps',contract=_training_contract(cfg,'mps',data),training_seconds=0.,metrics=[],best_metric=None,best_checkpoint=None)
    progress['weight_averaging_state']=ema.state_dict()
    save_checkpoint(path,model,opt,step=step,cfg=cfg,training_state=progress,payload_writer=write_payload)
    emit('saved',path=str(path),bytes=path.stat().st_size,sha256=sha(path),step=step)
def rejected(fn):
    try:fn()
    except ValueError as exc:return str(exc)
    raise AssertionError('Malformed checkpoint accepted')
def clear():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle();assert not ROOT.exists() and not (B/'checkpoint-resume001-trace.jsonl').exists()
    plan=json.loads((B/'checkpoint-resume001-plan.json').read_text())
    for n,h in plan['sources'].items():assert sha(c.ROOT/n)==h
    bind=read(reference(B/'model001.json'));assert bind['status']=='full_model_product_norm_cpu_mps_prerequisite_passed'
    frozen=c.source_inventory();assert frozen==bind['frozen_sources']
    for n,h in bind['new_sources'].items():assert sha(c.ROOT/n)==h
    assert shutil.disk_usage(c.ROOT).free>=8*1024**3+2*(2*RAW_CAP+EXPORT_CAP)+256*1024**2
    ROOT.mkdir();(B/'checkpoint-resume001-trace.jsonl').touch(exist_ok=False)
    outcome=None
    try:
        with StrictMpsGuard(B/'checkpoint-resume001-guard.json',c.ROOT):
            text=''.join(f'Example {i}: amber birds cross a quiet river. Red green blue; count {i%17}.\n' for i in range(800))
            corpus=ROOT/'generated.txt';corpus.write_text(text)
            seeds=dict(pilot=90841,screen=90842,fresh=[90851,90852,90853],evaluation=90843,bootstrap=90844)
            base=OmegaConf.create(pilot.config(0,seeds))
            base.model.tokenizer_architecture_policy=FFN_NORM_POLICY
            base.model.ffn_hidden_norm='none';base.model.ffn_hidden_norm_policy='mla_swiglu_product_rms_672_v1'
            base.data.path=str(corpus);base.data.train_fraction=.8;base.data.val_fraction=.2
            base.training.max_iters=3;base.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=3,start_step=1,interval=1,half_lives=[200])
            base.training.lr_schedule=dict(kind='linear_cooldown',total_steps=3,start_step=1,min_ratio=.1)
            loader=_make_loader(base,'cpu');base.model.output_bias_prior=tokenizer_no_bias_prior(loader,str(corpus));del loader
            configs={};rows=[]
            for arm in ('none','rms'):
                root=ROOT/arm;root.mkdir();cfg=copy.deepcopy(base)
                cfg.training.checkpoint_dir=str(root);cfg.training.metrics_file=str(root/'metrics.jsonl');cfg.training.summary_file=str(root/'summary.json')
                cfg.model.ffn_hidden_norm=arm
                configs[arm]=OmegaConf.to_container(cfg,resolve=True)
                e.exclusive(root/'config.json',configs[arm]);emit('begin_arm',arm=arm)
                torch.manual_seed(90845);a=components(cfg);obs=[update(a,0)]
                save(a,cfg,root/'step1.pt',1);initial=state(a)
                obs.extend(update(a,i) for i in (1,2));expected=state(a)
                save(a,cfg,root/'final.pt',3);del a;clear()
                # A new run must match before interrupted replay is interpreted.
                torch.manual_seed(90845);b=components(cfg)
                for i in range(3):
                    actual=update(b,i)
                    for key in ('x','y','cpu_draw','mps_draw'):compare(actual[key],obs[i][key],0.,0.)
                    compare(actual['loss'],obs[i]['loss'],1e-5,1e-5)
                repeated=state(b);repeat_error=compare(repeated,expected);compare(repeated['rng'],expected['rng'],0.,0.)
                del b,repeated,actual;clear();emit('fresh_repeat_passed',arm=arm,max_abs=repeat_error)
                b=components(cfg);saved=read_checkpoint(root/'step1.pt')
                # Reject malformed or inference-only resume before modifying model weights.
                before=cpu(b[0].state_dict());before_rng=cpu(capture_training_state(b[4],'mps'))
                errors={}
                for kind in ('missing_optimizer','group_order','missing_mps_rng'):
                    bad=dict(saved)
                    if kind=='missing_optimizer':del bad['optimizer_state']
                    elif kind=='missing_mps_rng':
                        bad['training_state']=dict(saved['training_state']);del bad['training_state']['mps_rng']
                    else:
                        bad['optimizer_state']=dict(saved['optimizer_state']);bad['optimizer_state']['optimizers']=[dict(p) for p in saved['optimizer_state']['optimizers']]
                        panel=bad['optimizer_state']['optimizers'][0];panel['param_groups']=list(reversed(panel['param_groups']))
                    errors[kind]=rejected(lambda:load_model_weights(b[0],bad,for_resume=True))
                    compare(b[0].state_dict(),before,0.,0.);compare(capture_training_state(b[4],'mps'),before_rng,0.,0.)
                del before,before_rng,bad
                load_model_weights(b[0],saved,for_resume=True);b[1].load_state_dict(saved['optimizer_state'])
                restore_training_state(saved['training_state'],b[4],'mps');b[3].load_state_dict(saved['training_state']['weight_averaging_state'],1)
                compare(state(b),initial,0.,0.);del saved,initial
                for i in (1,2):
                    actual=update(b,i)
                    for key in ('x','y','cpu_draw','mps_draw'):compare(actual[key],obs[i][key],0.,0.)
                    compare(actual['loss'],obs[i]['loss'],1e-5,1e-5)
                restored=state(b);resume_error=compare(restored,expected);compare(restored['rng'],expected['rng'],0.,0.)
                del b,restored,actual,expected,obs;clear();emit('resume_passed',arm=arm,max_abs=resume_error)
                raw_hash=sha(root/'final.pt');cpu_rng=torch.get_rng_state();mps_rng=torch.mps.get_rng_state()
                export=export_averaged_checkpoint(root/'final.pt',root/'h200.pt',200,payload_writer=write_payload)
                assert sha(root/'final.pt')==raw_hash and torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
                raw=read_checkpoint(root/'final.pt');avg=read_checkpoint(root/'h200.pt')
                averages=raw['training_state']['weight_averaging_state']['averages']['200']
                assert 'optimizer_state' not in avg and 'training_state' not in avg
                assert avg['parent_checkpoint']==dict(path=str(root/'final.pt'),sha256=raw_hash)
                assert avg['block_normalization_signature']==raw['block_normalization_signature']
                for n,v in avg['model_state'].items():assert torch.equal(v,averages[n] if n in averages else raw['model_state'][n])
                check_loader=checkpoint_loader(copy.deepcopy(cfg),avg,corpus,'cpu');assert set(check_loader.splits)=={'train','val'}
                cpu_model=GPT(cfg.model).eval();load_model_weights(cpu_model,avg)
                gpu_model=GPT(cfg.model).to('mps').eval();load_model_weights(gpu_model,avg)
                errors['h200_resume']=rejected(lambda:load_model_weights(gpu_model,avg,for_resume=True))
                inference=[]
                with torch.no_grad():
                    for batch in (1,8):
                        x=((torch.arange(batch*17)*7)%13).reshape(batch,17);full=gpu_model(x.to('mps'))[0];ref=cpu_model(x)[0]
                        cross=compare(full,ref,3e-4,3e-4);cache=build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32)
                        pieces=[];pos=0
                        for count in (5,1,4,7):pieces.append(gpu_model(x[:,pos:pos+count].to('mps'),cache=cache,start_pos=pos)[0]);pos+=count
                        cached=compare(torch.cat(pieces,1),full.cpu(),3e-4,3e-4)
                        inference.append(dict(batch=batch,cpu_mps_max_abs=cross,cached_max_abs=cached))
                row=dict(arm=arm,repeat_max_abs=repeat_error,resume_max_abs=resume_error,restored_state_and_rng_exact=True,rejections=errors,
                    inference=inference,files={p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in root.iterdir()},export=export)
                rows.append(row);emit('arm_completed',**row)
                del raw,avg,averages,cpu_model,gpu_model,check_loader,cache,pieces,full,ref;clear()
            assert c.source_inventory()==frozen
            for n,h in bind['new_sources'].items():assert sha(c.ROOT/n)==h
            outcome=dict(status='serialized_product_norm_mps_repeat_resume_h200_passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner=reference(__file__),
                root=str(ROOT),source_binding=reference(B/'model001.json'),configs=configs,arms=rows,generated_corpus=reference(corpus),
                generated_training_updates=16,scientific_training_launched=False,research_corpus_read=False,holdout_read=False,
                limits=dict(raw_cap_bytes=RAW_CAP,export_cap_bytes=EXPORT_CAP),resume_tolerance=dict(atol=2e-6,rtol=2e-4),
                remaining='Actual run_training/output sink plus serial worker and independent parent full-validation acceptance; scientific resource/storage and finite freezes.',
                retention='Keep new fixture files until downstream acceptance and exact cleanup audit; all failed attempts remain protected.')
    except Exception as exc:
        emit('failed',error=repr(exc))
        e.exclusive(B/'checkpoint-resume001-failure.json',dict(status='failed',error=repr(exc),runner=reference(__file__),root=str(ROOT),trace=reference(B/'checkpoint-resume001-trace.jsonl')))
        raise
    outcome['guard']=reference(B/'checkpoint-resume001-guard.json');assert read(outcome['guard'])['status']=='finished'
    e.exclusive(B/'checkpoint-resume001.json',outcome);print(json.dumps(dict(status=outcome['status'],arms=[dict(arm=r['arm'],resume_max_abs=r['resume_max_abs'],repeat_max_abs=r['repeat_max_abs'],inference=r['inference']) for r in rows])),flush=True)
