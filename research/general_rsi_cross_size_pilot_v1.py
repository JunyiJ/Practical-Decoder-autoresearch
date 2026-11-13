"""Bounded engineering qualification; no scientific quality selection."""
import argparse,datetime,fcntl,gc,json,math,os,platform,shutil,signal,statistics,time,traceback
from pathlib import Path
from types import SimpleNamespace
import torch
from research.general_rsi_pseudo_update_diagnostic_v1 import sha,put,check_stop,digest
from research.general_rsi_cross_size_training_v1 import Context
from research.general_rsi_cross_size_v1.model import DepthModel
from research.general_rsi_cross_size_v1.contract import Spec
from research.byte_input_model_v6 import build_codebook
from research.general_rsi_training_v2 import write_checkpoint,read_checkpoint,score
from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
from research.byte_context_cpu_reference_v1 import cpu_cross_entropy
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001'
RUN=ROOT/'research_runs/general-rsi-cross-size-pilot-001'
GIB=1024**3

def read(p):return json.loads(p.read_text())
def prepare():
    prior=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-fixture-001/plan.json')
    fixture=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-fixture-001/attempt-01/result.json'
    assert read(fixture)['accepted']
    sources=dict(prior['sources'])
    for path in [Path(__file__),ROOT/'research/general_rsi_cross_size_training_v1.py',ROOT/'research/general_rsi_training_v2.py',ROOT/'research/general_rsi_pseudo_update_diagnostic_v1.py',fixture]:sources[str(path.relative_to(ROOT))]=sha(path)
    allocation=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-boundary-001/allocation.json')
    p=dict(kind='cross_size_new_depth_engineering_v1',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,codebook_assets=prior['codebook_assets'],torch_version=torch.__version__,python_version=platform.python_version(),roles=dict(native_initializer=62111312020101,composer_initializer=62111312020102,train_sampler=62111312020103,validation_sampler=62111312020104),extra_group_seeds=[62111312020105,62111312020106,62111312020107],slots=[dict(depth=d,arm=a,steps=20 if d==4 else 40,replay_from=None if d==4 else 30) for d in [4,8,16] for a in ['native','raw_bytes']],fresh_role_audit='rg JSON/JSONL before preparation returned no matches',horizon=40,logits_atol=3e-5,logits_rtol=3e-4,scoring_ce_atol=1e-4,criteria='Finite full1024-target updates; exact new-depth complete-state save/load and10-update replay with EMA active; wrong-depth restore rejects before mutation; CPU/MPS corrected logits and training-only scorer oracle within fixed tolerances; exact paired batch streams. No quality gate.',max_seconds_per_slot=600,max_total_seconds=3600,rss_limit_bytes=4*GIB,mps_limit_bytes=6*GIB,reserve_bytes=12*GIB,required_free_bytes=allocation['storage_total_bytes'],whole_comparison_planning_record='research/tokenizer-evidence/general-rsi-cross-size-boundary-001/allocation.json',retained_inference_cap_bytes=GIB,restart_cap_bytes=640*1024**2,metadata_cap_bytes=32*1024**2,transient_policy='One exact step30 prefix per new shape. Retire only after full load/replay and endpoint+review evidence is durable, under held worker lock, path/size/SHA audit, no other open consumer, intent/deleted fsynced journal. Failed prefix retained.',scientific_training=False,validation_scored=False,test_opened=False)
    assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
    OUT.mkdir();put(OUT/'plan.json',p);print(sha(OUT/'plan.json'))

class TrainScoreCase:
    def __init__(self,data):
        self.tokens=data.view.train_data[:259].clone()
        self.view=SimpleNamespace(generators=data.view.generators)
        self.indexes={'val':[dict(tokens=259,source='engineering_train')]}
    def validation_chunks(self):
        for offset in range(0,258,128):
            n=min(128,258-offset)
            yield 0,'engineering_train',self.tokens[offset:offset+n][None,:],self.tokens[offset+1:offset+n+1][None,:]

def numerical(ctx,p,book):
    r=p['roles'];arm=ctx.config['arm'];depth=ctx.config['depth']
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r['native_initializer'])
        cpu=DepthModel(Spec(arm,depth,tuple(p['extra_group_seeds']),None if arm=='native' else r['composer_initializer'],None if arm=='native' else book.identity),None if arm=='native' else book)
    with torch.no_grad():
        for (_,dst),(_,src) in zip(cpu.ordered_parameters(),ctx.model.ordered_parameters()):dst.copy_(src.detach().cpu())
    cpu.eval();ctx.model.eval();x=ctx.data.view.train_data[:16].clone()[None,:]
    with torch.no_grad(),FP32RMSNormInference():
        a=cpu.primary_logits(x).values;b=ctx.model.primary_logits(x.to('mps')).values.cpu()
        torch.testing.assert_close(b,a,atol=p['logits_atol'],rtol=p['logits_rtol'])
    case=TrainScoreCase(ctx.data);t=time.monotonic();measured=score(ctx.model,case);torch.mps.synchronize();seconds=time.monotonic()-t
    expected=[];inventory=[]
    with torch.no_grad(),FP32RMSNormInference():
        for _,_,xx,yy in case.validation_chunks():
            expected.extend(cpu_cross_entropy(cpu.primary_logits(xx).values[0],yy[0]).double().tolist())
            inventory.extend(yy[0].tolist())
    assert inventory==case.tokens[1:].tolist() and measured['count']==258
    expected_ce=math.fsum(expected)/258
    assert abs(measured['ce']-expected_ce)<=p['scoring_ce_atol']
    result=dict(max_logits_absolute_error=float((a-b).abs().max()),score_count=258,score_cpu_ce=expected_ce,score_mps_ce=measured['ce'],score_seconds=seconds,score_targets_exact=True)
    del cpu;gc.collect();torch.mps.empty_cache();return result

def journal(path,row):
    with path.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
def retire(receipt,review,jpath):
    import subprocess
    path=Path(receipt['path']);assert path.parent==RUN and not path.is_symlink()
    assert path.stat().st_size==receipt['size'] and sha(path)==receipt['sha256']
    # A nonzero lsof status with no output means no open descriptors for this exact file.
    result=subprocess.run(['/usr/sbin/lsof','-t','--',str(path)],capture_output=True,text=True)
    assert result.returncode==1 and not result.stdout.strip() and not result.stderr.strip(),result.stderr
    entry=dict(path=str(path),size=receipt['size'],sha256=receipt['sha256'],consumer_check='exact lsof returned1 with empty stdout/stderr; replay loaded into memory and endpoint/review durable',review_sha256=sha(review))
    journal(jpath,dict(event='intent',**entry));path.unlink();assert not path.exists();journal(jpath,dict(event='deleted',**entry))

def execute(p,binding,attempt):
    assert torch.backends.mps.is_available() and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
    assert torch.__version__==p['torch_version'] and platform.python_version()==p['python_version'];torch.set_num_threads(4)
    raw=[]
    for a in p['codebook_assets']:assert sha(a['path'])==a['sha256'];raw.append(Path(a['path']).read_bytes())
    book=build_codebook(*raw,'raw_bytes');RUN.mkdir();summaries=[];streams={};start=time.monotonic()
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    for index,s in enumerate(p['slots']):
        check_stop();assert time.monotonic()-start<p['max_total_seconds'];name=f"d{s['depth']}-{s['arm']}";out=attempt/name;out.mkdir()
        with CpuPrerequisiteGuard(out/'guard.json',ROOT,timeout_seconds=p['max_seconds_per_slot'],rss_limit_bytes=p['rss_limit_bytes'],reserve_bytes=p['reserve_bytes']):
            ctx=Context(s['arm'],s['depth'],p['roles'],p['extra_group_seeds'],book,binding)
            timing=[];peak=0;prefix=None
            def update(label):
                nonlocal peak
                check_stop();torch.mps.synchronize();t=time.monotonic();row=ctx.update();torch.mps.synchronize();elapsed=time.monotonic()-t
                peak=max(peak,torch.mps.driver_allocated_memory());assert peak<p['mps_limit_bytes']
                journal(out/(label+'.jsonl'),dict(**row,seconds=elapsed,mps_bytes=torch.mps.driver_allocated_memory()))
                return elapsed
            for _ in range(s['steps']):
                timing.append(update('updates'))
                if ctx.step==s['replay_from']:
                    payload=ctx.capture();initial=digest(payload);t=time.monotonic();prefix=write_checkpoint(RUN/(name+'-prefix30.pt'),payload,cap=p['restart_cap_bytes']);io=time.monotonic()-t
                    loaded=read_checkpoint(prefix);assert digest(loaded)==initial;del loaded,payload;gc.collect()
            stream=[r['batch_sha256'] for r in ctx.history]
            if s['depth'] in streams:assert stream==streams[s['depth']]
            else:streams[s['depth']]=stream
            if s['depth']>4:
                assert ctx.snapshots==2 and ctx.ema_start==30
                ref=ctx.capture();reference=digest(ref);del ref,ctx;gc.collect();torch.mps.empty_cache()
                payload=read_checkpoint(prefix)
                wrong=Context(s['arm'],4,p['roles'],p['extra_group_seeds'],book,binding);before=digest(wrong.capture())
                try:wrong.restore(payload)
                except AssertionError:pass
                else:raise AssertionError('Wrong depth accepted')
                assert before==digest(wrong.capture());del wrong;gc.collect();torch.mps.empty_cache()
                ctx=Context(s['arm'],s['depth'],p['roles'],p['extra_group_seeds'],book,binding);ctx.restore(payload);assert digest(ctx.capture())==initial;del payload;gc.collect()
                for _ in range(10):update('replay')
                assert digest(ctx.capture())==reference
            oracle=numerical(ctx,p,book)
            weights={n:v.detach().cpu().clone() for n,v in ctx.model.ordered_parameters()}
            endpoint=write_checkpoint(RUN/(name+'-raw.pt'),dict(identity=ctx.model.identity,config=ctx.config,parameters=weights),cap=192*1024**2);del weights
            summary=dict(slot=s,accepted=True,steps=s['steps'],replayed_updates=0 if s['depth']==4 else 10,full_restart_replay_exact=s['depth']>4,active_ema_restart=s['depth']>4,wrong_depth_rejected_before_mutation=s['depth']>4,synchronized_update_seconds=timing,steady_median_seconds=statistics.median(timing[5:20]),peak_mps_bytes=peak,oracle=oracle,endpoint=endpoint,transient_prefix=prefix,restart_write_seconds=None if prefix is None else io)
            review=out/'review.json';put(review,summary)
            del ctx;gc.collect();torch.mps.empty_cache()
            if prefix:retire(prefix,review,attempt/'deletion-journal.jsonl')
            retained=sum(f.stat().st_size for f in RUN.iterdir());assert retained<=p['retained_inference_cap_bytes']
            summaries.append(summary);print(json.dumps(dict(slot=index,name=name,accepted=True,median_seconds=summary['steady_median_seconds'])),flush=True)
    return dict(accepted=True,plan_sha256=binding,summaries=summaries,all_paired_batch_streams_exact=True,total_optimizer_updates=240,total_seconds=time.monotonic()-start,retained_bytes=sum(f.stat().st_size for f in RUN.iterdir()),scientific_training=False,validation_scored=False,test_opened=False)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);a=ap.parse_args()
    if a.action=='prepare':prepare();return
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop();p=read(OUT/'plan.json');binding=sha(OUT/'plan.json')
        assert all(sha(ROOT/n)==h for n,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
        attempt=OUT/'attempt-01';attempt.mkdir();put(attempt/'launch.json',dict(pid=os.getpid(),at=datetime.datetime.now(datetime.timezone.utc).isoformat(),binding=binding))
        def stop(*_):
            import research.general_rsi_pseudo_update_diagnostic_v1 as shared
            shared.STOP=True
        signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
        try:
            r=execute(p,binding,attempt);assert all(sha(ROOT/n)==h for n,h in p['sources'].items());put(attempt/'result.json',r)
        except BaseException:put(attempt/'failure.json',dict(traceback=traceback.format_exc(),binding=binding));raise
if __name__=='__main__':main()
