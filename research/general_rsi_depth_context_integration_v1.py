"""Bounded four-shape qualification of the selected two-depth context recipe."""
import argparse,datetime,fcntl,gc,json,os,platform,shutil,signal,statistics,subprocess,time,traceback
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop,digest
OUT=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-integration-001'
RUN=ROOT/'research_runs/general-rsi-depth-context-integration-001'
GIB=2**30

def read(p):return json.loads(p.read_text())
def journal(p,row):
    with p.open('a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
def host_memory():
    return {name:subprocess.check_output(cmd,text=True) for name,cmd in {
        'vm_stat':['/usr/bin/vm_stat'],
        'swap':['/usr/sbin/sysctl','vm.swapusage'],
        'pressure':['/usr/sbin/sysctl','kern.memorystatus_vm_pressure_level'],
        'physical':['/usr/sbin/sysctl','hw.memsize']}.items()}
def prepare():
    import torch
    fixture=OUT.parent/'general-rsi-depth-context-fixture-002'
    assert read(fixture/'attempt-01/result.json')['accepted']
    old=read(OUT.parent/'general-rsi-context-length-integration-001/plan.json')
    p0=read(fixture/'plan.json');sources=dict(p0['sources'])
    for f in [Path(__file__),ROOT/'research/general_rsi_depth_context_training_v1.py',fixture/'attempt-01/result.json',ROOT/'research/general_rsi_context_length_evaluation_v1.py',ROOT/'research/general_rsi_context_length_generation_v1.py',ROOT/'research/general_rsi_equal_time_training_v1.py',ROOT/'research/general_rsi_training_clock_v1.py',ROOT/'research/general_rsi_training_v2.py']:
        sources[str(f.relative_to(ROOT))]=sha(f)
    forecast='research/tokenizer-evidence/general-rsi-context-transfer-two-depth-planning-001/forecast.json'
    p=dict(kind='depth_context_integration_v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,torch_version=torch.__version__,python_version=platform.python_version(),roles=dict(native_initializer=62111415011001,composer_initializer=62111415011002,train_sampler=62111415011003,validation_sampler=62111415011004),extra_group_seeds=[62111415011005,62111415011006,62111415011007],fresh_role_audit='rg over research/research_state JSON,JSONL,Python returned no matches before preparation',slots=[dict(depth=d,context=c) for d in (4,16) for c in (128,256)],budget_ns=30000000000,half_life_ns=1000000000,checkpoint_after_ns=24000000000,max_updates=120,training_document=old['training_document'],generation_protocol=old['generation_protocol'],ce_atol=1e-5,logits_atol=3e-5,logits_rtol=3e-4,max_seconds_per_slot=600,max_total_seconds=2400,rss_limit_bytes=4*GIB,mps_limit_bytes=6*GIB,reserve_bytes=12*GIB,required_free_bytes=read(ROOT/forecast)['total_required_free_bytes'],whole_comparison_planning_record=forecast,restart_cap_bytes=640*2**20,retained_inference_cap_bytes=GIB,metadata_cap_bytes=32*2**20,criteria='All four shapes: finite1024-target timed updates, active time-EMA complete state and exact one-step replay with recorded duration, wrong-depth/context rejection before mutation, RAW/EMA exact endpoint loading, CPU/MPS corrected logits at full native context within3e-5+3e-4 relative, all654-token training-document targets and FP64 CE within1e-5,24 fixed first generations, phase memory and synchronized timing. No new scientific quality gate.',transient_policy='One prefix and one next-step reference per shape. Retire exact path/size/hash receipts only after accepted replay, endpoints and durable review, shared lock and no open consumer; journal intent/deleted. Preserve failed payloads.',scientific_training=False,validation_opened=False,test_opened=False)
    assert all(sha(ROOT/f)==h for f,h in sources.items())
    assert shutil.disk_usage(ROOT).free>=p['required_free_bytes'];OUT.mkdir();put(OUT/'plan.json',p);print(sha(OUT/'plan.json'))
def retire(receipt,review,log):
    path=Path(receipt['path']);assert path.parent==RUN and not path.is_symlink()
    assert path.stat().st_size==receipt['size'] and sha(path)==receipt['sha256']
    r=subprocess.run(['/usr/sbin/lsof','-t','--',str(path)],capture_output=True,text=True)
    assert r.returncode==1 and not r.stdout.strip() and not r.stderr.strip(),r
    row=dict(path=str(path),size=receipt['size'],sha256=receipt['sha256'],review_sha256=sha(review),consumer_check='Replay complete, endpoint/review durable, exact lsof empty with status1')
    journal(log,dict(event='intent',**row));path.unlink();journal(log,dict(event='deleted',**row))
def shape(p,binding,a,s):
    import torch
    from research.general_rsi_depth_context_training_v1 import Context,load_inference
    from research.general_rsi_depth_context_v1.model import Model
    from research.general_rsi_depth_context_v1.contract import Spec
    from research.general_rsi_context_length_evaluation_v1 import evaluate
    from research.general_rsi_context_length_generation_v1 import generate
    from research.general_rsi_training_v2 import write_checkpoint,read_checkpoint
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    d=s['depth'];c=s['context'];name=f'd{d}-c{c}';peak=0;start=time.monotonic()
    def memory(phase):
        nonlocal peak
        torch.mps.synchronize();m=torch.mps.driver_allocated_memory();peak=max(peak,m)
        assert m<p['mps_limit_bytes']
        journal(a/'memory.jsonl',dict(phase=phase,mps_driver_bytes=m,mps_current_bytes=torch.mps.current_allocated_memory(),host=host_memory()))
    def make():return Context(d,c,p['budget_ns'],p['half_life_ns'],p['roles'],p['extra_group_seeds'],binding)
    memory('before_initialization');ctx=make();memory('initialized');timings=[]
    while ctx.clock.spent_ns<p['checkpoint_after_ns'] or ctx.snapshots<2:
        check_stop();assert ctx.step<p['max_updates'] and ctx.clock.spent_ns<ctx.clock.budget_ns
        t=time.monotonic();row=ctx.update();timings.append(time.monotonic()-t);journal(a/'updates.jsonl',row)
        memory('training_ema' if ctx.snapshots else 'training')
    assert ctx.history[-1]['lr_ratio']<1 and ctx.snapshots>=2
    t=time.monotonic();payload=ctx.capture();before=digest(payload);memory('captured_restart')
    receipt=write_checkpoint(RUN/f'{name}-prefix.pt',payload,cap=p['restart_cap_bytes'])
    loaded=read_checkpoint(receipt);assert digest(loaded)==before;memory('restart_readback_overlap');del loaded,payload
    restart_seconds=time.monotonic()-t;step=ctx.step;snaps=ctx.snapshots
    refrow=ctx.update();journal(a/'reference-next.jsonl',refrow);duration=refrow['end_ns']-refrow['begin_ns']
    reference=ctx.capture();refdigest=digest(reference);refreceipt=write_checkpoint(RUN/f'{name}-reference.pt',reference,cap=p['restart_cap_bytes']);memory('atomic_predecessor_overlap');del reference,ctx;gc.collect();torch.mps.empty_cache()
    payload=read_checkpoint(receipt)
    # Identity rejection is pre-copy, checked without allocating a wrong full model.
    ctx=make();pristine=digest(ctx.capture())
    for key,value in [('depth',16 if d==4 else 4),('context',256 if c==128 else 128)]:
        bad=dict(payload,config=dict(payload['config'],**{key:value}))
        try:ctx.restore(bad)
        except AssertionError:pass
        else:raise AssertionError('Wrong shape restore accepted')
        assert digest(ctx.capture())==pristine
    del bad;ctx.restore(payload);assert digest(ctx.capture())==before;del payload;gc.collect();memory('restored')
    ticks=iter([0,duration]);row=ctx.update(now_ns=lambda:next(ticks));journal(a/'replayed-next.jsonl',row)
    assert digest(ctx.capture())==refdigest;memory('replayed')
    receipts={endpoint:ctx.export(RUN/f'{name}-{endpoint}.pt',endpoint) for endpoint in ('raw','time_ema')};memory('exported_endpoints')
    before=digest(ctx.capture())
    for dd,cc in ((16 if d==4 else 4,c),(d,256 if c==128 else 128)):
        try:load_inference(receipts['raw'],dd,cc,binding)
        except AssertionError:pass
        else:raise AssertionError('Wrong-shape inference accepted')
    for endpoint in ('raw','time_ema'):
        model=load_inference(receipts[endpoint],d,c,binding);source=dict(ctx.model.ordered_parameters()) if endpoint=='raw' else ctx.ema
        assert all(torch.equal(q,source[n]) for n,q in model.ordered_parameters());memory('loaded_'+endpoint)
        if endpoint=='raw':del model;gc.collect();torch.mps.empty_cache()
    assert digest(ctx.capture())==before
    values=ctx.data.view.train_data;document=p['training_document'];t=time.monotonic();scoring={}
    for cc in sorted(set([128,c])):
        result=evaluate(model,values,[document],cc);put(a/f'train-c{cc}.json',result);scoring[cc]=result;memory('scoring_'+str(cc))
    expected=list(range(document['offset']+1,document['offset']+document['tokens']))
    for result in scoring.values():assert [i for r in result['rows'] for i in r['target_ids']]==expected
    model.eval();maxerr=0.
    with torch.no_grad(),FP32RMSNormInference():
        for cc,result in scoring.items():
            refs=[]
            for offset in range(0,document['tokens']-1,cc):
                startpos=document['offset']+offset;end=min(startpos+cc,document['offset']+document['tokens']-1)
                z=model.primary_logits(values[startpos:end][None,:].to('mps')).values[0].cpu().double();targets=values[startpos+1:end+1]
                refs.extend((torch.logsumexp(z,-1)-z.gather(1,targets[:,None]).squeeze(1)).tolist())
            actual=[v for r in result['rows'] for v in r['token_losses']];err=max(abs(x-y) for x,y in zip(refs,actual));maxerr=max(maxerr,err);assert err<=p['ce_atol']
    eval_seconds=time.monotonic()-t;del z
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(p['roles']['native_initializer']);cpu=Model(Spec(d,c,tuple(p['extra_group_seeds'])))
    with torch.no_grad():
        for (_,dst),(_,src) in zip(cpu.ordered_parameters(),model.ordered_parameters()):dst.copy_(src.detach().cpu())
    cpu.eval();x=values[:c][None,:]
    with torch.no_grad(),FP32RMSNormInference():
        aa=cpu.primary_logits(x).values;bb=model.primary_logits(x.to('mps')).values.cpu()
        torch.testing.assert_close(bb,aa,atol=p['logits_atol'],rtol=p['logits_rtol'])
        logits_error=float((aa-bb).abs().max())
    memory('cpu_mps_full_context_oracle');del aa,bb,cpu;gc.collect();torch.mps.empty_cache()
    model.train();t=time.monotonic();modelbefore=digest(dict(model.ordered_parameters()))
    rows=generate(model,p['generation_protocol'],a/'generations',receipts['time_ema'],binding);gen_seconds=time.monotonic()-t;memory('generation')
    assert len(rows)==24 and digest(dict(model.ordered_parameters()))==modelbefore and digest(ctx.capture())==before
    memory('final_capture')
    result=dict(accepted=True,shape=s,checkpoint_step=step,checkpoint_snapshots=snaps,final_snapshots=ctx.snapshots,full_active_time_ema_restart_replay_exact=True,wrong_shape_restore_and_endpoint_rejected=True,endpoint_parameters_exact=True,scoring_targets_exact=True,ce_max_absolute_error=maxerr,logits_max_absolute_error=logits_error,state_unchanged_after_scoring_generation=True,first_outputs=len(rows),generated_tokens=sum(len(r['generated_token_ids']) for r in rows),generation_seconds=gen_seconds,evaluation_and_oracle_seconds=eval_seconds,restart_io_seconds=restart_seconds,synchronized_update_seconds=timings,steady_median_seconds=statistics.median(timings[5:]),peak_mps_bytes=peak,restart_receipts=[receipt,refreceipt],endpoint_receipts=receipts,worker_seconds=time.monotonic()-start,batch_stream=[r['batch_sha256'] for r in ctx.history],validation_opened=False,test_opened=False)
    del ctx,model,source;gc.collect();torch.mps.empty_cache();return result

def execute(p,binding,a):
    import torch
    assert torch.__version__==p['torch_version'] and platform.python_version()==p['python_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0'
    torch.set_num_threads(4);RUN.mkdir();start=time.monotonic();summaries=[]
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    for s in p['slots']:
        check_stop();assert time.monotonic()-start<p['max_total_seconds'];out=a/f"d{s['depth']}-c{s['context']}";out.mkdir()
        with CpuPrerequisiteGuard(out/'guard.json',ROOT,timeout_seconds=p['max_seconds_per_slot'],rss_limit_bytes=p['rss_limit_bytes'],reserve_bytes=p['reserve_bytes']):
            result=shape(p,binding,out,s);put(out/'review.json',result)
            for receipt in result['restart_receipts']:retire(receipt,out/'review.json',a/'deletion-journal.jsonl')
            assert sum(f.stat().st_size for f in RUN.iterdir())<=p['retained_inference_cap_bytes']
        summaries.append(result);print(json.dumps(dict(shape=s,accepted=True,seconds=result['worker_seconds'],peak_mps_bytes=result['peak_mps_bytes'])),flush=True)
    minimum=min(len(r['batch_stream']) for r in summaries)
    assert all(r['batch_stream'][:minimum]==summaries[0]['batch_stream'][:minimum] for r in summaries)
    assert sum(q.stat().st_size for q in OUT.rglob('*') if q.is_file())<=p['metadata_cap_bytes']
    return dict(accepted=True,plan_sha256=binding,summaries=summaries,paired_prefix_batch_streams_exact=True,total_seconds=time.monotonic()-start,retained_bytes=sum(f.stat().st_size for f in RUN.iterdir()),scientific_training=False,validation_opened=False,test_opened=False)
def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','run']);args=ap.parse_args()
    if args.action=='prepare':prepare();return
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop();p=read(OUT/'plan.json');binding=sha(OUT/'plan.json')
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
        a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),plan_sha256=binding,at=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=host_memory()))
        def stop(*_):
            import research.general_rsi_pseudo_update_diagnostic_v1 as shared
            shared.STOP=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        try:
            result=execute(p,binding,a);assert all(sha(ROOT/f)==h for f,h in p['sources'].items());put(a/'result.json',result)
        except BaseException:put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
