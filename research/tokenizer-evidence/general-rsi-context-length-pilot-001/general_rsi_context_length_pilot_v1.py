"""Frozen24-update paired new-length pilot and complete256 restart/replay."""
import fcntl,json,os,shutil,signal,time,traceback,gc
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop,digest
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-length-pilot-001'
RUN=ROOT/'research_runs/general-rsi-context-length-pilot-001'

def execute(p,binding,a):
    import torch
    from research.general_rsi_context_length_training_v1 import Context
    from research.general_rsi_training_v2 import write_checkpoint,read_checkpoint
    assert torch.__version__==p['torch_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4)
    RUN.mkdir();summaries=[];paired=None;peak=0
    def memory():
        nonlocal peak
        peak=max(peak,torch.mps.driver_allocated_memory());assert peak<6*2**30
    def log(name,row):
        with open(a/name,'a') as f:f.write(json.dumps(row)+'\n');f.flush();os.fsync(f.fileno())
    for length in (128,256):
        ctx=Context(length,p['roles'],binding)
        for _ in range(24):
            check_stop();row=ctx.update();assert row['lr_ratio']==1. and ctx.snapshots==0;log(f'c{length}-updates.jsonl',row);memory()
        hashes=[r['batch_sha256'] for r in ctx.history]
        if paired is None:paired=hashes
        else:assert paired==hashes
        summary=dict(context=length,updates=24,targets=24576,training_seconds=ctx.clock.spent_ns/1e9,mean_seconds=ctx.clock.spent_ns/24e9,first_loss=ctx.history[0]['loss'],last_loss=ctx.history[-1]['loss'],finite=True,peak_mps_bytes=peak)
        if length==128:
            # Complete state is needed only for the new256 restart test.
            weights={n:q.detach().cpu().clone() for n,q in ctx.model.ordered_parameters()}
            receipt=write_checkpoint(RUN/'c128-raw.pt',dict(identity=ctx.model.identity,config=ctx.config,parameters=weights),cap=128*2**20)
            summary['inference_receipt']=receipt;del weights
        else:
            tick=time.monotonic();payload=ctx.capture();receipt=write_checkpoint(RUN/'c256-step24.pt',payload);loaded=read_checkpoint(receipt);assert digest(loaded)==digest(payload);del loaded
            initial_digest=digest(payload);summary['restart_io_seconds']=time.monotonic()-tick;summary['restart_receipt']=receipt
            check_stop();row=ctx.update();memory();log('c256-reference-next.jsonl',row);duration=row['end_ns']-row['begin_ns']
            reference=ctx.capture();expected=digest(reference);refreceipt=write_checkpoint(RUN/'c256-step25-reference.pt',reference)
            summary['reference_receipt']=refreceipt;summary['reference_next_seconds']=duration/1e9
            del reference,payload
        put(a/f'c{length}-review.json',summary);summaries.append(summary)
        del ctx;gc.collect();torch.mps.empty_cache()
    # Wrong-context load rejects before parameters, optimizer, sampler or RNG mutate.
    payload=read_checkpoint(receipt)
    wrong=Context(128,p['roles'],binding);before=digest(wrong.capture())
    try:wrong.restore(payload)
    except AssertionError:pass
    else:raise AssertionError('Wrong context accepted')
    assert before==digest(wrong.capture());del wrong;gc.collect();torch.mps.empty_cache()
    ctx=Context(256,p['roles'],binding);ctx.restore(payload);assert digest(ctx.capture())==initial_digest
    del payload;check_stop();clock=iter([0,duration]);row=ctx.update(now_ns=lambda:next(clock));memory();log('c256-replayed-next.jsonl',row)
    assert digest(ctx.capture())==expected
    retained=sum(q.stat().st_size for q in RUN.iterdir());assert retained<=2*512*2**20+4*128*2**20
    return dict(plan_sha256=binding,accepted=True,summaries=summaries,paired_draws_exact=24,restart_roundtrip_exact=True,full_replay_exact=True,wrong_context_rejected_before_mutation=True,optimizer_updates=50,training_targets=51200,retained_bytes=retained,peak_mps_bytes=peak,quality_adopted=False,validation_opened=False,test_opened=False,scope='Fixed24-update execution pilot; no cooldown/EMA activation qualified. Live timer replay uses the measured reference duration.')

def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
        p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json');assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
        a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=binding))
        def stop(*_):
            import research.general_rsi_pseudo_update_diagnostic_v1 as shared
            shared.STOP=True
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
        try:
            with CpuPrerequisiteGuard(a/'guard.json',ROOT,timeout_seconds=p['max_seconds'],rss_limit_bytes=4*2**30,reserve_bytes=10*2**30):
                r=execute(p,binding,a);assert all(sha(ROOT/f)==h for f,h in p['sources'].items());put(a/'result.json',r);print(json.dumps(r),flush=True)
        except BaseException:
            put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
