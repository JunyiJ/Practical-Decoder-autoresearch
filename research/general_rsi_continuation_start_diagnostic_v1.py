"""Fresh paired RAW/time-EMA start comparison, training-only, no new weights."""
import fcntl,gc,json,os,shutil,signal,time,traceback
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,digest,evaluate,check_stop
from research.general_rsi_pseudo_accumulated_v1 import append
from research.general_rsi_continuation_decay_diagnostic_v1 import inputs
OUT=ROOT/'research/tokenizer-evidence/general-rsi-continuation-start-diagnostic-001'

def execute(plan,binding,attempt):
    import torch
    from research.general_rsi_continuation_start_v1 import Context
    from research.general_rsi_qualification_v1 import equal
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4);peak=0;updates=0;seconds=0.;reports=[];loads=[]
    def boundary():
        nonlocal peak
        check_stop();torch.mps.synchronize();peak=max(peak,torch.mps.driver_allocated_memory());assert peak<plan['mps_limit_bytes']
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['metadata_cap_bytes']
    for c in plan['parents']:
        common=None;batches=None
        for endpoint in c['order']:
            boundary();tick=time.monotonic();ctx=Context(endpoint,c['starting_weights'][endpoint],plan['budget_ns'],plan['half_life_ns'],c['roles'],c['pseudo_seed'],c['parent'],binding)
            loaded=time.monotonic()-tick;loads.append(loaded)
            put(attempt/f"p{c['parent']['slot']:02d}-{endpoint}-loader.json",dict(**ctx.start_loader_check,seconds=loaded))
            if common is None:
                inventory,batches=inputs(c,ctx.data);assert inventory==c['inputs'];common=ctx.capture()
            torch.random.set_rng_state(common['cpu_rng']);torch.mps.set_rng_state(common['mps_rng']);initial=ctx.capture()
            equal({k:v for k,v in initial.items() if k not in ['config','parameters']},{k:v for k,v in common.items() if k not in ['config','parameters']})
            initial_hash=digest(initial['parameters']);del initial
            baseline=evaluate(ctx.model,batches)
            put(attempt/f"p{c['parent']['slot']:02d}-{endpoint}-initial.json",dict(evaluation=baseline,parameters_sha256=initial_hash,other_initial_state_exact=True))
            arm_ns=0;probes=[]
            for n in range(1,101):
                boundary();row=ctx.update();assert row['batch_sha256']==inventory['natural'][n-1]['sha256'] and row['objective']=='ce'
                assert row['lr_ratio']==1. and ctx.snapshots==0 and ctx.step==n and ctx.optimizer.decay==(0.,0.)
                duration=row['end_ns']-row['begin_ns'];arm_ns+=duration;updates+=1;seconds+=duration/1e9
                append(attempt/f"p{c['parent']['slot']:02d}-{endpoint}-steps.jsonl",row)
                if n in plan['probes']:
                    score=evaluate(ctx.model,batches);probe=dict(parent=c['parent']['slot'],endpoint=endpoint,n=n,evaluation=score,delta_from_own_start=score['ce']-baseline['ce'])
                    put(attempt/f"p{c['parent']['slot']:02d}-{endpoint}-n{n:03d}.json",probe);probes.append(probe)
                    print(json.dumps({k:v for k,v in probe.items() if k!='evaluation'}|{'ce':score['ce']}),flush=True)
            state=ctx.capture();report=dict(parent=c['parent']['slot'],endpoint=endpoint,steps=100,initial_nonweight_state_exact=True,
                final_state_sha256=digest(state),initial_evaluation=baseline,final_evaluation=probes[-1]['evaluation'],delta_from_own_start=probes[-1]['delta_from_own_start'],
                clipped=sum(r['clip']['coefficient']<1 for r in state['history']),training_seconds=arm_ns/1e9,total_seconds=time.monotonic()-tick)
            put(attempt/f"p{c['parent']['slot']:02d}-{endpoint}-result.json",report);reports.append(report)
            del state,ctx;gc.collect();torch.mps.empty_cache()
        del common,batches;gc.collect();torch.mps.empty_cache()
    assert updates==600 and len(reports)==6
    return dict(plan_sha256=binding,reports=reports,updates=updates,evaluations=36,training_seconds=seconds,peak_mps_bytes=peak,initial_nonweight_state_comparisons=6,
        load_seconds=loads,test_opened=False,validation_scored=False,new_checkpoints=0)

def main():
    os.chdir(ROOT);tick=time.monotonic();lock=open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    plan=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json')
    assert json.loads((OUT/'admission.json').read_text())['plan_sha256']==binding
    assert all(sha(ROOT/p)==h for p,h in plan['sources'].items());assert shutil.disk_usage(ROOT).free>=plan['required_free_bytes']
    attempt=OUT/'attempt-01';attempt.mkdir();put(attempt/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=binding))
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,timeout_seconds=plan['total_seconds'],rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
            result=execute(plan,binding,attempt);assert all(sha(ROOT/p)==h for p,h in plan['sources'].items());result['total_seconds']=time.monotonic()-tick;put(attempt/'result.json',result)
    except BaseException:
        put(attempt/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
