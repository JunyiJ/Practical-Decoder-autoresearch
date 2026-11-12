"""Fresh paired one/eight independent-batch first-update diagnostic."""
import fcntl,gc,json,os,shutil,signal,time,traceback
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,digest,check_stop,displacement
from research.general_rsi_update_response_v1 import parameters,score
from research.general_rsi_batch_averaging_v1 import inputs,update_once
OUT=ROOT/'research/tokenizer-evidence/general-rsi-batch-averaging-001'

def execute(p,binding,a):
    import torch
    from research.general_rsi_continuation_decay_v1 import Context
    from research.general_rsi_qualification_v1 import equal
    assert torch.__version__==p['torch_version'] and os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4)
    peak=0;records=[];score_times=[];training=0.;passes=0
    def boundary():
        nonlocal peak
        check_stop();torch.mps.synchronize();peak=max(peak,torch.mps.driver_allocated_memory());assert peak<p['mps_limit_bytes']
        assert sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file())<p['metadata_cap_bytes']
    def evaluate(model,batches):
        start=time.monotonic();v=score(model,batches);torch.mps.synchronize();score_times.append(time.monotonic()-start);boundary();return v
    for c in p['parents']:
        baseline=None;parent_hash=None;probes=None
        for case in c['cases']:
            common=None;i=case['case'];slot=c['parent']['slot']
            for count in case['order']:
                boundary();tick=time.monotonic();ctx=Context('none',p['budget_ns'],p['half_life_ns'],case['roles'],case['pseudo_seed'],c['parent'],binding)
                assert ctx.step==0 and not ctx.history and not ctx.ema and ctx.snapshots==0 and ctx.clock.spent_ns==0 and all(not child.state for child in ctx.optimizer._optimizers)
                if common is None:common=ctx.capture()
                torch.random.set_rng_state(common['cpu_rng']);torch.mps.set_rng_state(common['mps_rng']);equal(ctx.capture(),common)
                ph=digest(common['parameters'])
                if baseline is None:
                    inventory,probes=inputs(c,ctx.data);assert inventory==c['inputs'];parent_hash=ph;baseline=evaluate(ctx.model,probes)
                    put(a/f'p{slot:02d}-initial.json',dict(parent=c['parent'],parameters_sha256=ph,evaluation=baseline))
                assert ph==parent_hash
                row=update_once(ctx,count);boundary();expected=[v['sha256'] for v in c['inputs']['cases'][i]['batches'][:count]]
                assert row['microbatch_sha256']==expected and row['valid_loss_targets']==1024*count and ctx.pseudo.draws==0
                movement=displacement(ctx.model,common,None);s=evaluate(ctx.model,probes)
                training+=(row['end_ns']-row['begin_ns'])/1e9;passes+=count
                record=dict(parent=slot,case=i,microbatches=count,initial_state_exact=True,parameters_sha256=ph,update=row,evaluation=s,delta_from_parent=s['ce']-baseline['ce'],displacement=movement,seconds=time.monotonic()-tick)
                put(a/f'p{slot:02d}-c{i:02d}-k{count}.json',record);records.append(dict(parent=slot,case=i,microbatches=count,delta_from_parent=record['delta_from_parent']))
                print(json.dumps(records[-1]),flush=True);del ctx;gc.collect();torch.mps.empty_cache()
            del common
    assert len(records)==48 and passes==216 and len(score_times)==51
    return dict(plan_sha256=binding,records=records,updates=48,forward_backward_passes=passes,training_targets=passes*1024,evaluations=51,scored_targets=51*8192,initial_state_checks=48,training_seconds=training,score_seconds=score_times,peak_mps_bytes=peak,test_opened=False,validation_scored=False,new_checkpoints=0)

def main():
    lock=open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop();tick=time.monotonic()
    p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json');assert json.loads((OUT/'admission.json').read_text())['plan_sha256']==binding
    assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
    a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=binding))
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(a/'guard.json',ROOT,timeout_seconds=p['total_seconds'],rss_limit_bytes=p['rss_limit_bytes'],reserve_bytes=p['reserve_bytes']):
            result=execute(p,binding,a);assert all(sha(ROOT/f)==h for f,h in p['sources'].items());result['total_seconds']=time.monotonic()-tick;put(a/'result.json',result)
    except BaseException:
        put(a/'failure.json',dict(traceback=traceback.format_exc(),plan_sha256=binding));raise
if __name__=='__main__':main()
