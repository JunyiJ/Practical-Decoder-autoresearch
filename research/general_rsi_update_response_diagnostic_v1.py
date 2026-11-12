"""Frozen first-update displacement response; no persistent model payloads."""
import fcntl,gc,json,os,shutil,signal,time,traceback
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,digest,check_stop,displacement
from research.general_rsi_update_response_v1 import parameters,set_point,score,inputs
OUT=ROOT/'research/tokenizer-evidence/general-rsi-update-response-001'

def execute(plan,binding,a):
    import torch
    from research.general_rsi_continuation_decay_v1 import Context
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ['PYTORCH_ENABLE_MPS_FALLBACK']=='0';torch.set_num_threads(4)
    records=[];peak=0;matches=0;seconds=0.;scoring=[]
    def boundary():
        nonlocal peak
        check_stop();torch.mps.synchronize();peak=max(peak,torch.mps.driver_allocated_memory());assert peak<plan['mps_limit_bytes']
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<plan['metadata_cap_bytes']
    def measure(model,batches):
        t=time.monotonic();s=score(model,batches);torch.mps.synchronize();scoring.append(dict(targets=s['targets'],seconds=time.monotonic()-t));boundary();return s
    for c in plan['parents']:
        baseline=None;parent_hash=None;updates=None;probes=None
        for case in c['cases']:
            boundary();tick=time.monotonic();slot=c['parent']['slot'];i=case['case']
            ctx=Context('none',plan['budget_ns'],plan['half_life_ns'],case['roles'],case['pseudo_seed'],c['parent'],binding)
            assert ctx.step==0 and not ctx.history and not ctx.ema and ctx.snapshots==0 and ctx.clock.spent_ns==0
            assert all(not child.state for child in ctx.optimizer._optimizers) and ctx.optimizer.decay==(0.,0.)
            before=parameters(ctx.model);ph=digest(before)
            if baseline is None:
                inventory,updates,probes=inputs(c,ctx.data);assert inventory==c['inputs'];parent_hash=ph
                baseline=measure(ctx.model,probes)
                put(a/f'p{slot:02d}-initial.json',dict(parent=c['parent'],parameters_sha256=ph,evaluation=baseline))
            assert ph==parent_hash
            x,y=updates[i];batch=[(x[j:j+1],y[j:j+1]) for j in range(8)]
            batch_baseline=measure(ctx.model,batch)
            row=ctx.update();boundary()
            assert row['step']==1 and row['objective']=='ce' and row['lr_ratio']==1. and row['batch_sha256']==inventory['cases'][i]['sha256']
            assert ctx.step==1 and ctx.snapshots==0 and not ctx.ema
            seconds+=(row['end_ns']-row['begin_ns'])/1e9
            movement=displacement(ctx.model,{'parameters':before},None);after=parameters(ctx.model)
            point_records=[]
            for scale in plan['nonzero_scales']:
                boundary();matches+=set_point(ctx.model,before,after,scale)
                probe=measure(ctx.model,probes);batch_score=measure(ctx.model,batch)
                item=dict(parent=slot,case=i,scale=scale,probe=probe,batch=batch_score,probe_delta=probe['ce']-baseline['ce'],batch_delta=batch_score['ce']-batch_baseline['ce'])
                item['probe_secant']=item['probe_delta']/scale;item['batch_secant']=item['batch_delta']/scale
                put(a/f'p{slot:02d}-c{i:02d}-a{scale:g}.json',item);point_records.append(item)
            matches+=set_point(ctx.model,before,after,0.);assert digest(parameters(ctx.model))==parent_hash
            record=dict(parent=slot,case=i,pristine_optimizer_checked=True,initial_parameters_sha256=ph,update=row,displacement=movement,batch_initial=batch_baseline,exact_restoration=True,seconds=time.monotonic()-tick)
            put(a/f'p{slot:02d}-c{i:02d}-result.json',record)
            records.append(dict(parent=slot,case=i,points=[dict(scale=v['scale'],probe_delta=v['probe_delta'],batch_delta=v['batch_delta']) for v in point_records]))
            print(json.dumps(records[-1]),flush=True)
            del ctx,before,after;gc.collect();torch.mps.empty_cache()
    assert len(records)==24 and matches==24*4*88 and len(scoring)==171
    assert sum(s['targets'] for s in scoring)==712704
    return dict(plan_sha256=binding,records=records,updates=24,probe_evaluations=75,batch_evaluations=96,scored_targets=712704,parameter_copy_checks=matches,pristine_state_checks=24,training_seconds=seconds,score_timings=scoring,peak_mps_bytes=peak,test_opened=False,validation_scored=False,new_checkpoints=0)

def main():
    os.chdir(ROOT);tick=time.monotonic();lock=open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    p=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json');assert json.loads((OUT/'admission.json').read_text())['plan_sha256']==binding
    assert all(sha(ROOT/f)==h for f,h in p['sources'].items());assert shutil.disk_usage(ROOT).free>=p['required_free_bytes']
    a=OUT/'attempt-01';a.mkdir();put(a/'launch.json',dict(pid=os.getpid(),plan_sha256=binding,started_unix=time.time()))
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
