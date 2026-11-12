"""Fresh paired joint-gradient mechanism diagnostic, training split only."""
import fcntl,gc,json,math,os,shutil,signal,time,traceback
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,digest,batch_hash,ce_batch,evaluate,check_stop
from research.general_rsi_pseudo_accumulated_v1 import append
OUT=ROOT/'research/tokenizer-evidence/general-rsi-joint-diagnostic-001'
ARMS=['ce_full','ce_half','ce_pseudo_mean']


def inputs(c,data):
    import torch
    from research.general_rsi_pseudo_spans_v1 import Sampler
    values=data.view.train_data;rng=torch.Generator().manual_seed(c['roles']['train_sampler'])
    pseudo=Sampler(c['pseudo_seed']);natural=[];synthetic=[];excluded=[]
    for n in range(1,101):
        starts=torch.randint(len(values)-512,(2,),generator=rng).tolist()
        natural.append(dict(n=n,starts=starts,sha256=batch_hash(*ce_batch(values,starts))))
        excluded.extend((s,s+513) for s in starts)
        x,y,previous,mask,rows=pseudo.batch()
        synthetic.append(dict(n=n,sha256=batch_hash(x,y,previous,mask),rows=rows))
        for row in rows:
            _,start,length,prefix=map(int,pseudo.spans[row['inventory_index']]);excluded.append((start-prefix,start+length))
    rng=torch.Generator().manual_seed(c['evaluation_seed']);eligible=[i for i,r in enumerate(data.indexes['train']) if r['tokens']>=129]
    windows=[];proposals=0
    while len(windows)<64:
        proposals+=1;assert proposals<10000
        doc=eligible[int(torch.randint(len(eligible),(),generator=rng))];d=data.indexes['train'][doc]
        start=d['offset']+int(torch.randint(d['tokens']-128,(),generator=rng))
        if any(start<b and a<start+129 for a,b in excluded):continue
        windows.append(dict(document=doc,source=d['source'],start=start,targets=128));excluded.append((start,start+129))
    natural_eval=[(values[w['start']:w['start']+128][None,:],values[w['start']+1:w['start']+129][None,:]) for w in windows]
    probe=Sampler(c['repetition_seed']);rep=[];rep_inv=[];rp=0
    while len(rep)<8:
        rp+=1;assert rp<10000
        batch=probe.batch();intervals=[];valid=True
        for row in batch[-1]:
            _,start,length,prefix=map(int,probe.spans[row['inventory_index']]);a,b=start-prefix,start+length
            if any(a<d and s<b for s,d in excluded+intervals):valid=False;break
            intervals.append((a,b))
        if not valid:continue
        excluded.extend(intervals);rep.append(batch)
        rep_inv.append(dict(proposal=rp,sha256=batch_hash(*batch[:4]),rows=batch[-1],source_intervals=intervals))
    inventory=dict(natural=natural,synthetic=synthetic,evaluation_windows=windows,natural_proposals=proposals,
        evaluation_sha256=batch_hash(*(t for pair in natural_eval for t in pair)),repetition_batches=rep_inv,
        repetition_proposals=rp,disjoint_probe_source_positions=True)
    # JSON round-trip makes interval list types identical at preparation/execution.
    return json.loads(json.dumps(inventory)),natural_eval,rep


def probability_stats(probabilities,previous,mask):
    import torch
    anchor=probabilities.gather(1,previous);p=probabilities[mask];a=anchor[mask]
    assert bool(torch.isfinite(p).all()) and bool(torch.isfinite(a).all()) and p.numel()>0
    difference=(p-.5*a).abs();eps=torch.finfo(probabilities.dtype).eps
    loss=-torch.log1p(-difference.clamp(max=1-eps))
    return dict(targets=p.numel(),probability_sum=math.fsum(p.cpu().double().tolist()),
        previous_probability_sum=math.fsum(a.cpu().double().tolist()),penalty_sum=math.fsum(loss.cpu().double().tolist()),
        saturated=int((difference>1-eps).sum()))


def repetition_score(model,batches):
    import torch
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    prior=model.training;versions=tuple(p._version for _,p in model.ordered_parameters());rows=[]
    try:
        model.eval()
        with torch.no_grad(),FP32RMSNormInference():
            for x,y,previous,mask,_ in batches:
                check_stop();x,y,previous,mask=[t.to('mps') for t in (x,y,previous,mask)]
                logits=model.primary_logits(x).values
                probabilities=logits.log_softmax(-1).gather(-1,y[...,None]).squeeze(-1).exp()
                rows.append(probability_stats(probabilities,previous,mask))
    finally:model.train(prior)
    assert versions==tuple(p._version for _,p in model.ordered_parameters())
    totals={k:math.fsum(r[k] for r in rows) for k in rows[0]};n=int(totals['targets']);a=totals['previous_probability_sum'];assert a>0
    return dict(targets=n,mean_probability=totals['probability_sum']/n,mean_previous_probability=a/n,
        probability_ratio=totals['probability_sum']/a,penalty=totals['penalty_sum']/n,saturated=int(totals['saturated']),batches=rows)


def scorer_oracle():
    import torch
    p=torch.tensor([[.2,.4,.6]],dtype=torch.float64);prev=torch.tensor([[0,0,1]]);mask=torch.tensor([[False,True,True]])
    s=probability_stats(p,prev,mask)
    assert s['targets']==2 and abs(s['probability_sum']-1.)<1e-14 and abs(s['previous_probability_sum']-.6)<1e-14
    assert abs(s['penalty_sum']-(-math.log(.7)-math.log(.6)))<1e-14 and s['saturated']==0
    return s


def execute(plan,binding,attempt):
    import torch
    from research.general_rsi_joint_training_v1 import Context
    from research.general_rsi_qualification_v1 import equal
    assert torch.__version__==plan['torch_version'] and torch.backends.mps.is_available()
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0';torch.set_num_threads(4)
    put(attempt/'new-scorer-oracle.json',dict(passed=True,statistics=scorer_oracle(),scope='New aggregation only; existing model and repetition objective reused.'))
    peak=0;updates=0;seconds=0.;endpoints=[]
    def boundary():
        nonlocal peak
        check_stop();torch.mps.synchronize();peak=max(peak,torch.mps.driver_allocated_memory());assert peak<plan['mps_limit_bytes']
        assert sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file())<plan['metadata_cap_bytes']
    for c in plan['parents']:
        slot=c['parent']['slot'];common=None;natural_eval=None;rep=None
        for arm in c['arm_order']:
            boundary();tick=time.monotonic()
            ctx=Context(arm,plan['clock_budget_ns'],plan['half_life_ns'],c['roles'],c['pseudo_seed'],c['parent'],binding)
            if common is None:
                inventory,natural_eval,rep=inputs(c,ctx.data);assert inventory==c['inputs']
                common=ctx.capture();baseline=dict(natural=evaluate(ctx.model,natural_eval),repetition=repetition_score(ctx.model,rep))
                put(attempt/f'p{slot:02d}-initial.json',dict(parent=c['parent'],scores=baseline))
            torch.random.set_rng_state(common['cpu_rng']);torch.mps.set_rng_state(common['mps_rng'])
            initial=ctx.capture();equal({k:v for k,v in initial.items() if k!='config'},{k:v for k,v in common.items() if k!='config'})
            del initial;arm_seconds=0.;probes=[]
            for n in range(1,101):
                boundary();row=ctx.update();assert row['batch_sha256']==inventory['natural'][n-1]['sha256']
                if arm=='ce_pseudo_mean':
                    assert row['pseudo_sha256']==inventory['synthetic'][n-1]['sha256'] and row['pseudo_rows']==inventory['synthetic'][n-1]['rows']
                assert row['lr_ratio']==1. and ctx.snapshots==0 and ctx.step==n
                updates+=1;duration=(row['end_ns']-row['begin_ns'])/1e9;seconds+=duration;arm_seconds+=duration
                append(attempt/f'p{slot:02d}-{arm}-steps.jsonl',row)
                if n in plan['probes']:
                    scores=dict(natural=evaluate(ctx.model,natural_eval),repetition=repetition_score(ctx.model,rep))
                    probe=dict(parent_slot=slot,arm=arm,natural_updates=n,optimizer_steps=ctx.step,scores=scores,
                        delta_ce_from_parent=scores['natural']['ce']-baseline['natural']['ce'])
                    put(attempt/f'p{slot:02d}-{arm}-n{n:03d}.json',probe);probes.append(probe)
                    print(json.dumps(dict(parent=slot,arm=arm,n=n,ce=scores['natural']['ce'],repeated_probability=scores['repetition']['mean_probability'])),flush=True)
            captured=ctx.capture();assert captured['component_counts']==dict(ce=100,pseudo=100 if arm=='ce_pseudo_mean' else 0)
            result=dict(parent_slot=slot,arm=arm,steps=100,component_counts=captured['component_counts'],
                final_state_sha256=digest(captured),synchronized_update_seconds=arm_seconds,total_seconds=time.monotonic()-tick,
                final_scores=probes[-1]['scores'],pristine_initial_state_equal=True)
            put(attempt/f'p{slot:02d}-{arm}-result.json',result);endpoints.append(result)
            del captured,ctx;gc.collect();torch.mps.empty_cache()
        del common,natural_eval,rep;gc.collect();torch.mps.empty_cache()
    assert updates==900 and len(endpoints)==9
    return dict(plan_sha256=binding,updates=updates,natural_components=900,pseudo_components=300,endpoints=endpoints,
        synchronized_update_seconds=seconds,peak_mps_boundary_bytes=peak,test_opened=False,validation_scored=False,
        new_saved_models=0,initial_state_comparisons=9)


def main():
    tick=time.monotonic();os.chdir(ROOT)
    lock=(ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop()
    plan=json.loads((OUT/'plan.json').read_text());binding=sha(OUT/'plan.json')
    assert shutil.disk_usage(ROOT).free>=plan['required_free_bytes']
    attempt=OUT/'attempt-01';attempt.mkdir(exist_ok=False)
    def stop(*_):
        import research.general_rsi_pseudo_update_diagnostic_v1 as shared
        shared.STOP=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    put(attempt/'launch.json',dict(pid=os.getpid(),started_unix=time.time(),plan_sha256=binding))
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    try:
        with CpuPrerequisiteGuard(attempt/'guard.json',ROOT,timeout_seconds=plan['timeout_seconds'],rss_limit_bytes=plan['rss_limit_bytes'],reserve_bytes=plan['reserve_bytes']):
            for p,h in plan['sources'].items():assert sha(p)==h,p
            assert json.loads((OUT/'admission.json').read_text())['plan_sha256']==binding
            result=execute(plan,binding,attempt);result['total_seconds']=time.monotonic()-tick;put(attempt/'result.json',result)
    except BaseException:
        put(attempt/'failure.json',dict(traceback=traceback.format_exc(),seconds=time.monotonic()-tick));raise

if __name__=='__main__':main()
