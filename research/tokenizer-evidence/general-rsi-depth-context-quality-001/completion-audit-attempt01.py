"""Audit the completed frozen panel and summarize saved secondary evidence only."""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import threading
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT/'research/tokenizer-evidence/general-rsi-depth-context-quality-001'
RESERVE = 12*2**30

def read(p):
    return json.loads(p.read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def put(p, v):
    with p.open('x') as f: json.dump(v,f,indent=2);f.write('\n')

def main():
    started=time.perf_counter(); stop=threading.Event()
    # Saved JSON, hashes and small NumPy arrays only; no model or accelerator allocation.
    def limits():
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < 4*2**30 and shutil.disk_usage(ROOT).free >= RESERVE
    def guard():
        while not stop.wait(.25):
            if not limits(): os._exit(75)
    assert limits()
    threading.Thread(target=guard,daemon=True).start()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(P/'plan.json');binding=sha(P/'plan.json');saved=read(P/'decision.json')
        assert saved['binding']==binding
        assert all(sha(ROOT/n)==h for n,h in plan['sources'].items())
        for n,h in saved['evidence_sha256'].items():assert sha(ROOT/n)==h
        inputs={}; scores={};results={};rows=[];curves=[];weight_bytes=0
        flops=read(P/'flops.json');fp={(r['depth'],r['context']):r['estimated_accounted_flops_per_update'] for r in flops['rows']}
        def load(path):
            inputs[str(path.relative_to(ROOT))]=sha(path);return read(path)
        for s in plan['slots']:
            out=P/f"slot-{s['slot']:02d}";r=load(out/'result.json');v=load(out/'review.json');g=load(out/'guard-01.json')
            assert r['slot']==s and r['binding']==v['binding']==binding
            assert v['accepted_execution'] and v['result_sha256']==sha(out/'result.json')
            assert v['finite_cpu_endpoints']==2 and v['first_outputs']==24 and v['source_bindings_unchanged']==len(plan['sources'])
            assert v['schedule_and_ema_count_verified'] and v['exact_transient_retirements']==121
            assert not r['recovery_charges'] and g['status']=='finished' and g['reason'] is None
            assert r['training_ns']>=s['budget_ns'] and v['training_ns']==r['training_ns']
            for e in r['endpoints']:
                w=e['weights'];assert Path(w['path']).stat().st_size==w['size'] and sha(w['path'])==w['sha256'];weight_bytes+=w['size']
                for evaluation in ['native','common128']:
                    name=f"{e['endpoint']}-validation.json" if evaluation=='native' or s['context']==128 else f"{e['endpoint']}-common128-validation.json"
                    a=load(out/name);scores[s['depth'],s['bundle'],s['context'],e['endpoint'],evaluation]={k:a[k] for k in ['ce','count','loss_sum','sources','documents']}
                    assert a['count']==100416 and len(a['documents'])==108
            results[s['depth'],s['bundle'],s['context']]=r
            probes=[]
            for i in [1,2]:
                a=load(out/f'probe-{i}.json');assert a['count']==8586
                probes.append(dict(fraction=i/3,ce=a['ce'],sources=a['sources']))
            raw=scores[s['depth'],s['bundle'],s['context'],'raw','native']
            subset=[d for d in raw['documents'] if d['document'] in plan['probe_documents']]
            assert sum(d['count'] for d in subset)==8586
            probes.append(dict(fraction=1.0,ce=sum(d['loss_sum'] for d in subset)/8586))
            bins={};first_probe_steps={};last=None;peak_mps=0
            stepfile=out/'steps-01.jsonl';inputs[str(stepfile.relative_to(ROOT))]=sha(stepfile)
            with stepfile.open() as f:
                for line in f:
                    x=json.loads(line);last=x;peak_mps=max(peak_mps,x['mps_driver_bytes'])
                    k=min(119,int(x['begin_ns']/120e9));b=bins.setdefault(k,[0,0.,0,0,0])
                    b[0]+=1;b[1]+=x['loss'];b[2]=x['end_ns'];b[3]=x['step'];b[4]+=x['clip']['coefficient']<1
                    for i in [1,2]:
                        if i not in first_probe_steps and x['end_ns']>=s['budget_ns']*i/3:first_probe_steps[i]=x
            assert last['step']==r['steps'] and last['end_ns']==r['training_ns']
            for q,i in zip(probes[:2],[1,2]):q.update(training_seconds=first_probe_steps[i]['end_ns']/1e9,updates=first_probe_steps[i]['step'])
            probes[-1].update(training_seconds=r['training_ns']/1e9,updates=r['steps'])
            for line in (out/'memory-01.jsonl').read_text().splitlines():peak_mps=max(peak_mps,json.loads(line)['mps_driver_bytes'])
            assert peak_mps<plan['mps_limit_bytes'] and g['observation']['peak_rss_bytes']<plan['rss_limit_bytes']
            native_score_seconds=sum(e['score_seconds'] for e in r['endpoints']);generation_seconds=sum(e['generation_seconds'] for e in r['endpoints'])
            training_seconds=r['training_ns']/1e9
            row=dict(slot=s['slot'],depth=s['depth'],context=s['context'],bundle=s['bundle'],parameter_count=25732096+690080*s['depth'],updates=r['steps'],sampled_targets=1024*r['steps'],training_corpus_targets=10001163,nominal_exposure_passes=1024*r['steps']/10001163,training_seconds=training_seconds,guard_seconds=g['elapsed_seconds'],checkpoint_seconds=r['checkpoint_seconds_this_attempt'],native_score_seconds=native_score_seconds,generation_seconds=generation_seconds,other_guard_seconds=g['elapsed_seconds']-training_seconds-r['checkpoint_seconds_this_attempt']-native_score_seconds-generation_seconds,accounted_partial_flops=r['steps']*fp[s['depth'],s['context']],peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_driver_bytes=peak_mps,raw_ce=raw['ce'],ema_ce=r['endpoints'][1]['ce'],raw_probes=probes)
            rows.append(row)
            curves.append(dict(slot=s['slot'],depth=s['depth'],context=s['context'],bundle=s['bundle'],bins=[dict(bin_start_training_seconds=k*120,updates=z[0],mean_sampled_training_loss=z[1]/z[0],end_training_seconds=z[2]/1e9,cumulative_updates=z[3],cumulative_targets=1024*z[3],accounted_partial_flops=fp[s['depth'],s['context']]*z[3],clipped_fraction=z[4]/z[0]) for k,z in sorted(bins.items())]))
        # Independent frequency-weighted bootstrap, paired interaction before resampling.
        a=np.array([[[y['loss_sum']-x['loss_sum'] for x,y in zip(scores[d,b,128,'time_ema','native']['documents'],scores[d,b,256,'time_ema','native']['documents'])] for b in [1,2,3]] for d in [4,16]])
        docs=scores[4,1,128,'time_ema','native']['documents'];counts=np.array([x['count'] for x in docs]);sources=np.array([x['source'] for x in docs])
        contrasts=np.concatenate([a,(a[1]-a[0])[None]],axis=0);meanloss=contrasts.mean(axis=1)
        rng=np.random.Generator(np.random.PCG64(plan['bootstrap_seed']));draws=[]
        for _ in range(plan['bootstrap_resamples']):
            frequencies=np.bincount(rng.integers(0,len(counts),len(counts)),minlength=len(counts))
            draws.append((meanloss@frequencies)/(counts@frequencies))
        bounds=np.quantile(draws,plan['primary_interval_quantiles'],axis=0,method='linear').T
        for i,c in enumerate(saved['contrasts']):
            pair=contrasts[i].sum(axis=1)/counts.sum()
            assert np.allclose(pair,c['paired_deltas'],atol=1e-12,rtol=0)
            assert np.allclose(bounds[i],c['interval'],atol=1e-12,rtol=0)
            neg=bool(all(pair<0) and pair.mean()<=-.02 and bounds[i,1]<0);pos=bool(all(pair>0) and pair.mean()>=.02 and bounds[i,0]>0)
            if i<2:
                sd={z:float(contrasts[i][:,sources==z].sum()/3/counts[sources==z].sum()) for z in set(sources)}
                assert all(abs(sd[z]-c['source_deltas'][z])<1e-12 for z in sd)
                decision='context256_superior' if neg and max(sd.values())<=.05 else 'context128_superior' if pos and min(sd.values())>=-.05 else 'unresolved'
            else:decision='context_benefit_strengthens' if neg else 'context_benefit_weakens' if pos else 'unresolved_not_equivalent'
            assert decision==c['decision']
        secondary=[]
        for endpoint,evaluation in [('raw','native'),('time_ema','common128'),('raw','common128')]:
            depthrows=[]
            for d in [4,16]:
                deltas=[];sd={z:[] for z in set(sources)}
                for b in [1,2,3]:
                    x=scores[d,b,128,endpoint,evaluation];y=scores[d,b,256,endpoint,evaluation]
                    assert [(t['document'],t['source'],t['count']) for t in x['documents']]==[(t['document'],t['source'],t['count']) for t in y['documents']]
                    deltas.append(y['ce']-x['ce'])
                    for z in sd:sd[z].append(y['sources'][z]['ce']-x['sources'][z]['ce'])
                depthrows.append(dict(depth=d,paired_deltas=deltas,mean=float(np.mean(deltas)),source_means={z:float(np.mean(v)) for z,v in sd.items()}))
            secondary.append(dict(endpoint=endpoint,evaluation=evaluation,depths=depthrows,interaction_paired=(np.array(depthrows[1]['paired_deltas'])-depthrows[0]['paired_deltas']).tolist(),interaction_mean=depthrows[1]['mean']-depthrows[0]['mean']))
        import research.general_rsi_depth_context_generation_analysis_v1 as generation
        outputs={};original=generation.put
        generation.put=lambda path,value:outputs.update({path.name:value})
        try:generation.aggregate()
        finally:generation.put=original
        assert outputs['summary.json']==load(P/'generation-evaluation/summary.json')
        totals={k:sum(r[k] for r in rows) for k in ['updates','sampled_targets','training_seconds','guard_seconds','checkpoint_seconds','native_score_seconds','generation_seconds','other_guard_seconds','accounted_partial_flops']}
        totals['completed_engineering_seconds']=plan['completed_engineering_seconds'];totals['retained_endpoint_bytes']=weight_bytes
        result=dict(binding=binding,secondary_descriptive_only=secondary,rows=rows,totals=totals,training_curves_120s_bins=curves,flop_policy=flops['policy'],flop_assumptions=flops['assumptions'],timing_limits='Other guard time includes untimed common128 scoring, probes, initialization, serialization outside checkpoint timer and supervisor overhead. Guard sums exclude gaps between runs and this analysis. No failed scientific attempts or replay charges.',probe_limits='Eight fixed development documents, RAW at one-third/two-thirds and final RAW subset. Not full-development intermediate evidence or EMA trajectories.',test_opened=False,source_sha256=sha(Path(__file__)))
        put(P/'secondary-accounting.json',result)
        for n in ['decision.json','secondary-accounting.json','generation-evaluation/annotations.json','generation-evaluation/preparation.json','flops.json']:inputs[str((P/n).relative_to(ROOT))]=sha(P/n)
        host={n:subprocess.check_output(cmd,text=True).strip() for n,cmd in [('vm_stat',['vm_stat']),('swap',['sysctl','vm.swapusage']),('pressure',['sysctl','kern.memorystatus_vm_pressure_level'])]}
        receipt=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=True,binding=binding,all_12_execution_reviews_accepted=True,source_bindings_verified=len(plan['sources']),all_24_endpoint_hashes_verified=True,retained_endpoint_bytes=weight_bytes,independent_frequency_bootstrap_intervals=bounds.tolist(),independent_primary_decisions_match=True,generation_aggregate_replay_exact=True,all_288_first_outputs_rated=True,shared_lock_acquired=True,process_inspection='No local training worker in ps inspection immediately before audit; final supervisor exit0 retained.',free_bytes=shutil.disk_usage(ROOT).free,remaining_analysis_publication_allowance_bytes=256*2**20,reserve_bytes=RESERVE,host=host,analysis_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,analysis_elapsed_seconds=time.perf_counter()-started,analysis_uses_mps=False,evidence_sha256=inputs,review_source_sha256=sha(Path(__file__)),test_opened=False)
        assert receipt['free_bytes']>=RESERVE+receipt['remaining_analysis_publication_allowance_bytes'] and limits()
        put(P/'completion-review.json',receipt);stop.set()
        print(json.dumps({k:v for k,v in receipt.items() if k not in ['evidence_sha256','host']},indent=2))
if __name__=='__main__':main()
