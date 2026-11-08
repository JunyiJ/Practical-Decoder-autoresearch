"""Fixed active-row gradient comparison; no tolerance or training updates."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import traceback

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def publish(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def run():
    assert not (OUT/'guard.json').exists()
    plan=read(OUT/'plan.json');assert sha(Path(__file__))==plan['runner_sha256']
    for r in plan['bindings']:
        p=ROOT/r['path'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
    mapping=read(OUT.parent/'active-byte-source001/source-map.json')
    for r in (mapping['original'],mapping['candidate']):
        p=ROOT/r['path'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
    s=(ROOT/mapping['candidate']['path']).read_text()
    for a,c in reversed(mapping['exact_replacements']):s=s.replace(c,a)
    assert s==(ROOT/mapping['original']['path']).read_text()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json',ROOT,timeout_seconds=90,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,
                   weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            from research import byte_input_checkpoint_consumer_v4 as consumer
            from research import byte_input_training_v6 as training
            from research import byte_embedding_v1 as reference
            from research import byte_embedding_active_rows_v1 as candidate
            torch.set_num_threads(4);assert torch.backends.mps.is_available();guard.torch=torch
            before_cpu=torch.get_rng_state().clone();before_mps=torch.mps.get_rng_state().clone()
            recipe=read(OUT.parent/'bulk-hash-adoption001/recipe.json');loader=recipe['loader']
            model,identity,receipt=consumer.load(loader['implementation_record'],loader['plan_record'],loader['acceptance_record'],'cpu')
            view=training.load_view(identity['roles'],'cpu');analysis=read(OUT.parent/'active-byte-analysis001/result.json')
            full_windows=[w for w in analysis['nonoverlapping_windows'] if w['tokens']==1024]
            richest=max(full_windows,key=lambda w:w['active_rows'])['start']
            def real(start,masked):
                ids=view.train_data[start:start+1024];codes=model.code_ids.index_select(0,ids)
                lengths=model.code_lengths.index_select(0,ids)
                rows=codes-(codes>256).to(torch.int64)
                mask=torch.arange(130)[None,:]<lengths[:,None]
                return rows,mask if masked else None
            cases=[('prefix_masked',*real(0,True)),('richest_window_masked',*real(richest,True)),
                   ('prefix_dense_upstream',*real(0,False)),
                   ('all_rows_2049_control',torch.arange(2049,dtype=torch.int64)%259,None),
                   ('single_row_full_bound',torch.full((1024*130,),258,dtype=torch.int64),None),
                   ('four_rows_chunk_edge',torch.tensor([0,65,256,258],dtype=torch.int64)[torch.arange(2049)%4],None),
                   ('singleton',torch.tensor([258],dtype=torch.int64),None)]
            assert [c[0] for c in cases]==plan['fixed_cases']
            saved_weight=model.composer.byte_weight.detach().clone();del model,view
            def digest(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
            def compare(a,b):
                d=(a.double()-b.double()).abs()
                return dict(exact=torch.equal(a,b) and digest(a)==digest(b),reference_sha256=digest(a),candidate_sha256=digest(b),
                            unequal_coordinates=int((a!=b).sum()),max_abs=float(d.max()),coordinates=a.numel())
            records=[]
            for backend in ('cpu','mps'):
                weights={label:torch.nn.Parameter(saved_weight.to(backend).clone()) for label in ('reference','candidate')}
                for name,host_rows,mask in cases:
                    rows=host_rows.to(backend)
                    q=torch.arange(rows.numel()*15,dtype=torch.int64)
                    values=(((q*37+11)%1009-504).float()/997)*torch.where(q%257==0,128.,1.)
                    upstream=values.reshape(*rows.shape,15)
                    if mask is not None:upstream=upstream.masked_fill(~mask[:,:,None],0.)
                    upstream=upstream.to(backend)
                    def execute(label):
                        weight=weights[label];weight.grad=None
                        if backend=='mps':torch.mps.synchronize()
                        start=time.perf_counter();output=(reference if label=='reference' else candidate).embedding(weight,rows)
                        output.backward(upstream)
                        if backend=='mps':torch.mps.synchronize()
                        elapsed=time.perf_counter()-start
                        return elapsed,output.detach().cpu().clone(),weight.grad.detach().cpu().clone()
                    baseline=execute('reference');proposed=execute('candidate')
                    output=compare(baseline[1],proposed[1]);gradient=compare(baseline[2],proposed[2])
                    timings=None;selection=None;repeatable=True
                    if name=='prefix_masked':
                        timings={'reference':[],'candidate':[]};selection=[]
                        # Initial paired execution above is the declared warmup.
                        for repeat in range(7):
                            for label in (['reference','candidate'] if repeat%2==0 else ['candidate','reference']):
                                result=execute(label);timings[label].append(result[0])
                                expected=baseline if label=='reference' else proposed
                                repeatable=repeatable and compare(expected[1],result[1])['exact'] and compare(expected[2],result[2])['exact']
                            if backend=='mps':torch.mps.synchronize()
                            started=time.perf_counter();active=torch.unique(rows.reshape(-1).detach().cpu(),sorted=True).to(backend)
                            if backend=='mps':torch.mps.synchronize()
                            selection.append(time.perf_counter()-started);del active
                    row=dict(backend=backend,case=name,shape=list(rows.shape),active_rows=torch.unique(host_rows).numel(),
                        row_ids_sha256=digest(host_rows),upstream_sha256=digest(upstream),output=output,gradient=gradient,
                        exact=output['exact'] and gradient['exact'] and repeatable,repeatable=repeatable,
                        timings_seconds=timings,selection_seconds=selection)
                    publish(backend+'-'+name+'.json',row);records.append(row)
                    del baseline,proposed,rows,upstream
                assert all(digest(w)==digest(saved_weight) for w in weights.values())
                del weights
            assert torch.equal(before_cpu,torch.get_rng_state()) and torch.equal(before_mps,torch.mps.get_rng_state())
            gpu=next(r for r in records if r['backend']=='mps' and r['case']=='prefix_masked')
            timing=gpu['timings_seconds'];ratios=[b/a for a,b in zip(timing['reference'],timing['candidate'])]
            exact=all(r['exact'] for r in records);useful=statistics.median(ratios)<=.8 and sum(r<1 for r in ratios)>=5
            result=dict(completed=True,exact=exact,useful_mps_component_gain=useful,eligible_for_integration=exact and useful,
                adopted=False,cases=records,median_candidate_over_reference=statistics.median(ratios),paired_faster=sum(r<1 for r in ratios),
                selection_median_seconds=statistics.median(gpu['selection_seconds']),training_updates=0,quality_evaluations=0,
                weights_unchanged=True,rng_unchanged=True,plan_sha256=sha(OUT/'plan.json'),runner_sha256=sha(Path(__file__)),
                limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status']=='finished'
        result['guard_sha256']=sha(OUT/'guard.json');publish('result.json',result)
        publish('STOP',dict(status='complete_candidate_pending_integration' if exact and useful else 'complete_not_exact_not_adopted' if not exact else 'complete_no_useful_gain',no_replay=True))
    print(json.dumps({k:result[k] for k in ['completed','exact','useful_mps_component_gain','eligible_for_integration','median_candidate_over_reference','paired_faster','selection_median_seconds']}))
    for row in records:print(json.dumps({k:row[k] for k in ['backend','case','active_rows','exact','gradient']}))


if __name__=='__main__':
    try:run()
    except Exception:
        publish('failure.json',dict(traceback=traceback.format_exc(),no_automatic_replay=True));raise
