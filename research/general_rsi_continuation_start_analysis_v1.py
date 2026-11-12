"""Review all fresh paired starting-point trajectories without rescoring."""
import datetime,fcntl,json,math,shutil,statistics
from research.general_rsi_continuation_start_diagnostic_v1 import ROOT,OUT
from research.general_rsi_pseudo_update_diagnostic_v1 import sha,put

def read(p):return json.loads(p.read_text())
def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json');g=read(a/'guard.json')
        assert g['status']=='finished' and not list(a.glob('*failure*'))
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
        assert read(OUT/'admission.json')['plan_sha256']==sha(OUT/'plan.json')==r['plan_sha256']
        rows=[];updates=0;scored=0;tensors=0
        def check_score(s):
            assert s['targets']==8192 and len(s['window_loss_sums'])==64 and all(math.isfinite(x) for x in s['window_loss_sums'])
            assert math.fsum(s['window_loss_sums'])/8192==s['ce']
        for c in p['parents']:
            slot=c['parent']['slot'];baseline={}
            for endpoint in ['raw','time_ema']:
                loader=read(a/f'p{slot:02d}-{endpoint}-loader.json')
                assert loader['receipt']==c['starting_weights'][endpoint] and loader['exact_parameter_tensors']==88 and loader['optimizer_pristine'] and loader['model_buffer_identity_checked']
                tensors+=loader['exact_parameter_tensors'];base=read(a/f'p{slot:02d}-{endpoint}-initial.json');assert base['other_initial_state_exact']
                baseline[endpoint]=base['evaluation'];check_score(baseline[endpoint]);scored+=1
                history=[json.loads(l) for l in (a/f'p{slot:02d}-{endpoint}-steps.jsonl').read_text().splitlines()];assert len(history)==100
                for n,(h,want) in enumerate(zip(history,c['inputs']['natural']),1):
                    assert h['step']==n and h['batch_sha256']==want['sha256'] and h['objective']=='ce' and h['lr_ratio']==1.
                    assert math.isfinite(h['loss']) and math.isfinite(h['clip']['pre_norm']);updates+=1
                end=read(a/f'p{slot:02d}-{endpoint}-result.json');assert end['initial_nonweight_state_exact'] and end['steps']==100
                assert end==next(v for v in r['reports'] if v['parent']==slot and v['endpoint']==endpoint)
            for n in p['probes']:
                scores={};drifts={}
                for endpoint in ['raw','time_ema']:
                    v=read(a/f'p{slot:02d}-{endpoint}-n{n:03d}.json');s=v['evaluation'];check_score(s);scored+=1
                    assert v['parent']==slot and v['endpoint']==endpoint and v['n']==n and v['delta_from_own_start']==s['ce']-baseline[endpoint]['ce']
                    scores[endpoint]=s['ce'];drifts[endpoint]=v['delta_from_own_start']
                rows.append(dict(parent=slot,n=n,scores=scores,initial_ces={k:v['ce'] for k,v in baseline.items()},drifts=drifts,
                    drift_difference_raw_minus_ema=drifts['raw']-drifts['time_ema'],endpoint_raw_minus_ema=scores['raw']-scores['time_ema'],
                    gaps_to_unchanged_ema={k:v-baseline['time_ema']['ce'] for k,v in scores.items()}))
        assert updates==r['updates']==600 and scored==r['evaluations']==36 and tensors==528
        means=[dict(n=n,drift_difference_raw_minus_ema=statistics.mean(x['drift_difference_raw_minus_ema'] for x in rows if x['n']==n),
            endpoint_raw_minus_ema=statistics.mean(x['endpoint_raw_minus_ema'] for x in rows if x['n']==n),
            drifts={e:statistics.mean(x['drifts'][e] for x in rows if x['n']==n) for e in ['raw','time_ema']},
            gaps_to_unchanged_ema={e:statistics.mean(x['gaps_to_unchanged_ema'][e] for x in rows if x['n']==n) for e in ['raw','time_ema']}) for n in p['probes']]
        final=[x for x in rows if x['n']==100];support=all(x['drift_difference_raw_minus_ema']<0 for x in final)
        put(OUT/'analysis.json',dict(plan_sha256=sha(OUT/'plan.json'),rows=rows,means=means,primary_consistent_support=support,
            all_final_endpoints_above_unchanged_ema=all(v>0 for x in final for v in x['gaps_to_unchanged_ema'].values()),quality_adopted=False,
            scope='Fixed100-update paired training-only initial-weight contrast. Fresh optimizer shared; not an optimizer-state comparison. No validation, generation or quality adoption.'))
        put(OUT/'completion-review.json',dict(accepted_execution=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),
            result_sha256=sha(a/'result.json'),analysis_sha256=sha(OUT/'analysis.json'),analysis_source_sha256=sha(__file__),source_bindings_unchanged=len(p['sources']),
            updates=updates,evaluations=scored,loader_tensor_matches=tensors,nonweight_state_checks=6,guard_seconds=g['elapsed_seconds'],training_seconds=r['training_seconds'],
            peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_bytes=r['peak_mps_bytes'],lock_reacquired=True,failures=0,
            metadata_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),free_bytes=shutil.disk_usage(ROOT).free,new_checkpoints=0,test_opened=False,validation_scored=False,quality_adopted=False))
        print(json.dumps(dict(means=means,primary_consistent_support=support,final=final),indent=2))
if __name__=='__main__':main()
