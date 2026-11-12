"""Review every frozen norm-restoration condition without rescoring models."""
import datetime,fcntl,json,math,shutil,statistics
from research.general_rsi_norm_restoration_v1 import ROOT,OUT,read,sha,put,CONDITIONS

def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json');g=read(a/'guard.json')
        assert g['status']=='finished' and not list(a.glob('*failure*'))
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
        assert read(a/'admission.json')['plan_sha256']==sha(OUT/'plan.json') and read(a/'qualification.json')['passed']
        count=0
        for m in p['models']:
            parent=m['parent']['slot'];base=read(a/f'p{parent:02d}-parent.json');count+=1
            assert base['targets']==8192 and len(base['window_loss_sums'])==64
            for e in m['endpoints']:
                row=next(x for x in r['rows'] if x['parent']==parent and x['endpoint']==e['endpoint'])
                assert base['ce']==row['parent_ce']
                for c in CONDITIONS:
                    v=read(a/f"p{parent:02d}-{e['endpoint']}-{c}.json")['score'];count+=1
                    assert v['targets']==8192 and len(v['window_loss_sums'])==64
                    assert math.fsum(v['window_loss_sums'])/8192==v['ce']==row['scores'][c]
                    assert all(math.isfinite(x) for x in v['window_loss_sums'])
        assert count==r['conditions_scored']==27
        summary={e:{key:{c:statistics.mean(x[key][c] for x in r['rows'] if x['endpoint']==e) for c in r['rows'][0][key]} for key in ['effects','residuals']} for e in ['raw','time_ema']}
        put(OUT/'analysis.json',dict(result_sha256=sha(a/'result.json'),means=summary,paired_rows=r['rows'],
            all_readout_effects_negative=all(x['effects']['readout']<0 for x in r['rows']),
            all_other_and_both_effects_positive=all(x['effects'][c]>0 for x in r['rows'] for c in ['other','both']),
            all_residuals_above_parent=all(v>0 for x in r['rows'] for v in x['residuals'].values()),
            scope='Shared8192 training targets, descriptive six endpoint comparisons. No calibration metric, holdout improvement, decay causality, or model adoption established.'))
        files={str(f.relative_to(ROOT)):sha(f) for f in OUT.rglob('*') if f.is_file()}
        put(OUT/'completion-review.json',dict(accepted_execution=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            plan_sha256=sha(OUT/'plan.json'),retained_evidence=files,source_bindings_unchanged=len(p['sources']),conditions=27,targets_per_condition=8192,
            guard_seconds=g['elapsed_seconds'],peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_bytes=r['peak_mps_bytes'],
            metadata_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),actual_free_bytes=shutil.disk_usage(ROOT).free,
            lock_reacquired=True,training_updates=0,new_checkpoints=0,failures=0,test_opened=False,validation_scored=False,quality_adopted=False))
        print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
