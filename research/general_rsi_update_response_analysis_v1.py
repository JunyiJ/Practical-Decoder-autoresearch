"""Audit all first-update records and apply the unchanged finite-step criterion."""
import datetime,fcntl,json,math,shutil,statistics
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put
from research.general_rsi_update_response_diagnostic_v1 import OUT

def read(p):return json.loads(p.read_text())
def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as l:
        fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json');g=read(a/'guard.json');assert g['status']=='finished' and not list(a.glob('*failure*'))
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items());binding=sha(OUT/'plan.json');assert r['plan_sha256']==binding==read(OUT/'admission.json')['plan_sha256']
        evaluations=0;targets=0;updates=0;clipped=0;rows=[];groups=[]
        def check(s,n):
            nonlocal evaluations,targets
            assert s['targets']==n and s['window_targets']==[128]*(n//128) and len(s['window_loss_sums'])==n//128
            assert all(math.isfinite(x) for x in s['window_loss_sums']) and math.fsum(s['window_loss_sums'])/n==s['ce']
            evaluations+=1;targets+=n
        for c in p['parents']:
            slot=c['parent']['slot'];initial=read(a/f'p{slot:02d}-initial.json');baseline=initial['evaluation'];check(baseline,8192)
            for case in c['cases']:
                i=case['case'];v=read(a/f'p{slot:02d}-c{i:02d}-result.json');u=v['update'];check(v['batch_initial'],1024)
                assert v['pristine_optimizer_checked'] and v['exact_restoration'] and v['initial_parameters_sha256']==initial['parameters_sha256']
                assert u['step']==1 and u['objective']=='ce' and u['lr_ratio']==1. and u['batch_sha256']==c['inputs']['cases'][i]['sha256']
                assert u['valid_loss_targets']==1024 and math.isfinite(u['loss']) and math.isfinite(u['clip']['pre_norm']);updates+=1;clipped+=u['clip']['coefficient']<1
                for group,geometry in v['displacement'].items():
                    assert all(math.isfinite(geometry[k]) for k in ['parameter_sq','update_sq','update_norm','relative_update_norm'])
                    groups.append(dict(parent=slot,case=i,group=group,**geometry))
                actual=[]
                for scale in p['nonzero_scales']:
                    point=read(a/f'p{slot:02d}-c{i:02d}-a{scale:g}.json');assert point['parent']==slot and point['case']==i and point['scale']==scale
                    check(point['probe'],8192);check(point['batch'],1024)
                    assert point['probe_delta']==point['probe']['ce']-baseline['ce'] and point['batch_delta']==point['batch']['ce']-v['batch_initial']['ce']
                    assert point['probe_secant']==point['probe_delta']/scale and point['batch_secant']==point['batch_delta']/scale
                    rows.append({k:point[k] for k in ['parent','case','scale','probe_delta','batch_delta','probe_secant','batch_secant']});actual.append({k:point[k] for k in ['scale','probe_delta','batch_delta']})
                assert dict(parent=slot,case=i,points=actual)==next(x for x in r['records'] if x['parent']==slot and x['case']==i)
        assert updates==r['updates']==24 and evaluations==171 and targets==r['scored_targets']==712704
        assert r['parameter_copy_checks']==8448 and r['pristine_state_checks']==24
        means=[]
        for c in p['parents']:
            slot=c['parent']['slot']
            for scale in p['nonzero_scales']:
                subset=[v for v in rows if v['parent']==slot and v['scale']==scale]
                means.append(dict(parent=slot,scale=scale,probe_delta=statistics.mean(v['probe_delta'] for v in subset),batch_delta=statistics.mean(v['batch_delta'] for v in subset),probe_positive=sum(v['probe_delta']>0 for v in subset),batch_negative=sum(v['batch_delta']<0 for v in subset)))
        criteria=[dict(parent=c['parent']['slot'],supported=next(v['probe_delta'] for v in means if v['parent']==c['parent']['slot'] and v['scale']==1.)>0 and next(v['probe_delta'] for v in means if v['parent']==c['parent']['slot'] and v['scale']==.0625)<0) for c in p['parents']]
        across=[dict(scale=scale,probe_delta=statistics.mean(v['probe_delta'] for v in means if v['scale']==scale),batch_delta=statistics.mean(v['batch_delta'] for v in means if v['scale']==scale)) for scale in p['nonzero_scales']]
        timing={str(n):dict(mean=statistics.mean(v['seconds'] for v in r['score_timings'] if v['targets']==n),max=max(v['seconds'] for v in r['score_timings'] if v['targets']==n)) for n in [1024,8192]}
        put(OUT/'analysis.json',dict(plan_sha256=binding,rows=rows,parent_means=means,across_parent_means=across,primary_parent_criteria=criteria,primary_support=all(v['supported'] for v in criteria),groups=groups,quality_adopted=False,scope='Finite-scale training-only first-update diagnostic; no scale selection, population inference or exact derivative claim.'))
        review=dict(accepted_execution=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=binding,result_sha256=sha(a/'result.json'),analysis_sha256=sha(OUT/'analysis.json'),analysis_source_sha256=sha(__file__),source_bindings_unchanged=len(p['sources']),updates=updates,clipped_updates=clipped,evaluations=evaluations,scored_targets=targets,parameter_copy_checks=r['parameter_copy_checks'],pristine_state_checks=24,guard_seconds=g['elapsed_seconds'],training_seconds=r['training_seconds'],peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_bytes=r['peak_mps_bytes'],score_seconds=timing,failures=0,lock_reacquired=True,metadata_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),free_bytes=shutil.disk_usage(ROOT).free,new_checkpoints=0,test_opened=False,validation_scored=False,quality_adopted=False)
        put(OUT/'completion-review.json',review);print(json.dumps(dict(means=means,across=across,criteria=criteria,review=review),indent=2))
if __name__=='__main__':main()
