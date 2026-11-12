"""Review all first outputs and apply the frozen batch-averaging criterion."""
import datetime,fcntl,json,math,shutil,statistics
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put
from research.general_rsi_batch_averaging_diagnostic_v1 import OUT

def read(p):return json.loads(p.read_text())
def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as l:
        fcntl.flock(l,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json');g=read(a/'guard.json');binding=sha(OUT/'plan.json')
        assert g['status']=='finished' and not list(a.glob('*failure*')) and r['plan_sha256']==binding==read(OUT/'admission.json')['plan_sha256']
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
        evaluations=0;passes=0;rows=[];records=[]
        def check(s):
            nonlocal evaluations
            assert s['targets']==8192 and s['window_targets']==[128]*64 and len(s['window_loss_sums'])==64 and all(math.isfinite(x) for x in s['window_loss_sums'])
            assert math.fsum(s['window_loss_sums'])/8192==s['ce'];evaluations+=1
        for c in p['parents']:
            slot=c['parent']['slot'];initial=read(a/f'p{slot:02d}-initial.json');check(initial['evaluation'])
            for case in c['cases']:
                i=case['case'];deltas={}
                for count in p['arms']:
                    v=read(a/f'p{slot:02d}-c{i:02d}-k{count}.json');u=v['update'];check(v['evaluation'])
                    assert v['initial_state_exact'] and v['parameters_sha256']==initial['parameters_sha256']
                    assert u['step']==1 and u['microbatches']==count and u['lr_ratio']==1. and u['objective']=='ce'
                    assert u['microbatch_sha256']==[z['sha256'] for z in c['inputs']['cases'][i]['batches'][:count]]
                    assert u['valid_loss_targets']==count*1024 and len(u['microbatch_losses'])==count and math.fsum(u['microbatch_losses'])/count==u['loss']
                    assert all(math.isfinite(z) for z in u['microbatch_losses']) and math.isfinite(u['clip']['pre_norm'])
                    assert v['delta_from_parent']==v['evaluation']['ce']-initial['evaluation']['ce'];deltas[count]=v['delta_from_parent'];passes+=count
                    assert dict(parent=slot,case=i,microbatches=count,delta_from_parent=v['delta_from_parent'])==next(z for z in r['records'] if z['parent']==slot and z['case']==i and z['microbatches']==count)
                    records.append(dict(parent=slot,case=i,microbatches=count,delta_from_parent=v['delta_from_parent'],training_seconds=(u['end_ns']-u['begin_ns'])/1e9,pre_clip_norm=u['clip']['pre_norm'],clipped=u['clip']['coefficient']<1,group_update_norms={k:x['update_norm'] for k,x in v['displacement'].items()}))
                rows.append(dict(parent=slot,case=i,single_delta=deltas[1],eight_delta=deltas[8],paired_eight_minus_single=deltas[8]-deltas[1]))
        assert evaluations==r['evaluations']==51 and passes==r['forward_backward_passes']==216 and len(records)==r['updates']==48
        assert r['training_targets']==221184 and r['scored_targets']==417792 and r['initial_state_checks']==48
        means=[]
        for c in p['parents']:
            slot=c['parent']['slot'];rs=[v for v in rows if v['parent']==slot]
            m=dict(parent=slot,**{k:statistics.mean(v[k] for v in rs) for k in ['single_delta','eight_delta','paired_eight_minus_single']},eight_better_than_single=sum(v['paired_eight_minus_single']<0 for v in rs),eight_better_than_parent=sum(v['eight_delta']<0 for v in rs))
            m['supported']=m['paired_eight_minus_single']<0 and m['eight_delta']<0;means.append(m)
        overall={k:statistics.mean(v[k] for v in means) for k in ['single_delta','eight_delta','paired_eight_minus_single']}
        timing={str(k):dict(total_training_seconds=sum(v['training_seconds'] for v in records if v['microbatches']==k),mean_update_seconds=statistics.mean(v['training_seconds'] for v in records if v['microbatches']==k),clipped=sum(v['clipped'] for v in records if v['microbatches']==k)) for k in [1,8]}
        put(OUT/'analysis.json',dict(plan_sha256=binding,rows=rows,records=records,parent_means=means,overall=overall,primary_support=all(v['supported'] for v in means),timing=timing,quality_adopted=False,scope='Fresh one-step training-only fixed update count. Eightfold candidate targets; no equal-time/token quality or population claim.'))
        review=dict(accepted_execution=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=binding,result_sha256=sha(a/'result.json'),analysis_sha256=sha(OUT/'analysis.json'),analysis_source_sha256=sha(__file__),source_bindings_unchanged=len(p['sources']),updates=48,forward_backward_passes=216,training_targets=221184,evaluations=51,scored_targets=417792,initial_state_checks=48,guard_seconds=g['elapsed_seconds'],training_seconds=r['training_seconds'],peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_bytes=r['peak_mps_bytes'],failures=0,lock_reacquired=True,metadata_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),free_bytes=shutil.disk_usage(ROOT).free,new_checkpoints=0,test_opened=False,validation_scored=False,quality_adopted=False)
        put(OUT/'completion-review.json',review);print(json.dumps(dict(means=means,overall=overall,timing=timing,review=review),indent=2))
if __name__=='__main__':main()
