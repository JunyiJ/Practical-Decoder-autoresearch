"""Review all fixed trajectories and apply the predeclared diagnostic contrasts."""
import datetime,fcntl,json,math,shutil,statistics
from research.general_rsi_continuation_decay_diagnostic_v1 import ROOT,OUT,POLICIES
from research.general_rsi_pseudo_update_diagnostic_v1 import sha,put

def read(p):return json.loads(p.read_text())
def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=read(OUT/'plan.json');a=OUT/'attempt-01';r=read(a/'result.json');g=read(a/'guard.json')
        assert g['status']=='finished' and not list(a.glob('*failure*'))
        assert all(sha(ROOT/f)==h for f,h in p['sources'].items())
        assert read(OUT/'admission.json')['plan_sha256']==sha(OUT/'plan.json')==r['plan_sha256']
        rows=[];updates=0;scored=0
        def check_score(s):
            assert s['targets']==8192 and len(s['window_loss_sums'])==64 and all(math.isfinite(x) for x in s['window_loss_sums'])
            assert math.fsum(s['window_loss_sums'])/8192==s['ce']
        for c in p['parents']:
            slot=c['parent']['slot'];initial=read(a/f'p{slot:02d}-initial.json');base=initial['evaluation'];check_score(base);scored+=1
            for policy in c['order']:
                history=[json.loads(l) for l in (a/f'p{slot:02d}-{policy}-steps.jsonl').read_text().splitlines()]
                assert len(history)==100
                for n,(h,want) in enumerate(zip(history,c['inputs']['natural']),1):
                    assert h['step']==n and h['batch_sha256']==want['sha256'] and h['objective']=='ce' and h['lr_ratio']==1.
                    assert math.isfinite(h['loss']) and math.isfinite(h['clip']['pre_norm']);updates+=1
                end=read(a/f'p{slot:02d}-{policy}-result.json');assert end['initial_state_exact'] and end['steps']==100
                assert end==next(v for v in r['reports'] if v['parent']==slot and v['policy']==policy)
            for n in p['probes']:
                scores={}
                for policy in POLICIES:
                    v=read(a/f'p{slot:02d}-{policy}-n{n:03d}.json');s=v['evaluation'];check_score(s);scored+=1
                    assert v['parent']==slot and v['policy']==policy and v['n']==n and v['delta_from_parent']==s['ce']-base['ce']
                    scores[policy]=s['ce']
                rows.append(dict(parent=slot,n=n,scores=scores,parent_ce=base['ce'],drifts={k:v-base['ce'] for k,v in scores.items()},
                    remove_readout=scores['other_only']-scores['all'],remove_other=scores['readout_only']-scores['all'],remove_both=scores['none']-scores['all'],
                    interaction=scores['none']-scores['other_only']-scores['readout_only']+scores['all']))
        assert updates==r['updates']==1200 and scored==r['evaluations']==63 and r['initial_state_comparisons']==12
        contrasts=['remove_readout','remove_other','remove_both','interaction']
        means=[dict(n=n,**{k:statistics.mean(x[k] for x in rows if x['n']==n) for k in contrasts},drifts={policy:statistics.mean(x['drifts'][policy] for x in rows if x['n']==n) for policy in POLICIES}) for n in p['probes']]
        final=[x for x in rows if x['n']==100]
        support=all(x['remove_readout']<0 and x['drifts']['other_only']<x['drifts']['all'] for x in final)
        analysis=dict(plan_sha256=sha(OUT/'plan.json'),rows=rows,means=means,primary_consistent_support=support,
            all_final_arms_above_parent=all(d>0 for x in final for d in x['drifts'].values()),
            endpoint_norm_ratios=[{k:x[k] for k in ['parent','policy','norm_ratios','clipped']} for x in r['reports']],quality_adopted=False,
            scope='Fixed100-update training-only causal policy contrast in these parents/streams. No validation/generalization/generation or long-horizon claim.')
        put(OUT/'analysis.json',analysis)
        put(OUT/'completion-review.json',dict(accepted_execution=True,completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(OUT/'plan.json'),
            result_sha256=sha(a/'result.json'),analysis_sha256=sha(OUT/'analysis.json'),analysis_source_sha256=sha(__file__),source_bindings_unchanged=len(p['sources']),
            updates=updates,evaluations=scored,initial_state_matches=12,guard_seconds=g['elapsed_seconds'],training_seconds=r['training_seconds'],
            peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_bytes=r['peak_mps_bytes'],lock_reacquired=True,failures=0,
            metadata_bytes=sum(f.stat().st_size for f in OUT.rglob('*') if f.is_file()),free_bytes=shutil.disk_usage(ROOT).free,new_checkpoints=0,test_opened=False,validation_scored=False,quality_adopted=False))
        print(json.dumps(dict(means=means,primary_consistent_support=support,final=final),indent=2))
if __name__=='__main__':main()
