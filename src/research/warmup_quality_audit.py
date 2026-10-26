"""Independent warmup report arithmetic; no models, training or checkpoint loads.

The full checkpoint/request/source/optimizer audit is a separate required gate.
"""
import math
from statistics import mean
import numpy as np
from src.research.protocol import digest


def endpoint(row, raw=False):
    return row['coverage']['128'] if raw else row['averaged_outputs']['200']['coverage']['evaluations']['128']


def audit(p, state, report):
    rows = state['records']; selection = state['selection']
    families = ['baseline','warmup100','warmup300']; seeds = [2291,2292,2293]
    lr = p['recipe']['learning_rate']
    assert p['learning_rates'] == [lr]
    assert p['warmup_steps'] == dict(baseline=0,warmup100=100,warmup300=300)
    assert p['initial_ratio'] == .1 and p['max_trials'] == 24 and p['max_model_evaluations'] == 48
    assert p['seeds'] == [2281,2282] and p['confirmation_seeds'] == seeds and p['widths'] == [64,256]
    assert p['bootstrap_blocks'] == [2048,4096,8192] and p['bootstrap_draws'] == 2000 and p['bootstrap_seed'] == 20261024
    assert report['records'] == rows and report['selection'] == selection
    assert report['plan_sha256'] == digest(p) and report['scope'] == p['scope']
    assert report['holdout_consumed'] is False and report['adoption_requires_completion_review'] is True
    assert len(rows) == report['physical_training_trajectories'] == 24
    assert report['model_evaluations'] == 48 and report['reference_aliases'] == {}
    assert all(r['status']=='completed' for r in rows)
    assert len({r['trial_id'] for r in rows}) == 24
    screen_jobs = [[f,lr,64,s] for i,s in enumerate([2281,2282]) for f in (families if i==0 else families[::-1])]
    conf_jobs = [[f,lr,w,s] for w in [64,256] for i,s in enumerate(seeds) for f in (families if i%2==0 else families[::-1])]
    assert [[r[k] for k in ['family','lr','width','seed']] for r in rows] == screen_jobs+conf_jobs
    assert [r['stage'] for r in rows] == ['local']*6+['confirmation']*18
    for row in rows:
        assert set(row['averaged_outputs']) == {'200'}
        for raw in [False,True]:
            v=endpoint(row,raw)
            assert (v['context'],v['eval_batch_size'],v['block_tokens'],v['target_count'],v['window_count']) == (128,8,2048,111538,872)
            assert v['block_target_counts'] == [2048]*54+[946]
            assert len(v['block_loss_sums']) == 55 and all(math.isfinite(x) and x>=0 for x in v['block_loss_sums'])
            assert math.isclose(sum(v['block_loss_sums'])/111538,v['mean_ce'],abs_tol=1e-12,rel_tol=0)
    screen, conf = rows[:6],rows[6:]
    means = {f:mean(endpoint(r)['mean_ce'] for r in screen if r['family']==f) for f in families}
    best = min(families,key=lambda f:(means[f],families.index(f)))
    panels = {f:{r['seed']:endpoint(r)['mean_ce'] for r in screen if r['family']==f} for f in families}
    ds = {str(s):panels[best][s]-panels['baseline'][s] for s in [2281,2282]}
    eligible = best!='baseline' and mean(ds.values())<-.005 and max(ds.values())<0
    winner = best if eligible else 'incumbent_unchanged'
    assert selection['means']==means and selection['fixed_peak_lr']==lr
    assert selection['best_selected_arm']==best and selection['candidate_eligible']==eligible and selection['winner']==winner
    assert selection['local_contrast']==dict(candidate=best,reference='baseline',paired_seed_deltas=ds,mean_delta=mean(ds.values()),supported=eligible)
    assert selection['plan_sha256']==digest(p) and selection['local_evidence_sha256']==digest(screen)
    assert selection['confirmation']==dict(jobs=conf_jobs,reference_aliases={},total_distinct_trials=24,total_model_evaluations=48)
    pairs = [('warmup100','baseline'),('warmup300','baseline'),('warmup300','warmup100')]
    assert len(report['comparisons']) == 6
    assert [(v['width'],v['candidate'],v['reference']) for v in report['comparisons']] == [(w,a,b) for w in [64,256] for a,b in pairs]
    adoption = []
    for c in report['comparisons']:
        panel = {f:{r['seed']:r for r in conf if r['width']==c['width'] and r['family']==f} for f in [c['candidate'],c['reference']]}
        for x in panel.values(): assert set(x)==set(seeds)
        primary_mean=None; primary_seed=None; primary_upper=[]
        for raw,key in [(False,'sensitivity'),(True,'raw_sensitivity')]:
            aa=[endpoint(panel[c['candidate']][s],raw) for s in seeds]
            bb=[endpoint(panel[c['reference']][s],raw) for s in seeds]
            delta=np.asarray([[a-b for a,b in zip(x['block_loss_sums'],y['block_loss_sums'])] for x,y in zip(aa,bb)]).mean(axis=0)
            counts=np.asarray([2048]*54+[946])
            seed_delta={str(s):a['mean_ce']-b['mean_ce'] for s,a,b in zip(seeds,aa,bb)}
            assert len(c[key])==3
            for block,v in zip([2048,4096,8192],c[key]):
                factor=block//2048
                losses=np.asarray([sum(delta[i:i+factor]) for i in range(0,55,factor)])
                weights=np.asarray([sum(counts[i:i+factor]) for i in range(0,55,factor)])
                index=np.random.default_rng(20261024).integers(len(weights),size=(2000,len(weights)))
                draws=losses[index].sum(axis=1)/weights[index].sum(axis=1)
                interval=np.quantile(draws,[.025,.975])
                value=float(losses.sum()/weights.sum())
                assert v['text_block_tokens']==block and v['text_block_count']==len(weights)
                assert v['bootstrap_draws']==2000 and v['bootstrap_seed']==20261024
                assert v['paired_seed_deltas']==seed_delta
                assert math.isclose(v['mean_delta'],value,abs_tol=1e-12,rel_tol=0)
                assert np.allclose(v['conditional_text_bootstrap_95pct'],interval,atol=1e-12,rtol=0)
                assert v['scope']=='Text-block resampling conditional on these trained seeds and this adaptively reused validation split. Not seed uncertainty, independent holdout, or a multiple-comparison-adjusted significance test.'
                if not raw:
                    primary_mean=value; primary_seed=seed_delta; primary_upper.append(float(interval[1]))
        supported=primary_mean<-.005 and max(primary_seed.values())<0 and max(primary_upper)<0
        assert c['quality_improvement']==supported
        if c['candidate']==winner and c['reference']=='baseline': adoption.append(supported)
    assert report['quality_decision']==dict(frozen_screen_winner=winner,
        qualifies_for_incumbent_replacement=winner!='incumbent_unchanged' and len(adoption)==2 and all(adoption))
    fixed=[]
    for a,b in pairs:
        item=dict(lr=lr,candidate=a,reference=b)
        for raw,key in [(False,'primary'),(True,'raw')]:
            panel={f:{r['seed']:endpoint(r,raw)['mean_ce'] for r in screen if r['family']==f} for f in [a,b]}
            deltas={str(s):panel[a][s]-panel[b][s] for s in [2281,2282]}
            item[key]=dict(mean_delta=mean(deltas.values()),paired_seed_deltas=deltas)
        fixed.append(item)
    assert report['fixed_lr_screen']==fixed
    return dict(status='report_arithmetic_verified',physical_trajectories=24,
        correlated_outputs=48,quality_decision=report['quality_decision'])
