"""Independent report/selection/bootstrap arithmetic only; no checkpoint or model loads."""
import math
from statistics import mean
import numpy as np
from src.research.protocol import digest


def endpoint(row,raw=False):
    return row['coverage']['128'] if raw else row['averaged_outputs']['200']['coverage']['evaluations']['128']


def audit(p,state,report):
    rows=state['records']; selection=state['selection']
    families=['layer_affine','layer_no_bias','rms']; rates=[.0025,.005,.01]
    seeds=[2331,2332,2333]; screens=[2321,2322]; reference='incumbent_reference'
    assert p['families']==families and p['learning_rates']==rates
    assert p['readout_norm_kinds']==dict(layer_affine='layer_affine',layer_no_bias='layer_no_bias',rms='rms')
    assert p['readout_norm_policy']=='mha4_readout_norm' and p['readout_norm_epsilon']==1e-5 and p['max_trials']==42 and p['max_model_evaluations']==84
    incumbent_lr=p['recipe']['learning_rate']
    assert incumbent_lr in rates and p['recipe']['input_embedding_scale'] in (1.,.5,.25) and p['recipe']['weight_decay']==.00006
    assert p['recipe']['warmup_steps']==0
    assert p['seeds']==screens and p['confirmation_seeds']==seeds and p['widths']==[64,256]
    assert p['bootstrap_blocks']==[2048,4096,8192] and p['bootstrap_draws']==2000 and p['bootstrap_seed']==20261024
    assert report['records']==rows and report['selection']==selection
    assert report['plan_sha256']==digest(p) and report['scope']==p['scope']
    assert report['holdout_consumed'] is False and report['adoption_requires_completion_review'] is True
    assert all(r['status']=='completed' for r in rows) and len({r['trial_id'] for r in rows})==len(rows)
    for row in rows:
        assert set(row['averaged_outputs'])=={'200'}
        for raw in [False,True]:
            v=endpoint(row,raw)
            assert (v['context'],v['eval_batch_size'],v['block_tokens'],v['target_count'],v['window_count'])==(128,8,2048,111538,872)
            assert v['block_target_counts']==[2048]*54+[946]
            assert len(v['block_loss_sums'])==55 and all(math.isfinite(x) and x>=0 for x in v['block_loss_sums'])
            assert math.isclose(sum(v['block_loss_sums'])/111538,v['mean_ce'],abs_tol=1e-12,rel_tol=0)
    screen=rows[:18];conf=rows[18:]
    screen_jobs=[[f,lr,64,s] for lr in rates for i,s in enumerate(screens) for f in (families if i==0 else families[::-1])]
    assert [[r[k] for k in ['family','lr','width','seed']] for r in screen]==screen_jobs
    assert [r['stage'] for r in screen]==['local']*18
    choices={}
    for f in families:
        grid=[dict(lr=lr,mean_ce=mean(endpoint(r)['mean_ce'] for r in screen if r['family']==f and r['lr']==lr)) for lr in rates]
        choice=min(grid,key=lambda v:(v['mean_ce'],v['lr']))
        choices[f]={**choice,'grid':grid,'at_grid_boundary':choice['lr'] in (.0025,.01)}
    best=min(families,key=lambda f:(choices[f]['mean_ce'],families.index(f)))
    def panel(f,lr):return {r['seed']:endpoint(r)['mean_ce'] for r in screen if r['family']==f and r['lr']==lr}
    aa=panel(best,choices[best]['lr']);references=[(reference,incumbent_lr)]
    if best!='layer_affine':references.append(('layer_affine',choices['layer_affine']['lr']))
    contrasts=[]
    for ref,lr in references:
        bb=panel('layer_affine',lr);ds={str(s):aa[s]-bb[s] for s in screens}
        contrasts.append(dict(reference=ref,reference_lr=lr,paired_seed_deltas=ds,mean_delta=mean(ds.values()),supported=mean(ds.values())<-.005 and max(ds.values())<0))
    eligible=all(v['supported'] for v in contrasts);winner=best if eligible else 'incumbent_unchanged'
    aliases={reference:'layer_affine'} if choices['layer_affine']['lr']==incumbent_lr else {}
    conf_jobs=[]
    for width in [64,256]:
        for i,seed in enumerate(seeds):
            fs=families+([] if aliases else [reference])
            for f in (fs if i%2==0 else fs[::-1]):
                conf_jobs.append([f,incumbent_lr if f==reference else choices[f]['lr'],width,seed])
    manifest=dict(jobs=conf_jobs,reference_aliases=aliases,total_distinct_trials=18+len(conf_jobs),total_model_evaluations=2*(18+len(conf_jobs)))
    assert selection['choices']==choices and selection['best_selected_arm']==best
    assert selection['local_contrasts']==contrasts and selection['candidate_eligible']==eligible and selection['winner']==winner
    assert selection['plan_sha256']==digest(p) and selection['local_evidence_sha256']==digest(screen)
    assert selection['confirmation']==manifest
    assert len(rows)==report['physical_training_trajectories']==manifest['total_distinct_trials']
    assert report['model_evaluations']==2*len(rows) and report['reference_aliases']==aliases
    assert [[r[k] for k in ['family','lr','width','seed']] for r in conf]==conf_jobs
    assert [r['stage'] for r in conf]==['confirmation']*len(conf)
    expanded=list(conf)
    for alias,family in aliases.items():expanded += [{**r,'family':alias} for r in conf if r['family']==family]
    pairs=[('layer_no_bias','layer_affine'),('rms','layer_affine'),('rms','layer_no_bias'),('layer_affine',reference),('layer_no_bias',reference),('rms',reference)]
    assert len(report['comparisons'])==12
    assert [(v['width'],v['candidate'],v['reference']) for v in report['comparisons']]==[(w,a,b) for w in [64,256] for a,b in pairs]
    required=[reference] if winner=='layer_affine' else [reference,'layer_affine'];adoption=[]
    for c in report['comparisons']:
        panel = {f:{r['seed']:r for r in expanded if r['width']==c['width'] and r['family']==f} for f in [c['candidate'],c['reference']]}
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
        if c['candidate']==winner and c['reference'] in required: adoption.append(supported)
    assert report['quality_decision']==dict(frozen_screen_winner=winner,
        qualifies_for_incumbent_replacement=winner!='incumbent_unchanged' and len(adoption)==2*len(required) and all(adoption))
    fixed=[]
    for lr in rates:
        for a,b in pairs[:3]:
            item=dict(lr=lr,candidate=a,reference=b)
            for raw,key in [(False,'primary'),(True,'raw')]:
                panels={f:{r['seed']:endpoint(r,raw)['mean_ce'] for r in screen if r['family']==f and r['lr']==lr} for f in [a,b]}
                deltas={str(s):panels[a][s]-panels[b][s] for s in screens}
                item[key]=dict(mean_delta=mean(deltas.values()),paired_seed_deltas=deltas)
            fixed.append(item)
    assert report['fixed_lr_screen']==fixed
    return dict(status='report_arithmetic_verified',physical_trajectories=len(rows),correlated_outputs=2*len(rows),quality_decision=report['quality_decision'])
