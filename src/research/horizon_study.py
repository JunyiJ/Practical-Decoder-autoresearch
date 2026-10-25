"""Finite fresh-horizon/cooldown study, gated on final reviewed averaging adoption."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
from . import averaging_study as averaging
from . import optimizer_study as engine
from .capacity_study import BOOTSTRAP as RAW_BOOTSTRAP
from .context_interaction import adapter as raw_adapter
from .coverage_eval import paired_blocks, validate_evaluation
from .protocol import digest,file_digest
from .runner import validate_result
from .team.suite import validate_suite,_request,_launch,_fingerprint,source_digest

CELLS={'control3000':(3000,2000),'long6000_stretched':(6000,4000),'long6000_fixedtail':(6000,5000)}
SCREEN_SEEDS=[2161,2162]
CONFIRM_SEEDS=[2171,2172,2173]
WIDTHS=[64,256]
BOOTSTRAP=RAW_BOOTSTRAP+'''
if req['config']['training'].get('weight_averaging') is not None:
    from src.train.weight_averaging import export_averaged_checkpoint
    from src.research.averaging_eval import evaluate_averaged_job
    raw=json.loads(result_path.read_text());parent=Path(raw['final_checkpoint']).resolve()
    parent_sha=file_digest(parent);raw['averaged_outputs']={}
    for half_life in req['config']['training']['weight_averaging']['half_lives']:
        target=parent.parent/f'average_half_life_{half_life}.pt'
        metadata=export_averaged_checkpoint(parent,target,half_life)
        avg_job=dict(checkpoint=str(target),checkpoint_sha256=metadata['checkpoint_sha256'],
            parent_checkpoint=str(parent),parent_sha256=parent_sha,half_life=half_life,
            completed_steps=req['config']['training']['max_iters'],data_sha256=job['data_sha256'],contexts=[128])
        metadata['coverage']=evaluate_averaged_job(avg_job)
        raw['averaged_outputs'][str(half_life)]=metadata
        atomic_json(result_path,raw)
'''


def reviewed_averaging(path):
    path=Path(path);review=json.loads(path.read_text());run=path.parent
    if review.get('status')!='verified' or review.get('distinct_trajectories')!=8 or review.get('distinct_checkpoints')!=32:
        raise ValueError('Final independently verified averaging review required')
    p=json.loads((run/'plan.json').read_text());averaging.validate_plan(p)
    report=json.loads((run/'report.json').read_text());selection=json.loads((run/'selection.json').read_text())
    if (digest(p)!=review['plan_sha256'] or p['git_commit']!=review['source_commit']
            or p['source_sha256']!=review['source_sha256'] or file_digest(run/'report.json')!=review['report_sha256']
            or file_digest(run/'selection.json')!=review['selection_sha256'] or report['plan_sha256']!=digest(p)
            or report['selection']!=selection or p['recipe']!=dict(family='dense4',learning_rate=.005,schedule=averaging.capacity.schedule_spec('linear_cooldown'))):
        raise ValueError('Averaging predecessor lineage or backbone changed')
    chosen=selection['half_life'];matches=[c for c in report['comparisons'] if c['half_life']==chosen]
    if chosen not in averaging.HALF_LIVES or len(matches)!=2 or {c['width'] for c in matches}!={64,256}:
        raise ValueError('Incomplete frozen averaging confirmation')
    checks=[]
    for c in matches:
        v=c['sensitivity'];ds=v[0]['paired_seed_deltas']
        if len(v)!=3 or set(ds)!={'2151','2152','2153'} or any(not math.isfinite(x) for x in ds.values()):
            raise ValueError('Averaging confirmation panel changed')
        if not math.isclose(v[0]['mean_delta'],mean(ds.values()),abs_tol=1e-12):raise ValueError('Averaging paired mean changed')
        checks.append(mean(ds.values())<-.002 and all(x<0 for x in ds.values()) and all(x['conditional_text_bootstrap_95pct'][1]<0 for x in v))
    sd=selection['paired_seed_deltas']
    if set(sd)!={'2141','2142'}:raise ValueError('Averaging screen seeds changed')
    local=mean(sd.values())<-.002 and all(x<0 for x in sd.values())
    if selection['screen_eligible']!=local:raise ValueError('Averaging local eligibility changed')
    decision=dict(frozen_half_life=chosen,qualifies_for_incumbent_replacement=local and all(checks))
    if review['quality_decision']!=decision or report['quality_decision']!=decision:
        raise ValueError('Final averaging adoption disagrees with evidence')
    config=copy.deepcopy(p['config']);config['training'].pop('weight_averaging')
    return chosen if decision['qualifies_for_incumbent_replacement'] else None,config


def schedule(family):
    if family not in CELLS:raise ValueError('Unregistered horizon recipe')
    total,start=CELLS[family]
    return dict(kind='linear_cooldown',total_steps=total,start_step=start,min_ratio=.1)


def local_jobs(p):
    return [(f,.005,64,s) for i,s in enumerate(SCREEN_SEEDS) for f in (list(CELLS) if i%2==0 else list(CELLS)[::-1])]


def confirmation_jobs(p):
    return [[f,.005,w,s] for w in WIDTHS for i,s in enumerate(CONFIRM_SEEDS) for f in (list(CELLS) if i%2==0 else list(CELLS)[::-1])]


def validate_plan(p):
    expected=dict(study='horizon-quality-v1',families=list(CELLS),learning_rates=[.005],seeds=SCREEN_SEEDS,
        confirmation_seeds=CONFIRM_SEEDS,widths=WIDTHS,eval_seed=2026,min_improvement=.005,max_trials=24,
        trial_seconds=2400,session_seconds=21600,max_disk_mb=4096,max_process_rss_mb=4096,max_mps_driver_mb=6144,
        evaluation_targets=111538,bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
        primary_metric='full_validation_context128',horizon_cells={k:dict(total_steps=v[0],cooldown_start=v[1]) for k,v in CELLS.items()})
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen horizon protocol changed')
    h=p['returned_half_life']
    if h is not None and (type(h) is not int or h not in averaging.HALF_LIVES):raise ValueError('Invalid fixed returned-weight policy')
    if p['total_model_evaluations']!=(48 if h is not None else 24):raise ValueError('Returned-output budget changed')
    m,t,d=p['config']['model'],p['config']['training'],p['config']['data']
    em=dict(dim=64,n_heads=2,vocab_size=1,n_layers=4,ffn='swiglu',ffn_bias=False,mlp_type='dense',attn_type='mha',
        parameterization='mup',base_width=64,block_size=128,rope=1,norm_type='rmsnorm',dropout=0.,attn_dropout=0.,
        attention_width_policy='mha_head32',ffn_width_policy='swiglu_21_8',depth_width_policy='depth4_unscaled')
    et=dict(batch_size=8,max_iters=3000,learning_rate=.005,weight_decay=.00006,optimizer='muon',decay_mode='independent',
        grad_clip=1.,eval_every=300,eval_iters=30,device='mps',cpu_threads=4,batch_layout='paired_blocks512',lr_schedule=schedule('control3000'))
    if m!=em or t!=et:raise ValueError('Fixed backbone or control changed')
    if set(d)!={'path','train_fraction','val_fraction'} or d['train_fraction']!=.8 or d['val_fraction']!=.1:
        raise ValueError('Reserved corpus split changed')
    if p.get('screen_jobs')!=[list(x) for x in local_jobs(p)]:raise ValueError('Screen manifest changed')
    return p


def make_plan(root,prior_review):
    half_life,config=reviewed_averaging(prior_review);p=engine.make_plan(root,config)
    p.update(study='horizon-quality-v1',families=list(CELLS),learning_rates=[.005],seeds=SCREEN_SEEDS,
        confirmation_seeds=CONFIRM_SEEDS,widths=WIDTHS,eval_seed=2026,min_improvement=.005,max_trials=24,
        trial_seconds=2400,session_seconds=21600,max_disk_mb=4096,max_process_rss_mb=4096,max_mps_driver_mb=6144,
        evaluation_targets=111538,bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
        primary_metric='full_validation_context128',horizon_cells={k:dict(total_steps=v[0],cooldown_start=v[1]) for k,v in CELLS.items()},
        returned_half_life=half_life,total_model_evaluations=48 if half_life is not None else 24,
        predecessor_review=dict(path=str(Path(prior_review).resolve()),sha256=file_digest(Path(prior_review))),
        selection='Six width64 screens freeze eligible longer-horizon winner; confirm all three recipes on freshseeds64/256. No LR tuning or posthoc alternative promotion.',
        scope='Fresh recipes with different horizons/tokens and cumulative decay; two6000arms share budget and differ in cooldown shape. Adaptive validation, not holdout, equalcompute or optimal horizon/LR transfer.')
    p['screen_jobs']=[list(x) for x in local_jobs(p)];validate_plan(p)
    p['runtime']=_fingerprint(suite_spec(p,'control3000',.005))['runtime'];p['worker_bootstrap_sha256']=digest(BOOTSTRAP)
    return p


def suite_spec(p,family,lr):
    validate_plan(p)
    if family not in CELLS or lr!=.005:raise ValueError('Unregistered fixed recipe')
    cfg=copy.deepcopy(p['config']);t=cfg['training'];total,start=CELLS[family]
    t.update(max_iters=total,eval_every=300,lr_schedule=schedule(family))
    if p['returned_half_life'] is not None:
        t['weight_averaging']=dict(kind='spaced_tail_ema',total_steps=total,start_step=start,interval=10,half_lives=[p['returned_half_life']])
    arm=dict(root=p['source_root'],source_sha256=p['source_sha256'],config=cfg)
    return validate_suite(dict(version=1,hypothesis='Fixed longer training budget and cooldown shape versus fresh3000control',
        baseline=arm,candidate=copy.deepcopy(arm),targets=[dict(name=f'width-{w}',width=w,n_heads=w//32) for w in WIDTHS],
        seeds=SCREEN_SEEDS+CONFIRM_SEEDS,eval_seed=2026,min_improvement=.005,audit_test=False,data_sha256=p['data_sha256'],
        limits=dict(max_trials=24,trial_seconds=2400,session_seconds=21600,max_disk_mb=4096,max_failures=1,max_consecutive_failures=1)))


def endpoint(p,row,raw=False):
    h=p['returned_half_life']
    return row['coverage']['128'] if raw or h is None else row['averaged_outputs'][str(h)]['coverage']['evaluations']['128']


def validate_rows(p,rows,jobs,stage):
    actual=[(r['family'],r['lr'],r['width'],r['seed']) for r in rows]
    if sorted(actual)!=sorted(map(tuple,jobs)) or any(r['status']!='completed' or r['stage']!=stage for r in rows):raise ValueError('Incomplete, duplicate or extra trajectory panel')
    for r in rows:
        outputs=r.get('averaged_outputs',{})
        if set(outputs)!=(set() if p['returned_half_life'] is None else {str(p['returned_half_life'])}):raise ValueError('Returned-weight panel changed')
        validate_evaluation(endpoint(p,r),p['evaluation_targets']);validate_evaluation(endpoint(p,r,True),p['evaluation_targets'])


def freeze_selection(p,rows):
    validate_plan(p);validate_rows(p,rows,local_jobs(p),'local');refs={r['seed']:r for r in rows if r['family']=='control3000'};scores={};eligible=[]
    for f in CELLS:
        panel=[r for r in rows if r['family']==f];ds={str(r['seed']):endpoint(p,r)['mean_ce']-endpoint(p,refs[r['seed']])['mean_ce'] for r in panel}
        ok=f!='control3000' and mean(ds.values())<-.005 and all(v<0 for v in ds.values())
        scores[f]=dict(mean_ce=mean(endpoint(p,r)['mean_ce'] for r in panel),paired_seed_deltas=ds,mean_delta=mean(ds.values()),eligible=ok)
        if ok:eligible.append(f)
    winner=min(eligible,key=lambda f:(scores[f]['mean_ce'],f)) if eligible else 'control3000'
    return dict(plan_sha256=digest(p),local_evidence_sha256=digest(rows),winner=winner,scores=scores,
        confirmation=dict(jobs=confirmation_jobs(p),total_distinct_trials=24,total_model_evaluations=p['total_model_evaluations']),
        selected_by='Only width64screen primary endpoint, mean gain>.005 andbothseeds; tiesfamilyname. Allrecipes confirmed; frozenwinneronly canadopt.')


def summarize(p,state):
    validate_plan(p);rows=state['records'];screen=[r for r in rows if r['stage']=='local'];conf=[r for r in rows if r['stage']=='confirmation']
    if len(rows)!=24 or freeze_selection(p,screen)!=state['selection']:raise ValueError('Frozen horizon selection or total changed')
    validate_rows(p,conf,confirmation_jobs(p),'confirmation');comparisons=[]
    for w in WIDTHS:
        for a,b in [('long6000_stretched','control3000'),('long6000_fixedtail','control3000'),('long6000_fixedtail','long6000_stretched')]:
            def panel(f,raw):return {r['seed']:endpoint(p,r,raw) for r in conf if r['family']==f and r['width']==w}
            sensitivity=[paired_blocks(panel(a,False),panel(b,False),n) for n in p['bootstrap_blocks']];v=sensitivity[0]
            ok=v['mean_delta']<-.005 and all(x<0 for x in v['paired_seed_deltas'].values()) and all(x['conditional_text_bootstrap_95pct'][1]<0 for x in sensitivity)
            comparisons.append(dict(width=w,candidate=a,reference=b,sensitivity=sensitivity,quality_improvement=ok,
                raw_sensitivity=[paired_blocks(panel(a,True),panel(b,True),n) for n in p['bootstrap_blocks']]))
    winner=state['selection']['winner'];chosen=[c for c in comparisons if c['candidate']==winner and c['reference']=='control3000']
    adopt=winner!='control3000' and len(chosen)==2 and all(c['quality_improvement'] for c in chosen)
    return dict(plan_sha256=digest(p),selection=state['selection'],records=rows,comparisons=comparisons,
        physical_training_trajectories=24,model_evaluations=p['total_model_evaluations'],holdout_consumed=False,
        quality_decision=dict(frozen_screen_winner=winner,qualifies_for_incumbent_replacement=adopt),adoption_requires_completion_review=True,scope=p['scope'])


def adapter(p,raw,checked,request,directory):
    extra=raw_adapter(p,raw,checked,request,directory);h=p['returned_half_life'];outputs=raw.get('averaged_outputs',{})
    if set(outputs)!=(set() if h is None else {str(h)}):raise ValueError('Unexpected returned-weight outputs')
    if h is None:return extra
    item=outputs[str(h)];parent=Path(checked['checkpoint']).resolve();path=Path(item['checkpoint']).resolve();cov=item['coverage'];meta=item['averaged_weights'];t=request['config']['training'];spec=t['weight_averaging']
    if (path!=parent.parent/f'average_half_life_{h}.pt' or file_digest(path)!=item['checkpoint_sha256']
            or item['parent_checkpoint']!=dict(path=str(parent),sha256=checked['checkpoint_sha256'])
            or set(meta)!={'spec','half_life','snapshots','completed_steps','parameter_names','purpose'}
            or not meta['parameter_names'] or len(set(meta['parameter_names']))!=len(meta['parameter_names'])
            or meta['spec']!=spec or meta['half_life']!=h or meta['completed_steps']!=t['max_iters']
            or meta['snapshots']!=1+(t['max_iters']-spec['start_step'])//spec['interval'] or meta['purpose']!='inference_only_not_optimizer_resume'
            or cov['checkpoint_sha256']!=item['checkpoint_sha256'] or cov['parent_checkpoint_sha256']!=checked['checkpoint_sha256']
            or cov['half_life']!=h or cov['holdout_consumed'] is not False or cov['device']!='mps'
            or cov['torch']!=raw['runtime']['torch'] or set(cov['evaluations'])!={'128'}):raise ValueError('Derived horizon output contract changed')
    validate_evaluation(cov['evaluations']['128'],p['evaluation_targets'])
    return {**extra,'metric':cov['evaluations']['128']['mean_ce'],'raw_metric':extra['metric'],'averaged_outputs':outputs}


def verify_lineage(p):
    validate_plan(p);root=Path(p['source_root']);prior=p['predecessor_review']
    if file_digest(Path(prior['path']))!=prior['sha256']:raise ValueError('Predecessor review changed')
    h,cfg=reviewed_averaging(prior['path'])
    if h!=p['returned_half_life'] or cfg!=p['config']:raise ValueError('Final adopted returned-weight policy changed')
    if source_digest(root)!=p['source_sha256'] or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=p['git_commit']:raise ValueError('Source contract changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Dirty source')
    if file_digest(Path(p['config']['data']['path']))!=p['data_sha256']:raise ValueError('Corpus changed')
    if file_digest(Path(p['knowledge_snapshot']['path']))!=p['knowledge_snapshot']['sha256']:raise ValueError('Full bank snapshot changed')
    for a in p['artifacts']:
        if file_digest(Path(a['path']))!=a['sha256']:raise ValueError('Prerequisite artifact changed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Unset MPS fallback')
    if digest(BOOTSTRAP)!=p['worker_bootstrap_sha256'] or _fingerprint(suite_spec(p,'control3000',.005))['runtime']!=p['runtime']:raise ValueError('Worker/runtime changed')


def run(p,output):
    verify_lineage(p);output=Path(output)
    if json.loads((output/'plan.json').read_text())!=p:raise ValueError('Run plan changed')
    def check(records):
        for r in records:
            if r['status']!='completed':raise ValueError('No automatic retry')
            d=output/'trials'/r['trial_id']
            for name in ['request','result']:
                if file_digest(d/f'{name}.json')!=r[name+'_sha256']:raise ValueError('Saved evidence changed')
            req=json.loads((d/'request.json').read_text())
            if req!=_request(suite_spec(p,r['family'],r['lr']),r,d):raise ValueError('Saved request changed')
            raw=json.loads((d/'result.json').read_text());cfg=req['config'];checked=validate_result(raw,req,dict(device='mps',training={**cfg['training'],'block_size':128}),d)
            if any(r.get(k)!=v for k,v in {**checked,**adapter(p,raw,checked,req,d)}.items()):raise ValueError('Saved result changed')
    if (output/'state.json').exists():check(json.loads((output/'state.json').read_text())['records'])
    def spec(plan,f,lr):verify_lineage(plan);return suite_spec(plan,f,lr)
    def launch(*a,**kw):return _launch(*a,**kw,bootstrap=BOOTSTRAP)
    def adapt(*a):verify_lineage(p);return adapter(p,*a)
    def report(plan,state):verify_lineage(plan);check(state['records']);return summarize(plan,state)
    return engine.run(p,output,suite_builder=spec,job_builder=local_jobs,selection_builder=freeze_selection,report_builder=report,
        launcher=launch,result_adapter=adapt,completion_jobs=lambda plan,selection:selection['confirmation']['jobs'])


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-training',action='store_true');args=parser.parse_args()
    if not args.allow_training:parser.error('Explicit --allow-training required')
    run(json.loads(args.plan.read_text()),args.output)

if __name__=='__main__':main()
