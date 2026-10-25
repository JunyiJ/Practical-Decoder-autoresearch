"""Eight fresh trajectories with paired raw and spaced-tail averaged endpoints.

Training is serial and finite. Only a width64-screened average may be adopted;
all registered endpoints are evaluated on all fresh confirmation trajectories.
"""
import argparse
import copy
import json
import math
import os
from pathlib import Path
from statistics import mean
import subprocess
from . import optimizer_study as engine
from . import capacity_study as capacity
from .coverage_eval import paired_blocks, validate_evaluation
from .context_interaction import adapter as raw_coverage_adapter
from .protocol import digest, file_digest
from .runner import validate_result
from .team.suite import validate_suite, _launch, _request, _fingerprint, source_digest

SCREEN_SEEDS=[2141,2142]
CONFIRM_SEEDS=[2151,2152,2153]
WIDTHS=[64,256]
HALF_LIVES=[10,50,200]
AVERAGING=dict(kind='spaced_tail_ema',total_steps=3000,start_step=2000,interval=10,half_lives=HALF_LIVES)
BOOTSTRAP=capacity.BOOTSTRAP+'''
from src.train.weight_averaging import export_averaged_checkpoint
from src.research.averaging_eval import evaluate_averaged_job
raw=json.loads(result_path.read_text());parent=Path(raw['final_checkpoint']).resolve()
parent_sha=file_digest(parent);data_sha=job['data_sha256'];raw['averaged_outputs']={}
for half_life in req['config']['training']['weight_averaging']['half_lives']:
    output=parent.parent/f'average_half_life_{half_life}.pt'
    metadata=export_averaged_checkpoint(parent,output,half_life)
    job=dict(checkpoint=str(output),checkpoint_sha256=metadata['checkpoint_sha256'],
        parent_checkpoint=str(parent),parent_sha256=parent_sha,half_life=half_life,
        completed_steps=req['config']['training']['max_iters'],data_sha256=data_sha,contexts=[128])
    metadata['coverage']=evaluate_averaged_job(job)
    raw['averaged_outputs'][str(half_life)]=metadata
    atomic_json(result_path,raw)
'''


def reviewed_capacity(review_path):
    """Resolve a fixed recipe from audited completion, never from partial results."""
    path=Path(review_path);review=json.loads(path.read_text())
    if review.get('status')!='verified' or review.get('distinct_checkpoints') not in (54,60):
        raise ValueError('Completed verified cooldown capacity predecessor required')
    report_path=path.parent/'report.json';selection_path=path.parent/'selection.json'
    if file_digest(report_path)!=review['report_sha256'] or file_digest(selection_path)!=review['selection_sha256']:
        raise ValueError('Predecessor report or selection changed')
    report=json.loads(report_path.read_text());selection=json.loads(selection_path.read_text())
    plan=json.loads((path.parent/'plan.json').read_text());capacity.validate_plan(plan)
    if (digest(plan)!=review['plan_sha256'] or report['plan_sha256']!=digest(plan)
            or report['selection']!=selection or plan['git_commit']!=review['source_commit']
            or plan['source_sha256']!=review['source_sha256']
            or plan['incumbent_schedule']!={'kind':'linear_cooldown','learning_rate':.005}):
        raise ValueError('Capacity predecessor lineage or fixed cooldown changed')
    winner=selection['winner'];checks=[]
    if winner!='historical_incumbent_unchanged':
        if winner not in ('wide4','dense6'):raise ValueError('Unknown capacity screen winner')
        for width in WIDTHS:
            for reference in ('dense4','incumbent_reference','deep_reference'):
                matches=[c for c in report['comparisons'] if c['width']==width and c['candidate']==winner and c['reference']==reference]
                if len(matches)!=1:raise ValueError('Incomplete capacity confirmation panel')
                v=matches[0]['sensitivity'][0];ds=v['paired_seed_deltas']
                if set(ds)!={'2131','2132','2133'} or any(not math.isfinite(x) for x in ds.values()):
                    raise ValueError('Capacity confirmation seeds changed')
                if not math.isclose(v['mean_delta'],mean(ds.values()),abs_tol=1e-12):
                    raise ValueError('Capacity mean differs from paired seeds')
                checks.append(v['mean_delta']<-.005 and all(x<0 for x in ds.values()))
    qualified=bool(checks) and all(checks);decision=review['quality_decision']
    if decision['frozen_screen_winner']!=winner or decision['qualifies_for_incumbent_replacement']!=qualified:
        raise ValueError('Reviewed adoption disagrees with frozen capacity evidence')
    family=winner if qualified else 'dense4'
    lr=selection['choices'][winner]['lr'] if qualified else .005
    config=capacity.suite_spec(plan,family,lr)['candidate']['config']
    return dict(family=family,learning_rate=lr,schedule=copy.deepcopy(config['training']['lr_schedule'])),config


def local_jobs(p):
    return [('trajectory',p['recipe']['learning_rate'],64,seed) for seed in SCREEN_SEEDS]


def confirmation_jobs(p):
    return [['trajectory',p['recipe']['learning_rate'],w,seed] for w in WIDTHS for seed in CONFIRM_SEEDS]


def validate_plan(p):
    expected=dict(study='averaging-quality-v1',families=['trajectory'],seeds=SCREEN_SEEDS,
        confirmation_seeds=CONFIRM_SEEDS,widths=WIDTHS,eval_seed=2026,min_improvement=.002,
        max_trials=8,trial_seconds=1800,session_seconds=14400,max_disk_mb=2048,
        max_process_rss_mb=4096,max_mps_driver_mb=6144,evaluation_targets=111538,
        total_training_targets=3072000,model_outputs_per_trajectory=4,total_model_evaluations=32,
        bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
        primary_metric='full_validation_context128',averaging=AVERAGING)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Frozen averaging protocol changed')
    recipe=p['recipe']
    if (set(recipe)!={'family','learning_rate','schedule'} or recipe['family'] not in capacity.CELLS
            or recipe['learning_rate'] not in capacity.LRS or recipe['schedule']!=capacity.schedule_spec('linear_cooldown')
            or p['learning_rates']!=[recipe['learning_rate']]):raise ValueError('Fixed adopted recipe changed')
    m,t,d=p['config']['model'],p['config']['training'],p['config']['data'];family=recipe['family']
    em=dict(dim=64,n_heads=2,vocab_size=1,n_layers=6 if family=='dense6' else 4,
        ffn='swiglu',ffn_bias=False,mlp_type='dense',attn_type='mha',parameterization='mup',base_width=64,
        block_size=128,rope=1,norm_type='rmsnorm',dropout=0.,attn_dropout=0.,attention_width_policy='mha_head32',
        ffn_width_policy='swiglu_21_4' if family=='wide4' else 'swiglu_21_8',
        depth_width_policy={'dense4':'depth4_unscaled','wide4':'depth4_wide_unscaled','dense6':'depth6_dense_unscaled'}[family])
    et=dict(batch_size=8,max_iters=3000,learning_rate=recipe['learning_rate'],weight_decay=.00006,
        optimizer='muon',decay_mode='independent',grad_clip=1.,eval_every=300,eval_iters=30,device='mps',
        cpu_threads=4,batch_layout='paired_blocks512',lr_schedule=recipe['schedule'],weight_averaging=AVERAGING)
    if m!=em or t!=et:raise ValueError('Resolved backbone/training contract changed')
    if d.get('train_fraction')!=.8 or d.get('val_fraction')!=.1 or d.get('tokenizer','char')!='char':
        raise ValueError('Reserved corpus split changed')
    if p.get('screen_jobs')!=[list(x) for x in local_jobs(p)]:raise ValueError('Screen manifest changed')
    return p


def make_plan(root,predecessor_review):
    recipe,config=reviewed_capacity(predecessor_review)
    config['training']['weight_averaging']=copy.deepcopy(AVERAGING)
    p=engine.make_plan(root,config)
    p.update(study='averaging-quality-v1',recipe=recipe,averaging=copy.deepcopy(AVERAGING),
        families=['trajectory'],learning_rates=[recipe['learning_rate']],seeds=SCREEN_SEEDS,
        confirmation_seeds=CONFIRM_SEEDS,widths=WIDTHS,eval_seed=2026,min_improvement=.002,
        max_trials=8,trial_seconds=1800,session_seconds=14400,max_disk_mb=2048,
        max_process_rss_mb=4096,max_mps_driver_mb=6144,evaluation_targets=111538,
        total_training_targets=3072000,model_outputs_per_trajectory=4,total_model_evaluations=32,
        bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
        primary_metric='full_validation_context128',
        predecessor_review=dict(path=str(Path(predecessor_review).resolve()),sha256=file_digest(Path(predecessor_review))),
        selection='Select one half-life by mean width64 screening CE, ties shorter; freeze before fresh confirmation. Every registered output is confirmed. Only locally eligible frozen choice can adopt.',
        scope='Eight physical fresh training trajectories, raw plus three correlated averaged endpoints each. Same online training tokens and architecture; averaging/evaluation have measured overhead. Adaptive validation, not holdout or seed-population inference.')
    p['screen_jobs']=[list(x) for x in local_jobs(p)];validate_plan(p)
    p['runtime']=_fingerprint(suite_spec(p,'trajectory',recipe['learning_rate']))['runtime']
    p['worker_bootstrap_sha256']=digest(BOOTSTRAP)
    return p


def suite_spec(p,family,lr):
    validate_plan(p)
    if family!='trajectory' or lr!=p['recipe']['learning_rate']:raise ValueError('Unregistered trajectory recipe')
    arm=dict(root=p['source_root'],source_sha256=p['source_sha256'],config=copy.deepcopy(p['config']))
    return validate_suite(dict(version=1,hypothesis='Spaced tail averaging against the same trajectory raw endpoint',
        baseline=arm,candidate=copy.deepcopy(arm),targets=[dict(name=f'width-{w}',width=w,n_heads=w//32) for w in WIDTHS],
        seeds=SCREEN_SEEDS+CONFIRM_SEEDS,eval_seed=2026,min_improvement=.002,audit_test=False,data_sha256=p['data_sha256'],
        limits=dict(max_trials=8,trial_seconds=1800,session_seconds=14400,max_disk_mb=2048,max_failures=1,max_consecutive_failures=1)))


def endpoint(row,key):
    return row['coverage']['128'] if key=='raw' else row['averaged_outputs'][str(key)]['coverage']['evaluations']['128']


def freeze_selection(p,records):
    validate_plan(p);actual=[(r['family'],r['lr'],r['width'],r['seed']) for r in records]
    if set(actual)!=set(local_jobs(p)) or len(actual)!=2 or any(r['status']!='completed' for r in records):
        raise ValueError('Incomplete or extra averaging screens')
    for r in records:
        if set(r['averaged_outputs'])!={str(h) for h in HALF_LIVES}:raise ValueError('Screen average set changed')
        for key in ['raw']+HALF_LIVES:validate_evaluation(endpoint(r,key),p['evaluation_targets'])
    scores={str(h):mean(endpoint(r,h)['mean_ce'] for r in records) for h in HALF_LIVES}
    winner=min(HALF_LIVES,key=lambda h:(scores[str(h)],h))
    ds={str(r['seed']):endpoint(r,winner)['mean_ce']-endpoint(r,'raw')['mean_ce'] for r in records}
    eligible=mean(ds.values())<-.002 and all(x<0 for x in ds.values())
    return dict(plan_sha256=digest(p),local_evidence_sha256=digest(records),half_life=winner,
        screen_eligible=eligible,mean_ce_by_half_life=scores,mean_delta=mean(ds.values()),paired_seed_deltas=ds,
        confirmation=dict(jobs=confirmation_jobs(p),total_distinct_trials=8,total_model_evaluations=32),
        selected_by='Width64 only: minimum mean averaged CE, ties shorter half-life; mean gain>.002 and both seeds improve for local eligibility.')


def summarize(p,state):
    validate_plan(p);records=state['records'];screen=[r for r in records if r['stage']=='local']
    if freeze_selection(p,screen)!=state['selection']:raise ValueError('Frozen average choice changed')
    rows=[r for r in records if r['stage']=='confirmation']
    actual=[(r['family'],r['lr'],r['width'],r['seed']) for r in rows]
    if len(records)!=8 or sorted(actual)!=sorted(map(tuple,confirmation_jobs(p))) or any(r['status']!='completed' for r in rows):
        raise ValueError('Incomplete or extra confirmation trajectories')
    for r in rows:
        if set(r['averaged_outputs'])!={str(h) for h in HALF_LIVES}:raise ValueError('Confirmation average set changed')
        for key in ['raw']+HALF_LIVES:validate_evaluation(endpoint(r,key),p['evaluation_targets'])
    comparisons=[]
    for width in WIDTHS:
        panel=[r for r in rows if r['width']==width]
        for h in HALF_LIVES:
            a={r['seed']:endpoint(r,h) for r in panel};b={r['seed']:endpoint(r,'raw') for r in panel}
            sensitivity=[paired_blocks(a,b,n) for n in p['bootstrap_blocks']];v=sensitivity[0]
            supported=(v['mean_delta']<-.002 and all(d<0 for d in v['paired_seed_deltas'].values())
                and all(x['conditional_text_bootstrap_95pct'][1]<0 for x in sensitivity))
            comparisons.append(dict(width=width,half_life=h,reference='same_trajectory_raw_endpoint',
                sensitivity=sensitivity,quality_improvement=supported))
    selected=[c for c in comparisons if c['half_life']==state['selection']['half_life']]
    eligible=state['selection']['screen_eligible'] and len(selected)==2 and all(c['quality_improvement'] for c in selected)
    return dict(plan_sha256=digest(p),selection=state['selection'],records=records,comparisons=comparisons,
        physical_training_trajectories=8,model_evaluations=32,holdout_consumed=False,
        quality_decision=dict(frozen_half_life=state['selection']['half_life'],qualifies_for_incumbent_replacement=eligible),
        adoption_requires_completion_review=True,scope=p['scope'])


def adapter(p,raw,checked,request,directory):
    extra=raw_coverage_adapter(p,raw,checked,request,directory)
    outputs=raw.get('averaged_outputs')
    if not isinstance(outputs,dict) or set(outputs)!={str(h) for h in HALF_LIVES}:raise ValueError('Missing averaged outputs')
    parent=Path(checked['checkpoint']).resolve()
    spec=request['config']['training']['weight_averaging'];steps=request['config']['training']['max_iters']
    snapshots=1+(steps-spec['start_step'])//spec['interval']
    for h in HALF_LIVES:
        item=outputs[str(h)];path=Path(item['checkpoint']).resolve();cov=item['coverage'];meta=item['averaged_weights']
        if (set(meta)!={'spec','half_life','snapshots','completed_steps','parameter_names','purpose'}
                or not meta['parameter_names'] or len(set(meta['parameter_names']))!=len(meta['parameter_names'])
                or any(not isinstance(name,str) for name in meta['parameter_names'])
                or path!=parent.parent/f'average_half_life_{h}.pt' or file_digest(path)!=item['checkpoint_sha256']
                or item['parent_checkpoint']!=dict(path=str(parent),sha256=checked['checkpoint_sha256'])
                or meta['spec']!=spec or meta['half_life']!=h or meta['snapshots']!=snapshots or meta['completed_steps']!=steps
                or meta['purpose']!='inference_only_not_optimizer_resume'
                or cov['checkpoint_sha256']!=item['checkpoint_sha256'] or cov['parent_checkpoint_sha256']!=checked['checkpoint_sha256']
                or cov['half_life']!=h or cov['holdout_consumed'] is not False or cov['device']!='mps'
                or cov['torch']!=raw['runtime']['torch'] or set(cov['evaluations'])!={'128'}):
            raise ValueError('Averaged output lineage or scoring contract changed')
        validate_evaluation(cov['evaluations']['128'],p['evaluation_targets'])
    return {**extra,'averaged_outputs':outputs}


def verify_lineage(p):
    validate_plan(p);root=Path(p['source_root']);prior=p['predecessor_review']
    if file_digest(Path(prior['path']))!=prior['sha256']:raise ValueError('Predecessor review changed')
    recipe,config=reviewed_capacity(prior['path']);config['training']['weight_averaging']=copy.deepcopy(AVERAGING)
    if recipe!=p['recipe'] or config!=p['config']:raise ValueError('Resolved adopted recipe changed')
    if source_digest(root)!=p['source_sha256'] or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=p['git_commit']:
        raise ValueError('Source contract changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Dirty source')
    if file_digest(Path(p['config']['data']['path']))!=p['data_sha256']:raise ValueError('Corpus changed')
    if file_digest(Path(p['knowledge_snapshot']['path']))!=p['knowledge_snapshot']['sha256']:raise ValueError('Full knowledge snapshot changed')
    for a in p['artifacts']:
        if file_digest(Path(a['path']))!=a['sha256']:raise ValueError('Prerequisite artifact changed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Unset MPS fallback')
    if p['worker_bootstrap_sha256']!=digest(BOOTSTRAP):raise ValueError('Worker bootstrap changed')
    if _fingerprint(suite_spec(p,'trajectory',p['recipe']['learning_rate']))['runtime']!=p['runtime']:raise ValueError('Runtime changed')


def run(p,output):
    verify_lineage(p);output=Path(output)
    if json.loads((output/'plan.json').read_text())!=p:raise ValueError('Run plan mismatch')
    def check_records(records):
        for r in records:
            if r['status']!='completed':raise ValueError('No automatic retry')
            directory=output/'trials'/r['trial_id']
            for name in ('request','result'):
                if file_digest(directory/f'{name}.json')!=r[name+'_sha256']:raise ValueError('Saved raw evidence changed')
            request=json.loads((directory/'request.json').read_text())
            if request!=_request(suite_spec(p,r['family'],r['lr']),r,directory):raise ValueError('Saved request changed')
            raw=json.loads((directory/'result.json').read_text());cfg=request['config']
            checked=validate_result(raw,request,dict(device='mps',training={**cfg['training'],'block_size':128}),directory)
            if any(r.get(k)!=v for k,v in {**checked,**adapter(p,raw,checked,request,directory)}.items()):raise ValueError('Saved result changed')
    if (output/'state.json').exists():check_records(json.loads((output/'state.json').read_text())['records'])
    def spec(plan,f,lr):verify_lineage(plan);return suite_spec(plan,f,lr)
    def launch(*args,**kwargs):return _launch(*args,**kwargs,bootstrap=BOOTSTRAP)
    def adapt(*args):verify_lineage(p);return adapter(p,*args)
    def report(plan,state):verify_lineage(plan);check_records(state['records']);return summarize(plan,state)
    return engine.run(p,output,suite_builder=spec,job_builder=local_jobs,selection_builder=freeze_selection,
        report_builder=report,launcher=launch,result_adapter=adapt,completion_jobs=lambda plan,selection:selection['confirmation']['jobs'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-training',action='store_true');args=parser.parse_args()
    if not args.allow_training:parser.error('Explicit --allow-training required')
    run(json.loads(args.plan.read_text()),args.output)

if __name__=='__main__':main()
