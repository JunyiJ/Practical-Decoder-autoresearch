"""Bounded dense FFN/depth capacity comparison after reviewed schedule selection.

Full-budget runs, paired training targets, common-context full validation. Uses
historical serial worker limits; no test scoring or historical re-selection.
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
from .coverage_eval import paired_blocks
from .protocol import digest, file_digest
from .team.suite import validate_suite, _launch, _request, _fingerprint, source_digest
from .runner import validate_result
from .context_interaction import BOOTSTRAP as COVERAGE_BOOTSTRAP, validate_coverage, adapter as coverage_adapter

BOOTSTRAP = "from src.research.memory_guard import start_memory_guard\nstart_memory_guard()\n" + COVERAGE_BOOTSTRAP

LRS = [.00125, .0025, .005]
SCREEN_SEEDS = [2121, 2122]
CONFIRM_SEEDS = [2131, 2132, 2133]
WIDTHS = [64, 256]
CELLS = {'dense4': 4, 'wide4': 4, 'dense6': 6}
FIXED = {'gelu_reference': .002, 'shallow_reference': .0025, 'deep_reference': .0025}


def schedule_spec(kind):
    if kind == 'constant':return {'kind':'constant'}
    if kind == 'cosine':return dict(kind=kind,total_steps=3000,min_ratio=.1)
    if kind == 'linear_cooldown':return dict(kind=kind,total_steps=3000,min_ratio=.1,start_step=2000)
    raise ValueError('Unknown reviewed schedule')


def validate_incumbent(incumbent):
    if not isinstance(incumbent,dict) or set(incumbent)!={'kind','learning_rate'}:
        raise ValueError('Malformed reviewed schedule incumbent')
    schedule_spec(incumbent['kind'])
    if incumbent['learning_rate'] not in LRS:
        raise ValueError('Incumbent LR outside bounded grid')
    if incumbent['kind']=='constant' and incumbent['learning_rate']!=.0025:
        raise ValueError('Unchanged incumbent must preserve constant .0025')


def fixed_references(incumbent):
    validate_incumbent(incumbent)
    refs=dict(FIXED)
    if incumbent['kind']!='constant':refs['incumbent_reference']=incumbent['learning_rate']
    return refs


def reference_cells(incumbent):
    validate_incumbent(incumbent)
    return {('deep_reference' if incumbent['kind']=='constant' else 'incumbent_reference'):'dense4'}


def reviewed_schedule(review_path):
    """Resolve one shared schedule only from the completed audited predecessor."""
    path=Path(review_path);review=json.loads(path.read_text())
    if review.get('status')!='verified' or review.get('distinct_checkpoints') not in (48,54):
        raise ValueError('Completed verified dense schedule review required')
    report_path=path.parent/'report.json';selection_path=path.parent/'selection.json';plan_path=path.parent/'plan.json'
    if file_digest(report_path)!=review['report_sha256'] or file_digest(selection_path)!=review['selection_sha256']:
        raise ValueError('Predecessor evidence changed')
    report=json.loads(report_path.read_text());selection=json.loads(selection_path.read_text());plan=json.loads(plan_path.read_text())
    if (plan['study']!='schedule-quality-v1' or plan['schedule_backbone']!='dense'
            or digest(plan)!=review['plan_sha256'] or report['plan_sha256']!=review['plan_sha256']
            or report['selection']!=selection):
        raise ValueError('Schedule predecessor lineage changed')
    from .schedule_study import validate_plan as validate_predecessor
    validate_predecessor(plan)
    if plan['git_commit']!=review['source_commit'] or plan['source_sha256']!=review['source_sha256']:
        raise ValueError('Predecessor source review changed')
    winner=selection['winner'];checks=[]
    if winner!='historical_incumbent_unchanged':
        if winner not in ('cosine','linear_cooldown'):raise ValueError('Unregistered schedule winner')
        for width in WIDTHS:
            for ref in ('constant','deep_reference'):
                matches=[c for c in report['comparisons'] if c['width']==width and c['candidate']==winner and c['reference']==ref]
                if len(matches)!=1:raise ValueError('Incomplete confirmation comparisons')
                v=matches[0]['sensitivity'][0];ds=v['paired_seed_deltas']
                if set(ds)!={'2111','2112','2113'} or not all(math.isfinite(x) for x in ds.values()):
                    raise ValueError('Incomplete confirmation seeds')
                if not math.isclose(mean(ds.values()),v['mean_delta'],rel_tol=0,abs_tol=1e-12):
                    raise ValueError('Invalid confirmation mean')
                checks.append(v['mean_delta']<-.005 and all(x<0 for x in ds.values()))
    qualified=bool(checks) and all(checks);decision=review['quality_decision']
    if decision['frozen_screen_winner']!=winner or decision['qualifies_for_incumbent_replacement']!=qualified:
        raise ValueError('Adoption decision disagrees with completed evidence')
    incumbent=dict(kind=winner,learning_rate=selection['choices'][winner]['lr']) if qualified else dict(kind='constant',learning_rate=.0025)
    validate_incumbent(incumbent)
    return incumbent


def settings(family):
    if family in CELLS:return dict(depth=CELLS[family],attention='mha',context=128,batch=8)
    if family not in set(FIXED)|{'incumbent_reference'}:raise ValueError('Unregistered family')
    return dict(depth=4 if family in {'deep_reference','incumbent_reference'} else 2,attention='mha',context=128,batch=8)


def validate_plan(p):
    expected = dict(study='capacity-quality-v1', families=list(CELLS), learning_rates=LRS,
                    seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS, widths=WIDTHS,
                    eval_seed=2026, min_improvement=.005, max_trials=60,
                    trial_seconds=1800, session_seconds=28800, max_disk_mb=4096,
                    primary_metric='full_validation_context128', quality_margin=.01,
                    bootstrap_blocks=[2048,4096,8192], bootstrap_draws=2000,
                    bootstrap_seed=20261024, evaluation_targets=111538, total_training_targets=3072000,
                    batch_layout='paired_blocks512', capacity_policy='dense4_wide4_dense6_reviewed_schedule_v1', max_process_rss_mb=4096,max_mps_driver_mb=6144)
    if any(p.get(k) != v for k,v in expected.items()):
        raise ValueError('Frozen capacity quality protocol changed')
    fixed_references(p.get('incumbent_schedule'))
    m,t,d = p['config']['model'],p['config']['training'],p['config']['data']
    em = dict(n_layers=2,ffn='gelu',mlp_type='dense',attn_type='mha',parameterization='mup',
              base_width=64,block_size=128,rope=1,norm_type='rmsnorm',dropout=0.,attn_dropout=0.)
    et = dict(batch_size=8,max_iters=3000,learning_rate=.002,weight_decay=.00006,optimizer='muon',
              decay_mode='independent',grad_clip=1.,eval_every=300,eval_iters=30,device='mps',cpu_threads=4)
    if any(m.get(k)!=v for k,v in em.items()) or set(m)-set(em)-{'dim','n_heads','vocab_size'}:
        raise ValueError('Base model changed')
    if any(t.get(k)!=v for k,v in et.items()) or set(t)-set(et):
        raise ValueError('Base training changed')
    if d.get('train_fraction')!=.8 or d.get('val_fraction')!=.1 or d.get('tokenizer','char')!='char':
        raise ValueError('Reserved data split/tokenizer changed')
    if p.get('screen_jobs') != [list(x) for x in local_jobs(p)]:
        raise ValueError('Screen manifest changed')
    return p


def make_plan(root, config, predecessor_review):
    predecessor_review=Path(predecessor_review).resolve()
    incumbent=reviewed_schedule(predecessor_review)
    config=copy.deepcopy(config)
    config['model'].setdefault('attn_dropout',0.)
    from .team.suite import RUNTIME_TRAINING
    for key in RUNTIME_TRAINING:config['training'].pop(key,None)
    p=engine.make_plan(root,config)
    p.update(study='capacity-quality-v1',incumbent_schedule=incumbent,
             predecessor_review=dict(path=str(predecessor_review),sha256=file_digest(predecessor_review)),
             families=list(CELLS),learning_rates=LRS,
             seeds=SCREEN_SEEDS,confirmation_seeds=CONFIRM_SEEDS,widths=WIDTHS,
             max_trials=60,trial_seconds=1800,session_seconds=28800,max_disk_mb=4096,
             primary_metric='full_validation_context128',quality_margin=.01,
             bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
             evaluation_targets=111538,total_training_targets=3072000,
             batch_layout='paired_blocks512',capacity_policy='dense4_wide4_dense6_reviewed_schedule_v1',max_process_rss_mb=4096,max_mps_driver_mb=6144,
             selection='Three independent width64 LR choices on seeds2121/2122; freeze before any confirmation. No pruning or extension. Fixed identical references alias exact recipes.',
             scope='Adaptive development: three equally tuned dense capacity recipes, four-layer standard/wide FFN and six-layer standard FFN, one common reviewed schedule, fresh controls and confirmation, equal3000updates and3072000targets. Same per-update independent decay, different LR integral/update-to-decay ratio. Not optimal LR transfer, equalcompute, holdout or population inference.')
    p['screen_jobs']=[list(x) for x in local_jobs(p)]
    validate_plan(p)
    p['runtime']=_fingerprint(suite_spec(p,'dense4',.0025))['runtime']
    p['worker_bootstrap_sha256']=digest(BOOTSTRAP)
    return p


def suite_spec(p, family, lr):
    validate_plan(p)
    refs=fixed_references(p['incumbent_schedule'])
    if family not in CELLS and family not in refs:raise ValueError('Unregistered family')
    if lr not in ([refs[family]] if family in refs else LRS):
        raise ValueError('Unregistered LR')
    s=settings(family); cfg=copy.deepcopy(p['config']); m,t=cfg['model'],cfg['training']
    m.update(n_layers=s['depth'],attn_type=s['attention'],block_size=s['context'])
    t.update(learning_rate=lr,batch_size=8,batch_layout='paired_blocks512')
    if family!='gelu_reference':
        m.update(ffn='swiglu',ffn_bias=False,ffn_width_policy='swiglu_21_8',
                 attention_width_policy='mha_head32')
    if s['depth']==4:
        m['depth_width_policy']='depth4_unscaled'
    if family=='wide4':
        m.update(ffn_width_policy='swiglu_21_4',depth_width_policy='depth4_wide_unscaled')
    if family=='dense6':
        m['depth_width_policy']='depth6_dense_unscaled'
    kind=p['incumbent_schedule']['kind'] if family in CELLS or family=='incumbent_reference' else 'constant'
    t['lr_schedule']=schedule_spec(kind)
    arm=dict(root=p['source_root'],source_sha256=p['source_sha256'],config=cfg)
    return validate_suite(dict(version=1,hypothesis=f'Dense capacity comparison with one reviewed schedule: {family}',
        baseline=arm,candidate=copy.deepcopy(arm),
        targets=[dict(name=f'width-{w}',width=w,n_heads=w//32) for w in WIDTHS],
        seeds=SCREEN_SEEDS+CONFIRM_SEEDS,eval_seed=2026,min_improvement=.005,audit_test=False,
        data_sha256=p['data_sha256'],limits=dict(max_trials=60,trial_seconds=1800,
        session_seconds=28800,max_disk_mb=4096,max_failures=1,max_consecutive_failures=1)))


def local_jobs(p):
    return [(f,lr,64,s) for lr in LRS for i,s in enumerate(SCREEN_SEEDS)
            for f in (list(CELLS) if i%2==0 else list(CELLS)[::-1])]


def confirmation_manifest(choices, incumbent):
    refs=fixed_references(incumbent)
    if set(choices)!=set(CELLS) or any(v['lr'] not in LRS for v in choices.values()):
        raise ValueError('Invalid selected cells')
    aliases={f:c for f,c in reference_cells(incumbent).items() if choices[c]['lr']==refs[f]}
    jobs=[]
    for w in WIDTHS:
        for i,s in enumerate(CONFIRM_SEEDS):
            fs=list(CELLS)+[f for f in refs if f not in aliases]
            if i%2:fs.reverse()
            for f in fs:jobs.append([f,refs[f] if f in refs else choices[f]['lr'],w,s])
    if len(jobs) not in ([30,36] if incumbent['kind']=='constant' else [36,42]) or len({tuple(j) for j in jobs})!=len(jobs):
        raise ValueError('Invalid bounded confirmation manifest')
    return dict(jobs=jobs,reference_aliases=aliases,total_distinct_trials=18+len(jobs))


def freeze_selection(p, records):
    expected=set(local_jobs(p)); actual=[(r['family'],r['lr'],r['width'],r['seed']) for r in records]
    if set(actual)!=expected or len(actual)!=len(expected) or any(
            r['status']!='completed' or not math.isfinite(r['metric']) for r in records):
        raise ValueError('Incomplete, duplicate or non-screen evidence')
    choices={}
    for f in CELLS:
        grid=[dict(lr=lr,mean_ce=mean(r['metric'] for r in records if r['family']==f and r['lr']==lr)) for lr in LRS]
        chosen=min(grid,key=lambda x:(x['mean_ce'],x['lr']))
        choices[f]={**chosen,'grid':grid,'at_grid_boundary':chosen['lr'] in [LRS[0],LRS[-1]]}
    # Capacity choice uses ONLY width64 screens; wider results cannot revise it.
    def scores(f,lr):
        return {r['seed']:r['metric'] for r in records if r['family']==f and r['lr']==lr}
    eligible=[]; contrasts={}
    for f in ['wide4','dense6']:
        a=scores(f,choices[f]['lr']); checks=[]
        for lr in [choices['dense4']['lr'],p['incumbent_schedule']['learning_rate']]:
            b=scores('dense4',lr);ds={str(seed):a[seed]-b[seed] for seed in SCREEN_SEEDS}
            checks.append(dict(reference_lr=lr,mean_delta=mean(ds.values()),paired_seed_deltas=ds,
                               supported=mean(ds.values()) < -p['min_improvement'] and all(v<0 for v in ds.values())))
        contrasts[f]=checks
        if all(c['supported'] for c in checks):eligible.append(f)
    winner=min(eligible,key=lambda f:(choices[f]['mean_ce'],f)) if eligible else 'historical_incumbent_unchanged'
    return dict(choices=choices,plan_sha256=digest(p),local_evidence_sha256=digest(records),
                selected_by='Width64 screening only; LR mean CE then lower LR; capacity requires mean improvement>.005 and every seed against tuned dense4 and the fixed reviewed incumbent.',
                local_capacity_contrasts=contrasts,confirmation=confirmation_manifest(choices,p['incumbent_schedule']),winner=winner)



def paired(rows, a, b, width, context=128, seeds=CONFIRM_SEEDS):
    def panel(f):
        found=[r for r in rows if r['family']==f and r['width']==width and r['seed'] in seeds]
        if sorted(r['seed'] for r in found)!=sorted(seeds):raise ValueError('Missing paired panel')
        return {r['seed']:r['coverage'][str(context)] for r in found}
    return [paired_blocks(panel(a),panel(b),n) for n in [2048,4096,8192]]


def summarize(p, state):
    screen=[r for r in state['records'] if r['seed'] in SCREEN_SEEDS]
    if freeze_selection(p,screen)!=state['selection']:raise ValueError('Frozen selection changed')
    manifest=state['selection']['confirmation']
    rows=[r for r in state['records'] if r['seed'] in CONFIRM_SEEDS]
    actual=[(r['family'],r['lr'],r['width'],r['seed']) for r in rows]
    if sorted(actual)!=sorted(tuple(j) for j in manifest['jobs']) or any(r['status']!='completed' for r in rows):
        raise ValueError('Incomplete confirmation or extra jobs')
    for alias,f in manifest['reference_aliases'].items():
        rows += [{**r,'family':alias,'alias_of':r['trial_id']} for r in rows if r['family']==f]
    comparisons=[]
    for w in WIDTHS:
        for a,b in [('wide4','dense4'),('dense6','dense4'),('dense6','wide4')]:
            sensitivity=paired(rows,a,b,w)
            value=sensitivity[0]
            near_neutral=(value['mean_delta']<=p['quality_margin']
                and all(x<=p['quality_margin'] for x in value['paired_seed_deltas'].values())
                and all(x['conditional_text_bootstrap_95pct'][1]<=p['quality_margin'] for x in sensitivity))
            comparisons.append(dict(width=w,candidate=a,reference=b,contrast='tuned_capacity_equal_tokens',
                sensitivity=sensitivity,near_neutral_common128=near_neutral,quality_improvement=(value['mean_delta'] < -p['min_improvement'] and all(x<0 for x in value['paired_seed_deltas'].values()))))
        for f in CELLS:
            for ref in fixed_references(p['incumbent_schedule']):
                comparisons.append(dict(width=w,candidate=f,reference=ref,contrast='recipe_minus_fixed_reference',sensitivity=paired(rows,f,ref,w)))
    fixed=[r for r in screen if r['lr']==.0025]
    fixed_contrasts=[dict(candidate=a,reference=b,sensitivity=paired(fixed,a,b,64,seeds=SCREEN_SEEDS))
        for a,b in [('wide4','dense4'),('dense6','dense4'),('dense6','wide4')]]
    return dict(plan_sha256=digest(p),selection=state['selection'],records=state['records'],
        reference_aliases=manifest['reference_aliases'],comparisons=comparisons,
        screening_fixed_lr_comparisons=fixed_contrasts,
        holdout_consumed=False,scope=p['scope'],adoption_requires_completion_review=True)


def adapter(p,raw,checked,request,directory):
    from .moe_study import adapter as architecture_adapter
    extra=architecture_adapter(p,raw,checked,request,directory)
    # validate_result checks each scheduled metric against its frozen request.
    # Preserve the explicit observed sequence in the record for final auditing.
    extra['schedule_snapshots']=[{k:row[k] for k in ('step','lr_multiplier',
        'base_learning_rate','scheduled_learning_rate')} for row in raw['metrics']]
    return extra


def verify_lineage(p):
    validate_plan(p);root=Path(p['source_root'])
    prior=p['predecessor_review']
    if file_digest(Path(prior['path']))!=prior['sha256'] or reviewed_schedule(prior['path'])!=p['incumbent_schedule']:
        raise ValueError('Reviewed shared schedule changed')
    if source_digest(root)!=p['source_sha256'] or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=p['git_commit']:
        raise ValueError('Source contract changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Dirty source')
    if file_digest(Path(p['config']['data']['path']))!=p['data_sha256']:raise ValueError('Corpus changed')
    if file_digest(Path(p['knowledge_snapshot']['path']))!=p['knowledge_snapshot']['sha256']:raise ValueError('Full knowledge snapshot changed')
    for artifact in p['artifacts']:
        if file_digest(Path(artifact['path']))!=artifact['sha256']:raise ValueError('Prerequisite artifact changed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Unset MPS fallback')
    if p['worker_bootstrap_sha256']!=digest(BOOTSTRAP):raise ValueError('Worker bootstrap changed')
    if _fingerprint(suite_spec(p,'dense4',.0025))['runtime']!=p['runtime']:
        raise ValueError('Frozen runtime changed')


def run(p, output):
    verify_lineage(p);output=Path(output)
    if json.loads((output/'plan.json').read_text())!=p:raise ValueError('Run plan mismatch')
    def spec(plan,f,lr):
        verify_lineage(plan)
        return suite_spec(plan,f,lr)
    def launch(*args,**kwargs):return _launch(*args,**kwargs,bootstrap=BOOTSTRAP)
    def check_records(records):
        for r in records:
            if r['status']!='completed':raise ValueError('No automatic retry')
            directory=output/'trials'/r['trial_id']
            for name in ['request','result']:
                if file_digest(directory/f'{name}.json')!=r[name+'_sha256']:raise ValueError('Saved raw evidence changed')
            request=json.loads((directory/'request.json').read_text())
            expected=_request(suite_spec(p,r['family'],r['lr']),r,directory)
            if request!=expected:raise ValueError('Request differs from frozen recipe')
            raw=json.loads((directory/'result.json').read_text());cfg=request['config']
            checked=validate_result(raw,request,dict(device='mps',training={**cfg['training'],'block_size':cfg['model']['block_size']}),directory)
            extra=adapter(p,raw,checked,request,directory)
            if any(r.get(k)!=v for k,v in {**checked,**extra}.items()):raise ValueError('Saved row differs from raw evidence')
    if (output/'state.json').exists():
        check_records(json.loads((output/'state.json').read_text())['records'])
    def final_report(plan,state):
        verify_lineage(plan);check_records(state['records'])
        return summarize(plan,state)
    def adapt(*args):
        verify_lineage(p)
        return adapter(p,*args)
    return engine.run(p,output,suite_builder=spec,job_builder=local_jobs,selection_builder=freeze_selection,
        report_builder=final_report,launcher=launch,result_adapter=adapt,
        completion_jobs=lambda plan,selection:selection['confirmation']['jobs'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-training',action='store_true');args=parser.parse_args()
    if not args.allow_training:parser.error('Explicit --allow-training required')
    run(json.loads(args.plan.read_text()),args.output)

if __name__=='__main__':main()
