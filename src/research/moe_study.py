"""Bounded top2-of4 MoE and dense capacity comparison with fresh controls.

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
SCREEN_SEEDS = [2081, 2082]
CONFIRM_SEEDS = [2091, 2092, 2093]
WIDTHS = [64, 256]
CELLS = {'dense': {}, 'dense_wide': {}, 'moe4_top2': {}}
FIXED = {'gelu_reference': .002, 'shallow_reference': .0025, 'deep_reference': .0025}
REFERENCE_CELLS = {'deep_reference': 'dense'}


def settings(family):
    if family in CELLS:
        return dict(depth=4, attention='mha', context=128, batch=8)
    if family not in FIXED:
        raise ValueError('Unregistered family')
    return dict(depth=4 if family == 'deep_reference' else 2, attention='mha', context=128, batch=8)


def validate_plan(p):
    expected = dict(study='moe-quality-v1', families=list(CELLS), learning_rates=LRS,
                    seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS, widths=WIDTHS,
                    eval_seed=2026, min_improvement=.005, max_trials=54,
                    trial_seconds=1800, session_seconds=28800, max_disk_mb=4096,
                    primary_metric='full_validation_context128', quality_margin=.01,
                    bootstrap_blocks=[2048,4096,8192], bootstrap_draws=2000,
                    bootstrap_seed=20261024, evaluation_targets=111538, total_training_targets=3072000,
                    batch_layout='paired_blocks512', moe_policy='moe4_top2_half_readout', max_process_rss_mb=4096,max_mps_driver_mb=6144)
    if any(p.get(k) != v for k,v in expected.items()):
        raise ValueError('Frozen MoE quality protocol changed')
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


def make_plan(root, config):
    config=copy.deepcopy(config)
    config['model'].setdefault('attn_dropout',0.)
    from .team.suite import RUNTIME_TRAINING
    for key in RUNTIME_TRAINING:config['training'].pop(key,None)
    p=engine.make_plan(root,config)
    p.update(study='moe-quality-v1',families=list(CELLS),learning_rates=LRS,
             seeds=SCREEN_SEEDS,confirmation_seeds=CONFIRM_SEEDS,widths=WIDTHS,
             max_trials=54,trial_seconds=1800,session_seconds=28800,max_disk_mb=4096,
             primary_metric='full_validation_context128',quality_margin=.01,
             bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
             evaluation_targets=111538,total_training_targets=3072000,
             batch_layout='paired_blocks512',moe_policy='moe4_top2_half_readout',max_process_rss_mb=4096,max_mps_driver_mb=6144,
             selection='Three independent width64 LR choices on seeds2081/2082; freeze before any confirmation. No pruning or extension. Fixed identical references alias exact recipes.',
             scope='Adaptive development: three equally tuned families, fresh controls and confirmation, equal 3000 updates and3072000 targets. MoE active expert weights match dense; total expert weights match wide dense, excluding router. Not equal compute, optimal router/width scaling, holdout or population inference.')
    p['screen_jobs']=[list(x) for x in local_jobs(p)]
    validate_plan(p)
    p['runtime']=_fingerprint(suite_spec(p,'dense',.0025))['runtime']
    p['worker_bootstrap_sha256']=digest(BOOTSTRAP)
    return p


def suite_spec(p, family, lr):
    validate_plan(p)
    if lr not in ([FIXED[family]] if family in FIXED else LRS):
        raise ValueError('Unregistered LR')
    s=settings(family); cfg=copy.deepcopy(p['config']); m,t=cfg['model'],cfg['training']
    m.update(n_layers=s['depth'],attn_type=s['attention'],block_size=s['context'])
    t.update(learning_rate=lr,batch_size=8,batch_layout='paired_blocks512')
    if family!='gelu_reference':
        m.update(ffn='swiglu',ffn_bias=False,ffn_width_policy='swiglu_21_8',
                 attention_width_policy='mha_head32')
    if s['depth']==4:
        m['depth_width_policy']='depth4_unscaled'
    if family=='dense_wide':
        m.update(ffn_width_policy='swiglu_21_4',depth_width_policy='depth4_wide_unscaled')
    if family=='moe4_top2':
        m.update(mlp_type='moe',ffn_width_policy='moe_swiglu_21_16',depth_width_policy='depth4_moe_unscaled',
                 moe=dict(num_experts=4,top_k=2,shared_num_experts=0,ffn='swiglu',aux_loss_weight=.01,
                          record_expert_hist=True,width_policy='moe4_top2_half_readout'))
    arm=dict(root=p['source_root'],source_sha256=p['source_sha256'],config=cfg)
    return validate_suite(dict(version=1,hypothesis=f'Four-layer MHA MoE capacity comparison: {family}',
        baseline=arm,candidate=copy.deepcopy(arm),
        targets=[dict(name=f'width-{w}',width=w,n_heads=w//32) for w in WIDTHS],
        seeds=SCREEN_SEEDS+CONFIRM_SEEDS,eval_seed=2026,min_improvement=.005,audit_test=False,
        data_sha256=p['data_sha256'],limits=dict(max_trials=54,trial_seconds=1800,
        session_seconds=28800,max_disk_mb=4096,max_failures=1,max_consecutive_failures=1)))


def local_jobs(p):
    return [(f,lr,64,s) for lr in LRS for i,s in enumerate(SCREEN_SEEDS)
            for f in (list(CELLS) if i%2==0 else list(CELLS)[::-1])]


def confirmation_manifest(choices):
    if set(choices)!=set(CELLS) or any(v['lr'] not in LRS for v in choices.values()):
        raise ValueError('Invalid selected cells')
    aliases={f:c for f,c in REFERENCE_CELLS.items() if choices[c]['lr']==FIXED[f]}
    jobs=[]
    for w in WIDTHS:
        for i,s in enumerate(CONFIRM_SEEDS):
            fs=list(CELLS)+[f for f in FIXED if f not in aliases]
            if i%2:fs.reverse()
            for f in fs:jobs.append([f,FIXED[f] if f in FIXED else choices[f]['lr'],w,s])
    if len(jobs) not in [30,36] or len({tuple(j) for j in jobs})!=len(jobs):
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
    # Architecture choice uses ONLY width64 screens; wider results cannot revise it.
    def scores(f,lr):
        return {r['seed']:r['metric'] for r in records if r['family']==f and r['lr']==lr}
    eligible=[]; contrasts={}
    for f in ['dense_wide','moe4_top2']:
        a=scores(f,choices[f]['lr']); checks=[]
        for lr in [choices['dense']['lr'],.0025]:
            b=scores('dense',lr);ds={str(seed):a[seed]-b[seed] for seed in SCREEN_SEEDS}
            checks.append(dict(reference_lr=lr,mean_delta=mean(ds.values()),paired_seed_deltas=ds,
                               supported=mean(ds.values()) < -p['min_improvement'] and all(v<0 for v in ds.values())))
        contrasts[f]=checks
        if all(c['supported'] for c in checks):eligible.append(f)
    winner=min(eligible,key=lambda f:(choices[f]['mean_ce'],f)) if eligible else 'historical_incumbent_unchanged'
    return dict(choices=choices,plan_sha256=digest(p),local_evidence_sha256=digest(records),
                selected_by='Width64 screening only; LR mean CE then lower LR; architecture requires mean improvement>.005 and every seed against tuned and fixed dense.',
                local_architecture_contrasts=contrasts,confirmation=confirmation_manifest(choices),winner=winner)



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
        for a,b in [('dense_wide','dense'),('moe4_top2','dense'),('moe4_top2','dense_wide')]:
            sensitivity=paired(rows,a,b,w)
            value=sensitivity[0]
            near_neutral=(value['mean_delta']<=p['quality_margin']
                and all(x<=p['quality_margin'] for x in value['paired_seed_deltas'].values())
                and all(x['conditional_text_bootstrap_95pct'][1]<=p['quality_margin'] for x in sensitivity))
            comparisons.append(dict(width=w,candidate=a,reference=b,contrast='tuned_architecture_equal_tokens',
                sensitivity=sensitivity,near_neutral_common128=near_neutral,quality_improvement=(value['mean_delta'] < -p['min_improvement'] and all(x<0 for x in value['paired_seed_deltas'].values()))))
        for f in CELLS:
            for ref in FIXED:
                comparisons.append(dict(width=w,candidate=f,reference=ref,contrast='recipe_minus_fixed_reference',sensitivity=paired(rows,f,ref,w)))
    fixed=[r for r in screen if r['lr']==.0025]
    fixed_contrasts=[dict(candidate=a,reference=b,sensitivity=paired(fixed,a,b,64,seeds=SCREEN_SEEDS))
        for a,b in [('dense_wide','dense'),('moe4_top2','dense'),('moe4_top2','dense_wide')]]
    return dict(plan_sha256=digest(p),selection=state['selection'],records=state['records'],
        reference_aliases=manifest['reference_aliases'],comparisons=comparisons,
        screening_fixed_lr_comparisons=fixed_contrasts,
        holdout_consumed=False,scope=p['scope'],adoption_requires_completion_review=True)


def adapter(p,raw,checked,request,directory):
    extra=coverage_adapter(p,raw,checked,request,directory)
    cfg=request['config']
    if cfg['model']['mlp_type']=='moe':
        metrics=[json.loads(x) for x in Path(cfg['training']['metrics_file']).read_text().splitlines()]
        for row in metrics:
            aux=row['train_aux_loss']; ce=row['train_ce_loss']; total=row['train_loss']
            if (not all(math.isfinite(x) for x in [aux,ce,total]) or aux<0
                    or not math.isclose(total,ce+.01*aux,abs_tol=2e-6)):
                raise ValueError('MoE training objective contract failed')
            usage=row['moe_usage']
            if len(usage['per_layer'])!=4:raise ValueError('Missing MoE routing diagnostics')
            for layer in usage['per_layer']:
                us=layer['usage']
                if len(us)!=4 or any(not math.isfinite(x) or x<0 for x in us) or not math.isclose(sum(us),1,abs_tol=1e-6):
                    raise ValueError('Invalid MoE routing distribution')
        extra['routing_snapshots']=[dict(step=x['step'],train_aux_loss=x['train_aux_loss'],moe_usage=x['moe_usage']) for x in metrics]
    return extra


def verify_lineage(p):
    validate_plan(p);root=Path(p['source_root'])
    if source_digest(root)!=p['source_sha256'] or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()!=p['git_commit']:
        raise ValueError('Source contract changed')
    if subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip():raise ValueError('Dirty source')
    if file_digest(Path(p['config']['data']['path']))!=p['data_sha256']:raise ValueError('Corpus changed')
    if file_digest(Path(p['knowledge_snapshot']['path']))!=p['knowledge_snapshot']['sha256']:raise ValueError('Full knowledge snapshot changed')
    for artifact in p['artifacts']:
        if file_digest(Path(artifact['path']))!=artifact['sha256']:raise ValueError('Prerequisite artifact changed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='1':raise ValueError('Unset MPS fallback')
    if p['worker_bootstrap_sha256']!=digest(BOOTSTRAP):raise ValueError('Worker bootstrap changed')
    if _fingerprint(suite_spec(p,'dense',.0025))['runtime']!=p['runtime']:
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
