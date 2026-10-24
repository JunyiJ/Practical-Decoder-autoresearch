"""Bounded depth x GQA x context factorial with independent development seeds.

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
from .coverage_eval import paired_blocks, validate_evaluation
from .protocol import digest, file_digest
from .team.suite import validate_suite, _launch, _request, _fingerprint, WORKER_BOOTSTRAP, source_digest
from .runner import validate_result

LRS = [.00125, .0025, .005]
SCREEN_SEEDS = [2021, 2022]
CONFIRM_SEEDS = [2031, 2032, 2033]
WIDTHS = [64, 256]
CELLS = {f'd{d}_{a}_c{c}': {'depth': d, 'attention': a, 'context': c}
         for d in [2, 4] for a in ['mha', 'gqa'] for c in [128, 512]}
FIXED = {'gelu_reference': .002, 'shallow_reference': .0025, 'deep_reference': .0025}
REFERENCE_CELLS = {'shallow_reference': 'd2_mha_c128', 'deep_reference': 'd4_mha_c128'}
BOOTSTRAP = WORKER_BOOTSTRAP + '''
import json
from pathlib import Path
from src.research.coverage_eval import evaluate_job
from src.research.protocol import file_digest
from src.research.runner import atomic_json
request_path=Path(__import__('sys').argv[__import__('sys').argv.index('--request')+1])
req=json.loads(request_path.read_text());result_path=Path(req['result_path'])
raw=json.loads(result_path.read_text());checkpoint=raw['final_checkpoint']
job={'checkpoint':checkpoint,'checkpoint_sha256':file_digest(Path(checkpoint)),
     'completed_steps':req['config']['training']['max_iters'],
     'data_sha256':file_digest(Path(req['config']['data']['path'])),
     'contexts':sorted(set([128,req['config']['model']['block_size']]))}
raw['coverage']=evaluate_job(job);atomic_json(result_path,raw)
'''


def settings(family):
    if family in CELLS:
        return CELLS[family]
    if family not in FIXED:
        raise ValueError('Unregistered family')
    return dict(depth=4 if family == 'deep_reference' else 2, attention='mha', context=128)


def validate_plan(p):
    expected = dict(study='context-interaction-v1', families=list(CELLS), learning_rates=LRS,
                    seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS, widths=WIDTHS,
                    eval_seed=2026, min_improvement=.005, max_trials=114,
                    trial_seconds=1200, session_seconds=28800, max_disk_mb=4096,
                    primary_metric='full_validation_context128', quality_margin=.01,
                    bootstrap_blocks=[2048,4096,8192], bootstrap_draws=2000,
                    bootstrap_seed=20261024, evaluation_targets=111538)
    if any(p.get(k) != v for k,v in expected.items()):
        raise ValueError('Frozen context interaction protocol changed')
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
    p.update(study='context-interaction-v1',families=list(CELLS),learning_rates=LRS,
             seeds=SCREEN_SEEDS,confirmation_seeds=CONFIRM_SEEDS,widths=WIDTHS,
             max_trials=114,trial_seconds=1200,session_seconds=28800,max_disk_mb=4096,
             primary_metric='full_validation_context128',quality_margin=.01,
             bootstrap_blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_seed=20261024,
             evaluation_targets=111538,
             selection='Eight independent width64 LR choices on seeds2021/2022; freeze before any confirmation. No pruning or extension. Fixed identical references alias exact recipes.',
             scope='Adaptive development with fresh confirmation seeds, shared paired training targets, equal updates and token exposures. Not equal compute, holdout, proof of muP transfer or population non-inferiority.')
    p['screen_jobs']=[list(x) for x in local_jobs(p)]
    validate_plan(p)
    p['runtime']=_fingerprint(suite_spec(p,'d2_mha_c128',.0025))['runtime']
    p['worker_bootstrap_sha256']=digest(BOOTSTRAP)
    return p


def suite_spec(p, family, lr):
    validate_plan(p)
    if lr not in ([FIXED[family]] if family in FIXED else LRS):
        raise ValueError('Unregistered LR')
    s=settings(family); cfg=copy.deepcopy(p['config']); m,t=cfg['model'],cfg['training']
    m.update(n_layers=s['depth'],attn_type=s['attention'],block_size=s['context'])
    t.update(learning_rate=lr,batch_size=1024//s['context'],batch_layout='paired_blocks512')
    if family!='gelu_reference':
        m.update(ffn='swiglu',ffn_bias=False,ffn_width_policy='swiglu_21_8',
                 attention_width_policy='gqa_half_head32' if s['attention']=='gqa' else 'mha_head32')
    if s['depth']==4:
        m['depth_width_policy']='depth4_gqa_unscaled' if s['attention']=='gqa' else 'depth4_unscaled'
    arm=dict(root=p['source_root'],source_sha256=p['source_sha256'],config=cfg)
    return validate_suite(dict(version=1,hypothesis=f'Depth-attention-context interaction: {family}',
        baseline=arm,candidate=copy.deepcopy(arm),
        targets=[dict(name=f'width-{w}',width=w,n_heads=w//32) for w in WIDTHS],
        seeds=SCREEN_SEEDS+CONFIRM_SEEDS,eval_seed=2026,min_improvement=.005,audit_test=False,
        data_sha256=p['data_sha256'],limits=dict(max_trials=114,trial_seconds=1200,
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
    if len(jobs) not in [54,60,66] or len({tuple(j) for j in jobs})!=len(jobs):
        raise ValueError('Invalid bounded confirmation manifest')
    return dict(jobs=jobs,reference_aliases=aliases,total_distinct_trials=48+len(jobs))


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
    return dict(choices=choices,plan_sha256=digest(p),local_evidence_sha256=digest(records),
                selected_by='Full-validation context128 on width64 screening seeds only',
                confirmation=confirmation_manifest(choices),winner='historical_incumbent_unchanged')


def validate_coverage(raw, request, p):
    cov=raw['coverage'];c=request['config']['model']['block_size']
    if (cov['holdout_consumed'] is not False or cov['device']!='mps'
            or cov['torch']!=raw['runtime']['torch'] or set(cov['evaluations'])!={str(v) for v in [128,c]}):
        raise ValueError('Coverage runtime or context contract mismatch')
    for context,value in cov['evaluations'].items():
        context=int(context)
        if value['context']!=context or value['eval_batch_size']!=1024//context:
            raise ValueError('Coverage context/batch mismatch')
        if value['window_count']!=math.ceil(p['evaluation_targets']/context):
            raise ValueError('Native coverage window count mismatch')
        # Reuse audited count/sum validator after checking native shape above.
        common={**value,'context':128,'eval_batch_size':8,
                'window_count':math.ceil(p['evaluation_targets']/128)}
        validate_evaluation(common,p['evaluation_targets'])
    return cov['evaluations']


def adapter(p, raw, row, request, directory):
    coverage=validate_coverage(raw,request,p)
    if raw['coverage']['checkpoint_sha256']!=row['checkpoint_sha256']:
        raise ValueError('Coverage checkpoint mismatch')
    return dict(sampled_metric=row['metric'],metric=coverage['128']['mean_ce'],coverage=coverage,
                request_sha256=file_digest(directory/'request.json'),result_sha256=file_digest(directory/'result.json'))


def paired(rows, a, b, width, context=128, seeds=CONFIRM_SEEDS):
    def panel(f):
        found=[r for r in rows if r['family']==f and r['width']==width and r['seed'] in seeds]
        if sorted(r['seed'] for r in found)!=sorted(seeds):raise ValueError('Missing paired panel')
        return {r['seed']:r['coverage'][str(context)] for r in found}
    return [paired_blocks(panel(a),panel(b),n) for n in [2048,4096,8192]]


def interaction_values(rows, width, seeds):
    """Explicit paired finite contrasts, including recipe-level three-way effect."""
    panel={}
    for f in CELLS:
        for seed in seeds:
            found=[r for r in rows if r['width']==width and r['family']==f and r['seed']==seed]
            if len(found)!=1:raise ValueError('Incomplete factorial')
            panel[f,seed]=found[0]['metric']
    out={}
    for seed in seeds:
        q=lambda d,a,c:panel[f'd{d}_{a}_c{c}',seed]
        g=lambda d,c:q(d,'gqa',c)-q(d,'mha',c)
        out[str(seed)]={
            'depth_x_attention_c128':g(4,128)-g(2,128),
            'depth_x_attention_c512':g(4,512)-g(2,512),
            'attention_x_context_d2':g(2,512)-g(2,128),
            'attention_x_context_d4':g(4,512)-g(4,128),
            **{f'depth_x_context_{a}':(q(4,a,512)-q(2,a,512))-(q(4,a,128)-q(2,a,128)) for a in ['mha','gqa']},
            'depth_x_attention_x_context':(g(4,512)-g(2,512))-(g(4,128)-g(2,128))}
    return dict(paired_seed_contrasts=out,mean_contrasts={k:mean(v[k] for v in out.values()) for k in next(iter(out.values()))})


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
        for d in [2,4]:
            for c in [128,512]:
                a,b=f'd{d}_gqa_c{c}',f'd{d}_mha_c{c}'
                sensitivity=paired(rows,a,b,w)
                s=sensitivity[0]
                eligible=(s['mean_delta']<=p['quality_margin']
                    and all(x<=p['quality_margin'] for x in s['paired_seed_deltas'].values())
                    and all(x['conditional_text_bootstrap_95pct'][1]<=p['quality_margin'] for x in sensitivity))
                comparisons.append(dict(width=w,candidate=a,reference=b,contrast='gqa_minus_mha',
                    sensitivity=sensitivity,quality_eligible=eligible,
                    native_sensitivity=paired(rows,a,b,w,c),inference_eligible='unmeasured for new trained cells'))
        for a in ['mha','gqa']:
            for c in [128,512]:
                comparisons.append(dict(width=w,candidate=f'd4_{a}_c{c}',reference=f'd2_{a}_c{c}',contrast='depth4_minus_depth2',
                    sensitivity=paired(rows,f'd4_{a}_c{c}',f'd2_{a}_c{c}',w)))
            for d in [2,4]:
                comparisons.append(dict(width=w,candidate=f'd{d}_{a}_c512',reference=f'd{d}_{a}_c128',contrast='context512_minus_context128',
                    sensitivity=paired(rows,f'd{d}_{a}_c512',f'd{d}_{a}_c128',w)))
        for f in CELLS:
            for ref in FIXED:
                comparisons.append(dict(width=w,candidate=f,reference=ref,contrast='recipe_minus_fixed_reference',sensitivity=paired(rows,f,ref,w)))
    return dict(plan_sha256=digest(p),selection=state['selection'],records=state['records'],
        reference_aliases=manifest['reference_aliases'],comparisons=comparisons,
        screening_fixed_lr_interactions=interaction_values([r for r in screen if r['lr']==.0025],64,SCREEN_SEEDS),
        confirmation_tuned_recipe_interactions={str(w):interaction_values(rows,w,CONFIRM_SEEDS) for w in WIDTHS},
        holdout_consumed=False,scope=p['scope'],historical_incumbent_unchanged=True)


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
    if _fingerprint(suite_spec(p,'d2_mha_c128',.0025))['runtime']!=p['runtime']:
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
