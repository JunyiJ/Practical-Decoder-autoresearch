"""Stage 2A: bounded GELU/SiLU study with a frozen stage-1 Muon control.

Reuses the audited serial worker engine. Selection is frozen from width 64,
including the architecture decision, before any larger-width measurements.
"""
import argparse
import copy
import json
from pathlib import Path
from statistics import mean, stdev

from . import optimizer_study as engine
from .protocol import digest, file_digest
from .team.suite import validate_suite

FAMILIES = {'control': 'gelu', 'gelu_tuned': 'gelu', 'silu': 'silu'}
LRS = [.0015, .002, .0025]
SEEDS = [44, 55, 66]


def validate_plan(plan):
    if (plan.get('study') != 'stage2a-activation-v1' or plan['families'] != FAMILIES
            or plan['learning_rates'] != LRS or plan['seeds'] != SEEDS
            or plan['widths'] != [64, 128, 256] or plan['eval_seed'] != 2026
            or plan['min_improvement'] != .005 or plan['max_trials'] != 39
            or plan['trial_seconds'] != 600 or plan['session_seconds'] != 14400
            or plan['max_disk_mb'] != 4096):
        raise ValueError('Stage 2A frozen protocol changed')
    model, train = plan['config']['model'], plan['config']['training']
    expected_model = dict(n_layers=2, dropout=0., attn_dropout=0., norm_type='rmsnorm',
                          rope=1, attn_type='mha', mlp_type='dense', ffn='gelu',
                          parameterization='mup', base_width=64, block_size=128)
    expected_train = dict(optimizer='muon', decay_mode='independent', learning_rate=.002,
                          weight_decay=.00006, grad_clip=1., batch_size=8, max_iters=3000,
                          eval_every=300, eval_iters=30, cpu_threads=4, device='mps')
    if any(model.get(k) != v for k,v in expected_model.items()) or 'hidden_dim' in model:
        raise ValueError('Stage-1 control model changed')
    if any(train.get(k) != v for k,v in expected_train.items()):
        raise ValueError('Stage-1 control training changed')
    if set(model) - set(expected_model) - {'dim','n_heads','vocab_size'}:
        raise ValueError('Unexpected control model override')
    return plan


def make_plan(root, config):
    config = copy.deepcopy(config)
    config['model'].setdefault('attn_dropout', config['model'].get('dropout', 0.))
    plan = engine.make_plan(root, config)
    plan.update(study='stage2a-activation-v1', families=FAMILIES,
                learning_rates=LRS, seeds=SEEDS, max_trials=39, max_disk_mb=4096,
                selection='Equal three-LR screens for GELU and SiLU at width64. Lowest mean final CE per family; ties lower LR. SiLU wins only if all seed deltas are negative and mean < -.005 versus BOTH fixed control and tuned GELU. Freeze before larger widths.',
                scope='Fresh development seeds on the same adaptively used Shakespeare split. Equal parameters, shapes and tokens; not equal measured compute, independent validation, convergence or production-scale evidence.')
    validate_plan(plan)
    return plan


def suite_spec(plan, family, lr):
    validate_plan(plan)
    if family not in FAMILIES or lr not in ([.002] if family=='control' else LRS):
        raise ValueError('Unregistered architecture/LR')
    cfg=copy.deepcopy(plan['config']); cfg['model']['ffn']=FAMILIES[family]
    cfg['training']['learning_rate']=lr
    arm={'root':plan['source_root'],'source_sha256':plan['source_sha256'],'config':cfg}
    return validate_suite({'version':1,'hypothesis':f'Activation study: {family}',
        'baseline':arm,'candidate':copy.deepcopy(arm),
        'targets':[{'name':f'width-{w}','width':w,'n_heads':w//32} for w in plan['widths']],
        'seeds':plan['seeds'],'eval_seed':plan['eval_seed'],'min_improvement':plan['min_improvement'],
        'audit_test':False,'data_sha256':plan['data_sha256'],
        'limits':{'max_trials':plan['max_trials'],'session_seconds':plan['session_seconds'],
                  'trial_seconds':plan['trial_seconds'],'max_disk_mb':plan['max_disk_mb'],
                  'max_failures':1,'max_consecutive_failures':1}})


def local_jobs(plan):
    jobs=[]
    for lr in plan['learning_rates']:
        for n,seed in enumerate(plan['seeds']):
            families=list(plan['families'])
            if n % 2: families.reverse()
            for family in families:
                if family=='control' and lr!=.002: continue
                jobs.append((family,lr,64,seed))
    return jobs


def paired(rows, refs, threshold):
    refs={r['seed']:r for r in refs}
    if set(refs)!={r['seed'] for r in rows} or len(rows)!=3 or len(refs)!=3:
        raise ValueError('Missing or duplicate paired seeds')
    deltas={str(r['seed']):r['metric']-refs[r['seed']]['metric'] for r in rows}
    return {'mean_delta':mean(deltas.values()),'sample_sd_delta':stdev(deltas.values()),
            'paired_deltas':deltas,
            'supported':mean(deltas.values()) < -threshold and all(d<0 for d in deltas.values())}


def freeze_selection(plan, records):
    expected=set(local_jobs(plan))
    actual=[(r['family'],r['lr'],r['width'],r['seed']) for r in records]
    if (set(actual)!=expected or len(actual)!=len(expected)
            or any(r['status']!='completed' for r in records)):
        raise ValueError('Cannot select from incomplete, duplicate or nonlocal evidence')
    choices={}
    for family in plan['families']:
        lrs=[.002] if family=='control' else plan['learning_rates']
        grid=[{'lr':lr,'mean_ce':mean(r['metric'] for r in records if r['family']==family and r['lr']==lr)} for lr in lrs]
        choice=min(grid,key=lambda x:(x['mean_ce'],x['lr']))
        choices[family]={**choice,'grid':grid,'at_grid_boundary':len(lrs)>1 and choice['lr'] in [min(lrs),max(lrs)]}
    def chosen(f): return [r for r in records if r['family']==f and r['lr']==choices[f]['lr']]
    contrasts={f:paired(chosen('silu'),chosen(f),plan['min_improvement']) for f in ['control','gelu_tuned']}
    return {'choices':choices,'local_evidence_sha256':digest(records),'plan_sha256':digest(plan),
            'selected_by':'width-64 only','architecture_contrasts':contrasts,
            'winner':'silu' if all(c['supported'] for c in contrasts.values()) else 'control'}


def summarize(plan, state):
    records=state['records'];selection=state['selection'];result={}
    def rows(f,w):
        found=[r for r in records if r['family']==f and r['lr']==selection['choices'][f]['lr'] and r['width']==w]
        if sorted(r['seed'] for r in found)!=sorted(plan['seeds']) or any(r['status']!='completed' for r in found):
            raise ValueError('Missing complete selected-family evidence')
        return found
    for family in plan['families']:
        result[family]={'selected_lr':selection['choices'][family]['lr'],'widths':{}}
        for w in plan['widths']:
            selected=rows(family,w)
            result[family]['widths'][str(w)]={
                'mean_ce':mean(r['metric'] for r in selected),
                'vs_control':paired(selected,rows('control',w),plan['min_improvement']),
                'vs_tuned_gelu':paired(selected,rows('gelu_tuned',w),plan['min_improvement']),
                'mean_training_seconds':mean(r['training_time_sec'] for r in selected),
                'mean_peak_process_rss_mb':mean(r['rss_mb'] for r in selected)}
    return {'version':1,'plan_sha256':digest(plan),'selection':selection,'families':result,
            'records':records,'holdout_consumed':False,'scope':plan['scope'],
            'limitations':['Small LR grid; boundary winners are not optimal.',
                'Fresh development seeds, same adaptively used validation split; no independent holdout confirmation.',
                'Equal parameters and token budgets, not equal elapsed time or compute.',
                'Fixed-aspect-ratio scaling implementation checks are not proof of optimal LR transfer.',
                'No depth, head dimension, gated FFN, attention family or MoE change in this round.']}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--allow-training',action='store_true')
    args=p.parse_args()
    if not args.allow_training:p.error('Explicit --allow-training required')
    plan=validate_plan(json.loads(args.plan.read_text()))
    if file_digest(Path(plan['config']['data']['path']))!=plan['data_sha256']:
        raise ValueError('Corpus changed')
    engine.run(plan,args.output,suite_builder=suite_spec,job_builder=local_jobs,
               selection_builder=freeze_selection,report_builder=summarize)

if __name__=='__main__': main()
