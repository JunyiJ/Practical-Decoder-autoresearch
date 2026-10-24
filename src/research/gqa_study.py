"""Stage 2D: bounded GQA versus MHA study with a frozen stage-1 Muon control.

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

FAMILIES = {'control': 'gelu', 'incumbent': 'swiglu', 'mha_tuned': 'swiglu', 'gqa_half': 'swiglu'}
FIXED_LRS = {'control': .002, 'incumbent': .0025}
HEAD_DIMS = {'control':32, 'incumbent':32, 'mha_tuned':32, 'gqa_half':32}
LRS = [.0015, .002, .0025]
SEEDS = [444, 555, 666]


def validate_plan(plan):
    if (plan.get('study') != 'stage2d-gqa-v1' or plan['families'] != FAMILIES
            or plan['learning_rates'] != LRS or plan['seeds'] != SEEDS
            or plan['widths'] != [64, 128, 256] or plan['eval_seed'] != 2026
            or plan['min_improvement'] != .005 or plan['max_trials'] != 48 or plan.get('head_dimensions') != HEAD_DIMS
            or plan['trial_seconds'] != 600 or plan['session_seconds'] != 14400
            or plan['max_disk_mb'] != 4096):
        raise ValueError('Stage 2D frozen protocol changed')
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
    plan.update(study='stage2d-gqa-v1', families=FAMILIES,
                learning_rates=LRS, seeds=SEEDS, max_trials=48, head_dimensions=HEAD_DIMS, max_disk_mb=4096,
                selection='Equal three-LR screens for SwiGLU MHA/GQA at width64. Lowest mean final CE per family; ties lower LR. GQA wins only if all seed deltas are negative and mean < -.005 versus fixed GELU, fixed SwiGLU incumbent and tuned MHA. Otherwise retain prior incumbent. Freeze before larger widths.',
                scope='Fresh development seeds on the same adaptively used Shakespeare split. Fewer attention parameters in GQA; same SwiGLU FFN, query head32 and equal tokens; not equal measured compute, independent validation, convergence or production-scale evidence.')
    validate_plan(plan)
    return plan


def suite_spec(plan, family, lr):
    validate_plan(plan)
    if family not in FAMILIES or lr not in ([FIXED_LRS[family]] if family in FIXED_LRS else LRS):
        raise ValueError('Unregistered architecture/LR')
    cfg=copy.deepcopy(plan['config']); cfg['model']['ffn']=FAMILIES[family]
    cfg['training']['learning_rate']=lr
    if family != 'control': cfg['model']['ffn_bias']=False
    if family != 'control': cfg['model']['ffn_width_policy']='swiglu_21_8'
    if family == 'gqa_half':
        cfg['model'].update(attn_type='gqa', attention_width_policy='gqa_half_head32')
    cfg['model']['n_heads']=cfg['model']['dim']//HEAD_DIMS[family]
    arm={'root':plan['source_root'],'source_sha256':plan['source_sha256'],'config':cfg}
    return validate_suite({'version':1,'hypothesis':f'GQA sharing study: {family}',
        'baseline':arm,'candidate':copy.deepcopy(arm),
        'targets':[{'name':f'width-{w}','width':w,'n_heads':w//HEAD_DIMS[family]} for w in plan['widths']],
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
                if family in FIXED_LRS and lr!=FIXED_LRS[family]: continue
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
        lrs=[FIXED_LRS[family]] if family in FIXED_LRS else plan['learning_rates']
        grid=[{'lr':lr,'mean_ce':mean(r['metric'] for r in records if r['family']==family and r['lr']==lr)} for lr in lrs]
        choice=min(grid,key=lambda x:(x['mean_ce'],x['lr']))
        choices[family]={**choice,'grid':grid,'at_grid_boundary':len(lrs)>1 and choice['lr'] in [min(lrs),max(lrs)]}
    def chosen(f): return [r for r in records if r['family']==f and r['lr']==choices[f]['lr']]
    contrasts={f:paired(chosen('gqa_half'),chosen(f),plan['min_improvement']) for f in ['control','incumbent','mha_tuned']}
    return {'choices':choices,'local_evidence_sha256':digest(records),'plan_sha256':digest(plan),
            'selected_by':'width-64 only','architecture_contrasts':contrasts,
            'winner':'gqa_half' if all(c['supported'] for c in contrasts.values()) else 'incumbent'}


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
                'vs_incumbent':paired(selected,rows('incumbent',w),plan['min_improvement']),
                'vs_tuned_mha':paired(selected,rows('mha_tuned',w),plan['min_improvement']),
                'mean_training_seconds':mean(r['training_time_sec'] for r in selected),
                'mean_peak_process_rss_mb':mean(r['rss_mb'] for r in selected)}
    return {'version':1,'plan_sha256':digest(plan),'selection':selection,'families':result,
            'records':records,'holdout_consumed':False,'scope':plan['scope'],
            'limitations':['Small LR grid; boundary winners are not optimal.',
                'Fresh development seeds, same adaptively used validation split; no independent holdout confirmation.',
                'Fewer attention parameters in GQA; same SwiGLU FFN, query head32 and equal tokens, not equal elapsed time or compute.',
                'Fixed-aspect-ratio scaling checks do not prove optimal LR transfer or equal activation statistics.',
                'GQA halves KV heads and changes fused-QKV shape and its Muon aspect factor. Query head dimension, RoPE basis and SwiGLU FFN stay fixed. Different initialization draws; no pure sharing causal attribution or inference-cache benchmark.']}


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
