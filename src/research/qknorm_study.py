"""Finite Q/K RMS quality comparison with fresh tuning and frozen confirmation.

Uses the serial trusted worker. A final independently reviewed horizon result,
clean source, prerequisite artifacts and full knowledge snapshot are required.
"""
import argparse
import copy
import json
import os
from pathlib import Path
from statistics import mean
import subprocess

from . import horizon_study as horizon, optimizer_study as engine
from .coverage_eval import paired_blocks, validate_evaluation
from .protocol import digest, file_digest
from .qknorm_predecessor import reviewed_horizon
from .runner import validate_result
from .team.suite import validate_suite, _request, _launch, _fingerprint, source_digest

FAMILIES = ['standard', 'qkrms']
LRS = [.00125, .0025, .005]
SCREEN_SEEDS = [2181, 2182]
CONFIRM_SEEDS = [2191, 2192, 2193]
WIDTHS = [64, 256]
REFERENCE = 'incumbent_reference'
BOOTSTRAP = horizon.BOOTSTRAP


def local_jobs(p):
    return [(f, lr, 64, seed) for lr in LRS for i, seed in enumerate(SCREEN_SEEDS)
            for f in (FAMILIES if i % 2 == 0 else FAMILIES[::-1])]


def backbone(recipe):
    if (not isinstance(recipe, dict) or set(recipe) != {'family', 'total_steps', 'cooldown_start'}
            or recipe['family'] not in horizon.CELLS
            or (recipe['total_steps'], recipe['cooldown_start']) != horizon.CELLS[recipe['family']]
            or type(recipe['total_steps']) is not int or type(recipe['cooldown_start']) is not int):
        raise ValueError('Invalid final adopted horizon recipe')
    total, start = recipe['total_steps'], recipe['cooldown_start']
    model = dict(dim=64, n_heads=2, vocab_size=1, n_layers=4, ffn='swiglu', ffn_bias=False,
        mlp_type='dense', attn_type='mha', parameterization='mup', base_width=64, block_size=128,
        rope=1, norm_type='rmsnorm', dropout=0., attn_dropout=0., attention_width_policy='mha_head32',
        ffn_width_policy='swiglu_21_8', depth_width_policy='depth4_unscaled')
    training = dict(batch_size=8, max_iters=total, learning_rate=.005, weight_decay=.00006,
        optimizer='muon', decay_mode='independent', grad_clip=1., eval_every=300, eval_iters=30,
        device='mps', cpu_threads=4, batch_layout='paired_blocks512',
        lr_schedule=horizon.schedule(recipe['family']),
        weight_averaging=dict(kind='spaced_tail_ema', total_steps=total, start_step=start,
                             interval=10, half_lives=[200]))
    return model, training


def validate_plan(p):
    expected = dict(study='qknorm-quality-v1', families=FAMILIES, learning_rates=LRS,
        seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS, widths=WIDTHS, eval_seed=2026,
        min_improvement=.005, max_trials=30, trial_seconds=2400, session_seconds=21600,
        max_disk_mb=4096, max_process_rss_mb=4096, max_mps_driver_mb=6144,
        evaluation_targets=111538, bootstrap_blocks=[2048,4096,8192], bootstrap_draws=2000,
        bootstrap_seed=20261024, primary_metric='H200_full_validation_context128',
        returned_half_life=200, max_model_evaluations=60, normalization_policy='rms_head32_pre_rope_v1')
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen QK RMS protocol changed')
    m, t = backbone(p['recipe'])
    if p['config']['model'] != m or p['config']['training'] != t:
        raise ValueError('Frozen QK RMS backbone changed')
    d = p['config']['data']
    if set(d) != {'path', 'train_fraction', 'val_fraction'} or d['train_fraction'] != .8 or d['val_fraction'] != .1:
        raise ValueError('Reserved data split changed')
    if p.get('screen_jobs') != [list(x) for x in local_jobs(p)]:
        raise ValueError('Frozen screening manifest changed')
    return p


def make_plan(root, prior_review):
    resolved = reviewed_horizon(prior_review)
    p = engine.make_plan(root, resolved['config'])
    p.update(study='qknorm-quality-v1', families=FAMILIES, learning_rates=LRS,
        seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS, widths=WIDTHS, eval_seed=2026,
        min_improvement=.005, max_trials=30, trial_seconds=2400, session_seconds=21600,
        max_disk_mb=4096, max_process_rss_mb=4096, max_mps_driver_mb=6144,
        evaluation_targets=111538, bootstrap_blocks=[2048,4096,8192], bootstrap_draws=2000,
        bootstrap_seed=20261024, primary_metric='H200_full_validation_context128',
        returned_half_life=200, max_model_evaluations=60, normalization_policy='rms_head32_pre_rope_v1',
        recipe={k:resolved[k] for k in ('family','total_steps','cooldown_start')},
        predecessor_review=resolved['review'],
        selection='Equal bounded family LR tuning at width64; freeze QK RMS if it beats tuned standard and incumbent, otherwise improved tuned standard if it beats incumbent, otherwise retain incumbent. Fresh64/256 confirmation; frozen winner only. Exact reference aliases only.',
        scope='Adaptive validation; equal updates/tokens and shared schedule/returned-weight policy within this round, not equalcompute, optimal muP transfer, independent holdout or seed-population significance.')
    p['screen_jobs'] = [list(x) for x in local_jobs(p)]
    validate_plan(p)
    p['runtime'] = _fingerprint(suite_spec(p, 'standard', .005))['runtime']
    p['worker_bootstrap_sha256'] = digest(BOOTSTRAP)
    return p


def suite_spec(p, family, lr):
    validate_plan(p)
    if family not in FAMILIES + [REFERENCE] or lr not in ([.005] if family == REFERENCE else LRS):
        raise ValueError('Unregistered QK RMS family or LR')
    cfg = copy.deepcopy(p['config'])
    cfg['training']['learning_rate'] = lr
    if family == 'qkrms':
        cfg['model'].update(qk_norm='rms_head32', depth_width_policy='depth4_qkrms_unscaled',
                            attention_width_policy='mha_head32_qkrms')
    arm = dict(root=p['source_root'], source_sha256=p['source_sha256'], config=cfg)
    return validate_suite(dict(version=1, hypothesis='Per-head Q/K RMS before RoPE under equal LR tuning',
        baseline=arm, candidate=copy.deepcopy(arm),
        targets=[dict(name=f'width-{w}', width=w, n_heads=w//32) for w in WIDTHS],
        seeds=SCREEN_SEEDS+CONFIRM_SEEDS, eval_seed=2026, min_improvement=.005, audit_test=False,
        data_sha256=p['data_sha256'], limits=dict(max_trials=30, trial_seconds=2400,
        session_seconds=21600, max_disk_mb=4096, max_failures=1, max_consecutive_failures=1)))


def endpoint(p, row, raw=False):
    return horizon.endpoint(p, row, raw)


def validate_rows(p, rows, jobs, stage):
    actual = [(r['family'], r['lr'], r['width'], r['seed']) for r in rows]
    if (sorted(actual) != sorted(map(tuple, jobs))
            or any(r['status'] != 'completed' or r['stage'] != stage for r in rows)):
        raise ValueError('Incomplete, duplicate or extra QK RMS evidence panel')
    for r in rows:
        if set(r.get('averaged_outputs', {})) != {'200'}:
            raise ValueError('Returned-weight panel changed')
        for raw in (False, True):
            validate_evaluation(endpoint(p, r, raw), p['evaluation_targets'])


def confirmation_manifest(choices):
    if set(choices) != set(FAMILIES) or any(c['lr'] not in LRS for c in choices.values()):
        raise ValueError('Invalid frozen LR choices')
    aliases = {REFERENCE:'standard'} if choices['standard']['lr'] == .005 else {}
    jobs = []
    for w in WIDTHS:
        for i, seed in enumerate(CONFIRM_SEEDS):
            families = FAMILIES + ([] if aliases else [REFERENCE])
            if i % 2:
                families = families[::-1]
            for f in families:
                jobs.append([f, .005 if f == REFERENCE else choices[f]['lr'], w, seed])
    if len(jobs) not in (12, 18) or len(set(map(tuple, jobs))) != len(jobs):
        raise ValueError('Invalid bounded confirmation manifest')
    return dict(jobs=jobs, reference_aliases=aliases, total_distinct_trials=12+len(jobs),
                total_model_evaluations=2*(12+len(jobs)))


def freeze_selection(p, rows):
    validate_plan(p)
    validate_rows(p, rows, local_jobs(p), 'local')
    choices = {}
    for f in FAMILIES:
        grid = [dict(lr=lr, mean_ce=mean(endpoint(p,r)['mean_ce'] for r in rows
                                       if r['family']==f and r['lr']==lr)) for lr in LRS]
        best = min(grid, key=lambda v:(v['mean_ce'],v['lr']))
        choices[f] = {**best, 'grid':grid, 'at_grid_boundary':best['lr'] in (LRS[0],LRS[-1])}
    def scores(f, lr):
        return {r['seed']:endpoint(p,r)['mean_ce'] for r in rows if r['family']==f and r['lr']==lr}
    a = scores('qkrms', choices['qkrms']['lr'])
    contrasts = []
    for ref, lr in [('standard',choices['standard']['lr']), (REFERENCE,.005)]:
        b = scores('standard',lr)
        ds = {str(s):a[s]-b[s] for s in SCREEN_SEEDS}
        contrasts.append(dict(reference=ref, reference_lr=lr, paired_seed_deltas=ds,
                              mean_delta=mean(ds.values()), supported=mean(ds.values())<-.005 and max(ds.values())<0))
    standard = scores('standard', choices['standard']['lr'])
    incumbent = scores('standard', .005)
    ds = {str(s):standard[s]-incumbent[s] for s in SCREEN_SEEDS}
    standard_contrast = dict(mean_delta=mean(ds.values()),paired_seed_deltas=ds,
        supported=mean(ds.values())<-.005 and max(ds.values())<0)
    qk_eligible = all(c['supported'] for c in contrasts)
    winner = 'qkrms' if qk_eligible else ('standard' if standard_contrast['supported'] else 'incumbent_unchanged')
    return dict(plan_sha256=digest(p), local_evidence_sha256=digest(rows), choices=choices,
                local_contrasts=contrasts, candidate_eligible=qk_eligible,
                standard_vs_incumbent=standard_contrast, winner=winner,
                confirmation=confirmation_manifest(choices),
                selected_by='Only width64 H200 means; ties lower LR. QK RMS requires gain>.005/every seed against both references; otherwise tuned standard may qualify against incumbent. Frozen winner only; no larger-width reselection.')


def summarize(p, state):
    validate_plan(p)
    rows = state['records']
    screen = [r for r in rows if r['stage']=='local']
    selection = freeze_selection(p,screen)
    if selection != state['selection']:
        raise ValueError('Frozen QK RMS selection changed')
    manifest = selection['confirmation']
    conf = [r for r in rows if r['stage']=='confirmation']
    validate_rows(p,conf,manifest['jobs'],'confirmation')
    if len(rows) != manifest['total_distinct_trials']:
        raise ValueError('QK RMS trajectory count changed')
    expanded = list(conf)
    for alias, family in manifest['reference_aliases'].items():
        expanded += [{**r,'family':alias,'alias_of':r['trial_id']} for r in conf if r['family']==family]
    comparisons = []
    for width in WIDTHS:
        for a,b in [('qkrms','standard'),('qkrms',REFERENCE),('standard',REFERENCE)]:
            def panel(f, raw):
                found = [r for r in expanded if r['family']==f and r['width']==width]
                if sorted(r['seed'] for r in found) != CONFIRM_SEEDS:
                    raise ValueError('Incomplete paired confirmation panel')
                return {r['seed']:endpoint(p,r,raw) for r in found}
            primary = [paired_blocks(panel(a,False),panel(b,False),n) for n in p['bootstrap_blocks']]
            v = primary[0]
            ok = v['mean_delta']<-.005 and max(v['paired_seed_deltas'].values())<0 and all(x['conditional_text_bootstrap_95pct'][1]<0 for x in primary)
            comparisons.append(dict(width=width,candidate=a,reference=b,sensitivity=primary,quality_improvement=ok,
                raw_sensitivity=[paired_blocks(panel(a,True),panel(b,True),n) for n in p['bootstrap_blocks']]))
    fixed_lr_screen = []
    for lr in LRS:
        entry = dict(lr=lr)
        for raw,key in [(False,'primary'),(True,'raw')]:
            panels = {f:{r['seed']:endpoint(p,r,raw)['mean_ce'] for r in screen if r['family']==f and r['lr']==lr} for f in FAMILIES}
            ds = {str(s):panels['qkrms'][s]-panels['standard'][s] for s in SCREEN_SEEDS}
            entry[key] = dict(mean_delta=mean(ds.values()),paired_seed_deltas=ds)
        fixed_lr_screen.append(entry)
    winner = selection['winner']
    checks = [c['quality_improvement'] for c in comparisons if c['candidate']==winner]
    adopt = winner!='incumbent_unchanged' and len(checks)==(4 if winner=='qkrms' else 2) and all(checks)
    return dict(plan_sha256=digest(p),selection=selection,records=rows,comparisons=comparisons,
        fixed_lr_screen=fixed_lr_screen,physical_training_trajectories=len(rows),
        model_evaluations=2*len(rows),reference_aliases=manifest['reference_aliases'],holdout_consumed=False,
        quality_decision=dict(frozen_screen_winner=winner,
                              qualifies_for_incumbent_replacement=adopt),
        adoption_requires_completion_review=True,scope=p['scope'])


def adapter(p, raw, checked, request, directory):
    return horizon.adapter(p,raw,checked,request,directory)


def verify_lineage(p):
    validate_plan(p)
    prior = p['predecessor_review']
    if file_digest(Path(prior['path'])) != prior['sha256']:
        raise ValueError('Final horizon review changed')
    resolved = reviewed_horizon(prior['path'])
    if ({k:resolved[k] for k in ('family','total_steps','cooldown_start')} != p['recipe']
            or resolved['config'] != p['config']):
        raise ValueError('Final adopted horizon backbone changed')
    root = Path(p['source_root'])
    if (source_digest(root) != p['source_sha256']
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() != p['git_commit']
            or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()):
        raise ValueError('Frozen QK RMS source changed or dirty')
    if file_digest(Path(p['config']['data']['path'])) != p['data_sha256']:
        raise ValueError('Corpus changed')
    if file_digest(Path(p['knowledge_snapshot']['path'])) != p['knowledge_snapshot']['sha256']:
        raise ValueError('Full knowledge snapshot changed')
    for a in p['artifacts']:
        if file_digest(Path(a['path'])) != a['sha256']:
            raise ValueError('Prerequisite artifact changed')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('Unset MPS fallback')
    if (digest(BOOTSTRAP) != p['worker_bootstrap_sha256']
            or _fingerprint(suite_spec(p,'standard',.005))['runtime'] != p['runtime']):
        raise ValueError('Worker or runtime changed')


def run(p, output):
    verify_lineage(p)
    output = Path(output)
    if json.loads((output/'plan.json').read_text()) != p:
        raise ValueError('Frozen run plan changed')
    def check(records):
        for r in records:
            if r['status'] != 'completed':
                raise ValueError('No automatic retry')
            d = output/'trials'/r['trial_id']
            for name in ('request','result'):
                if file_digest(d/f'{name}.json') != r[name+'_sha256']:
                    raise ValueError('Saved evidence changed')
            req = json.loads((d/'request.json').read_text())
            if req != _request(suite_spec(p,r['family'],r['lr']),r,d):
                raise ValueError('Saved request changed')
            raw = json.loads((d/'result.json').read_text())
            checked = validate_result(raw,req,dict(device='mps',training={**req['config']['training'],'block_size':128}),d)
            if any(r.get(k)!=v for k,v in {**checked,**adapter(p,raw,checked,req,d)}.items()):
                raise ValueError('Saved result changed')
    if (output/'state.json').exists():
        check(json.loads((output/'state.json').read_text())['records'])
    def spec(plan,f,lr):
        verify_lineage(plan)
        return suite_spec(plan,f,lr)
    def launch(*a,**kw):
        return _launch(*a,**kw,bootstrap=BOOTSTRAP)
    def adapt(*a):
        verify_lineage(p)
        return adapter(p,*a)
    def report(plan,state):
        verify_lineage(plan)
        check(state['records'])
        return summarize(plan,state)
    return engine.run(p,output,suite_builder=spec,job_builder=local_jobs,selection_builder=freeze_selection,
        report_builder=report,launcher=launch,result_adapter=adapt,
        completion_jobs=lambda plan,selection:selection['confirmation']['jobs'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-training',action='store_true')
    args = parser.parse_args()
    if not args.allow_training:
        parser.error('Explicit --allow-training required')
    run(json.loads(args.plan.read_text()),args.output)


if __name__ == '__main__':
    main()
