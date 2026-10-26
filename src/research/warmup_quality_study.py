"""Bounded fixed-peak warmup comparison after the independently reviewed KDA round."""
import argparse
import copy
import json
import os
from pathlib import Path
from statistics import mean
import subprocess

from . import horizon_study as horizon, optimizer_study as engine
from . import kda_quality_study as kda
from .coverage_eval import paired_blocks, validate_evaluation
from .protocol import digest, file_digest
from .runner import validate_result
from .team.suite import validate_suite, _request, _launch, _fingerprint, source_digest
from .warmup_quality_predecessor import reviewed_kda

FAMILIES = ['baseline', 'warmup100', 'warmup300']
WARMUP_STEPS = dict(baseline=0, warmup100=100, warmup300=300)
SCREEN_SEEDS = [2281, 2282]
CONFIRM_SEEDS = [2291, 2292, 2293]
WIDTHS = [64, 256]
BOOTSTRAP = horizon.BOOTSTRAP


def local_jobs(p):
    lr = p['recipe']['learning_rate']
    return [(f, lr, 64, seed) for i, seed in enumerate(SCREEN_SEEDS)
            for f in (FAMILIES if i % 2 == 0 else FAMILIES[::-1])]


def backbone(recipe):
    if (not isinstance(recipe, dict)
            or set(recipe) != {'attention_family', 'learning_rate', 'weight_decay'}
            or recipe['attention_family'] not in kda.FAMILIES):
        raise ValueError('Invalid final adopted KDA/MHA recipe')
    model, training = kda.backbone({k: recipe[k] for k in ('learning_rate', 'weight_decay')})
    if recipe['attention_family'] == 'kda':
        model.update(attn_type='kda', rope=0, attn_bias=True, causal=True,
            attention_width_policy='kda_head32_alpha_quarter', depth_width_policy='depth4_kda_unscaled')
    return model, training


def validate_plan(p):
    expected = dict(study='warmup-quality-v1', families=FAMILIES,
        warmup_steps=WARMUP_STEPS, initial_ratio=.1, seeds=SCREEN_SEEDS,
        confirmation_seeds=CONFIRM_SEEDS, widths=WIDTHS, eval_seed=2026,
        min_improvement=.005, max_trials=24, trial_seconds=2400, session_seconds=21600,
        max_disk_mb=4096, max_process_rss_mb=4096, max_mps_driver_mb=6144,
        evaluation_targets=111538, bootstrap_blocks=[2048,4096,8192], bootstrap_draws=2000,
        bootstrap_seed=20261024, primary_metric='H200_full_validation_context128',
        returned_half_life=200, max_model_evaluations=48,
        schedule_policy='fixed_peak_warmup100_300_cooldown2000_v1')
    if any(p.get(k) != v for k, v in expected.items()):
        raise ValueError('Frozen warmup-quality protocol changed')
    model, training = backbone(p['recipe'])
    if (p['config']['model'] != model or p['config']['training'] != training
            or p.get('learning_rates') != [p['recipe']['learning_rate']]):
        raise ValueError('Frozen warmup backbone or fixed peak learning rate changed')
    d = p['config']['data']
    if set(d) != {'path', 'train_fraction', 'val_fraction'} or d['train_fraction'] != .8 or d['val_fraction'] != .1:
        raise ValueError('Reserved data split changed')
    if p.get('screen_jobs') != [list(x) for x in local_jobs(p)]:
        raise ValueError('Frozen warmup screening manifest changed')
    return p


def make_plan(root, prior_review):
    resolved = reviewed_kda(prior_review)
    p = engine.make_plan(root, resolved['config'])
    p.update(study='warmup-quality-v1', families=FAMILIES, warmup_steps=WARMUP_STEPS,
        initial_ratio=.1, learning_rates=[resolved['learning_rate']],
        seeds=SCREEN_SEEDS, confirmation_seeds=CONFIRM_SEEDS, widths=WIDTHS, eval_seed=2026,
        min_improvement=.005, max_trials=24, trial_seconds=2400, session_seconds=21600,
        max_disk_mb=4096, max_process_rss_mb=4096, max_mps_driver_mb=6144,
        evaluation_targets=111538, bootstrap_blocks=[2048,4096,8192], bootstrap_draws=2000,
        bootstrap_seed=20261024, primary_metric='H200_full_validation_context128',
        returned_half_life=200, max_model_evaluations=48,
        schedule_policy='fixed_peak_warmup100_300_cooldown2000_v1',
        recipe={k:resolved[k] for k in ('attention_family','learning_rate','weight_decay')},
        predecessor_review=resolved['review'],
        selection='Fixed common peak LR. Freeze best width64 H200 schedule, ties baseline then shorter warmup; require gain>.005 and every seed versus baseline. Only frozen eligible warmup can adopt after fresh64/256 confirmation.',
        scope='Adaptive validation at a fixed inherited peak LR, not independently optimized schedules or empirical optimal muP transfer. Equal updates/targets and nominal cumulative independent decay; warmup reduces LR integral and changes update/decay balance. Raw/H200 outputs are correlated, holdout closed, no stability-cause or inference-speed claim.')
    p['screen_jobs'] = [list(x) for x in local_jobs(p)]
    validate_plan(p)
    p['runtime'] = _fingerprint(suite_spec(p, 'baseline', p['recipe']['learning_rate']))['runtime']
    p['worker_bootstrap_sha256'] = digest(BOOTSTRAP)
    return p


def suite_spec(p, family, lr):
    validate_plan(p)
    if family not in FAMILIES or lr != p['recipe']['learning_rate']:
        raise ValueError('Unregistered warmup family or changed fixed peak LR')
    cfg = copy.deepcopy(p['config'])
    if family != 'baseline':
        cfg['training']['lr_schedule'].update(kind='warmup_linear_cooldown',
            warmup_steps=WARMUP_STEPS[family], initial_ratio=.1)
    arm = dict(root=p['source_root'], source_sha256=p['source_sha256'], config=cfg)
    return validate_suite(dict(version=1, hypothesis='Early LR warmup at the reviewed fixed peak followed by unchanged cooldown/H200',
        baseline=arm, candidate=copy.deepcopy(arm),
        targets=[dict(name=f'width-{w}', width=w, n_heads=w//32) for w in WIDTHS],
        seeds=SCREEN_SEEDS+CONFIRM_SEEDS, eval_seed=2026, min_improvement=.005, audit_test=False,
        data_sha256=p['data_sha256'], limits=dict(max_trials=24, trial_seconds=2400,
        session_seconds=21600, max_disk_mb=4096, max_failures=1, max_consecutive_failures=1)))


def endpoint(p, row, raw=False):
    return horizon.endpoint(p, row, raw)


def validate_rows(p, rows, jobs, stage):
    actual = [(r['family'], r['lr'], r['width'], r['seed']) for r in rows]
    if (sorted(actual) != sorted(map(tuple, jobs))
            or any(r['status'] != 'completed' or r['stage'] != stage for r in rows)):
        raise ValueError('Incomplete, duplicate or extra warmup-quality evidence panel')
    for r in rows:
        if set(r.get('averaged_outputs', {})) != {'200'}:
            raise ValueError('Returned-weight panel changed')
        for raw in (False, True):
            validate_evaluation(endpoint(p, r, raw), p['evaluation_targets'])


def confirmation_manifest(p):
    lr = p['recipe']['learning_rate']
    jobs = [[f, lr, w, seed] for w in WIDTHS for i, seed in enumerate(CONFIRM_SEEDS)
            for f in (FAMILIES if i % 2 == 0 else FAMILIES[::-1])]
    if len(jobs) != 18 or len(set(map(tuple, jobs))) != 18:
        raise ValueError('Invalid bounded warmup confirmation manifest')
    return dict(jobs=jobs, reference_aliases={}, total_distinct_trials=24,
                total_model_evaluations=48)


def freeze_selection(p, rows):
    validate_plan(p)
    validate_rows(p, rows, local_jobs(p), 'local')
    means = {f: mean(endpoint(p, r)['mean_ce'] for r in rows if r['family'] == f) for f in FAMILIES}
    best = min(FAMILIES, key=lambda f:(means[f], FAMILIES.index(f)))
    def scores(f):
        return {r['seed']:endpoint(p, r)['mean_ce'] for r in rows if r['family'] == f}
    a, b = scores(best), scores('baseline')
    ds = {str(s):a[s]-b[s] for s in SCREEN_SEEDS}
    eligible = best != 'baseline' and mean(ds.values()) < -.005 and max(ds.values()) < 0
    return dict(plan_sha256=digest(p), local_evidence_sha256=digest(rows),
        fixed_peak_lr=p['recipe']['learning_rate'], means=means, best_selected_arm=best,
        local_contrast=dict(candidate=best, reference='baseline', paired_seed_deltas=ds,
                            mean_delta=mean(ds.values()), supported=eligible),
        candidate_eligible=eligible, winner=best if eligible else 'incumbent_unchanged',
        confirmation=confirmation_manifest(p),
        selected_by='Only width64 H200 means, ties baseline then shorter warmup. Best warmup requires gain>.005/every seed versus baseline, no alternate fallback or larger-width reselection.')


def summarize(p, state):
    validate_plan(p)
    rows = state['records']
    screen = [r for r in rows if r['stage'] == 'local']
    selection = freeze_selection(p, screen)
    if selection != state['selection']:
        raise ValueError('Frozen warmup selection changed')
    conf = [r for r in rows if r['stage'] == 'confirmation']
    validate_rows(p, conf, selection['confirmation']['jobs'], 'confirmation')
    if len(rows) != 24:
        raise ValueError('Warmup-quality trajectory count changed')
    comparisons = []
    pairs = [('warmup100','baseline'), ('warmup300','baseline'), ('warmup300','warmup100')]
    for width in WIDTHS:
        for a, b in pairs:
            def panel(f, raw):
                found = [r for r in conf if r['family'] == f and r['width'] == width]
                if sorted(r['seed'] for r in found) != CONFIRM_SEEDS:
                    raise ValueError('Incomplete paired confirmation panel')
                return {r['seed']:endpoint(p, r, raw) for r in found}
            primary = [paired_blocks(panel(a,False), panel(b,False), n) for n in p['bootstrap_blocks']]
            v = primary[0]
            ok = v['mean_delta'] < -.005 and max(v['paired_seed_deltas'].values()) < 0 and all(
                x['conditional_text_bootstrap_95pct'][1] < 0 for x in primary)
            comparisons.append(dict(width=width, candidate=a, reference=b,
                sensitivity=primary, quality_improvement=ok,
                raw_sensitivity=[paired_blocks(panel(a,True), panel(b,True), n) for n in p['bootstrap_blocks']]))
    screen_contrasts = []
    for a, b in pairs:
        entry = dict(lr=p['recipe']['learning_rate'], candidate=a, reference=b)
        for raw, key in [(False,'primary'), (True,'raw')]:
            panels = {f:{r['seed']:endpoint(p,r,raw)['mean_ce'] for r in screen if r['family']==f} for f in (a,b)}
            ds = {str(s):panels[a][s]-panels[b][s] for s in SCREEN_SEEDS}
            entry[key] = dict(mean_delta=mean(ds.values()), paired_seed_deltas=ds)
        screen_contrasts.append(entry)
    winner = selection['winner']
    checks = [c['quality_improvement'] for c in comparisons if c['candidate']==winner and c['reference']=='baseline']
    adopt = winner != 'incumbent_unchanged' and len(checks)==2 and all(checks)
    return dict(plan_sha256=digest(p), selection=selection, records=rows, comparisons=comparisons,
        fixed_lr_screen=screen_contrasts, physical_training_trajectories=len(rows),
        model_evaluations=2*len(rows), reference_aliases={}, holdout_consumed=False,
        quality_decision=dict(frozen_screen_winner=winner, qualifies_for_incumbent_replacement=adopt),
        adoption_requires_completion_review=True, scope=p['scope'])


def adapter(p, raw, checked, request, directory):
    return horizon.adapter(p, raw, checked, request, directory)


def verify_lineage(p):
    validate_plan(p)
    prior = p['predecessor_review']
    if file_digest(Path(prior['path'])) != prior['sha256']:
        raise ValueError('Final KDA review changed')
    resolved = reviewed_kda(prior['path'])
    if ({k:resolved[k] for k in ('attention_family','learning_rate','weight_decay')} != p['recipe']
            or resolved['config'] != p['config']):
        raise ValueError('Final adopted KDA/MHA backbone changed')
    root = Path(p['source_root'])
    if (source_digest(root) != p['source_sha256']
            or subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip() != p['git_commit']
            or subprocess.check_output(['git','status','--porcelain'],cwd=root,text=True).strip()):
        raise ValueError('Frozen warmup source changed or dirty')
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
            or _fingerprint(suite_spec(p,'baseline',p['recipe']['learning_rate']))['runtime'] != p['runtime']):
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
