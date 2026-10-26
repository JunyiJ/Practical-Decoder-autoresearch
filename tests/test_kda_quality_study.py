"""Prospective finite KDA controller checks using synthetic scores only."""
import copy
from types import SimpleNamespace

import pytest

from src.attention.kda_scaling import resolve_alpha_dimension, validate_kda_contract
from src.research import kda_quality_study as study
from src.research.team.suite import _request
from tests.test_decay_quality_study import plan as decay_plan
from tests.test_qknorm_study import row
from tests.test_schedule_study import coverage


def plan(lr=.005, weight_decay=.00006):
    p = decay_plan()
    p.pop('decay_values'); p.pop('decay_policy')
    p.update(study='kda-quality-v1', families=study.FAMILIES,
             learning_rates=study.LRS, seeds=study.SCREEN_SEEDS,
             confirmation_seeds=study.CONFIRM_SEEDS, widths=study.WIDTHS,
             max_trials=30, max_model_evaluations=60,
             attention_policy='kda_head32_alpha_quarter',
             recipe=dict(learning_rate=lr, weight_decay=weight_decay))
    p['config']['model'], p['config']['training'] = study.backbone(p['recipe'])
    p['screen_jobs'] = [list(j) for j in study.local_jobs(p)]
    return study.validate_plan(p)


def screening(p, mha_lr=.005, best='kda'):
    means = dict(mha=2., kda=1.97) if best == 'kda' else dict(mha=1.96, kda=1.99)
    choices = dict(mha=mha_lr, kda=.0025)
    return [row(j, ce=means[j[0]] + (0 if j[1] == choices[j[0]] else .08))
            for j in study.local_jobs(p)]


def complete(p, mha_lr=.005, best='kda'):
    local = screening(p, mha_lr, best)
    selection = study.freeze_selection(p, local)
    means = dict(mha=2., kda=1.96, incumbent_reference=2.03)
    if best == 'mha':
        means.update(mha=1.96, kda=1.99)
    conf = [row(j, 'confirmation', ce=means[j[0]], raw_ce=4. if j[0] == best else 3.)
            for j in selection['confirmation']['jobs']]
    return dict(records=local+conf, selection=selection)


@pytest.mark.parametrize('weight_decay', study.decay.DECAYS.values())
@pytest.mark.parametrize('lr', study.LRS)
@pytest.mark.parametrize('family', study.FAMILIES+[study.REFERENCE])
def test_exact_common_training_and_explicit_attention_family(weight_decay, lr, family, tmp_path):
    p = plan(lr, weight_decay)
    spec = study.suite_spec(p, family, lr)
    assert spec['candidate']['config']['training'] == p['config']['training']
    assert len(study.local_jobs(p)) == 12
    for width in study.WIDTHS:
        job = dict(width=width, n_heads=width//32, arm='candidate',
                   seed=study.SCREEN_SEEDS[0], stage='local')
        req = _request(spec, job, tmp_path)
        m = req['config']['model']
        assert m['dim'] == width and m['n_heads'] == width//32
        if family == 'kda':
            assert 'dim_alpha' not in m  # Derived after target width is applied.
            cfg = SimpleNamespace(**m)
            validate_kda_contract(cfg)
            assert resolve_alpha_dimension(cfg) == width//4
            assert m['rope'] == 0 and m['attn_type'] == 'kda'
        else:
            assert m['attn_type'] == 'mha' and m['rope'] == 1
            assert 'dim_alpha' not in m
        t = req['config']['training']
        assert t['weight_decay'] == weight_decay
        assert t['max_iters']*t['batch_size']*m['block_size'] == 3072000
        assert t['weight_averaging']['half_lives'] == [200]


@pytest.mark.parametrize('mha_lr,total', [(.005, 24), (.0025, 30), (.00125, 30)])
def test_finite_schedule_exact_aliases_and_h200_only_selection(mha_lr, total):
    p = plan(); state = complete(p, mha_lr)
    selection = state['selection']; report = study.summarize(p, state)
    assert selection['winner'] == 'kda'
    assert report['quality_decision']['qualifies_for_incumbent_replacement']
    assert len(state['records']) == total == report['physical_training_trajectories']
    assert report['model_evaluations'] == 2*total
    assert report['reference_aliases'] == ({study.REFERENCE:'mha'} if mha_lr == .005 else {})
    assert len(report['comparisons']) == 6 and len(report['fixed_lr_screen']) == 3
    jobs = selection['confirmation']['jobs']
    assert {j[3] for j in jobs} == set(study.CONFIRM_SEEDS)
    assert not set(study.SCREEN_SEEDS) & set(study.CONFIRM_SEEDS)
    for c in report['comparisons']:
        if c['candidate'] == 'kda':
            assert c['quality_improvement']
            assert c['raw_sensitivity'][0]['mean_delta'] > 0


def test_tuned_mha_can_qualify_against_the_fixed_reference():
    p = plan(); state = complete(p, .0025, best='mha')
    assert state['selection']['winner'] == 'mha'
    assert study.summarize(p, state)['quality_decision']['qualifies_for_incumbent_replacement']


def test_failed_local_every_seed_gate_freezes_retention():
    p = plan(); rows = screening(p)
    for r in rows:
        if r['family'] == 'kda' and r['lr'] == .0025 and r['seed'] == 2262:
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(2.001)
    selected = study.freeze_selection(p, rows)
    assert selected['best_selected_arm'] == 'kda'
    assert selected['winner'] == 'incumbent_unchanged'
    assert not selected['candidate_eligible']


def test_larger_width_failure_prevents_adoption_and_cannot_change_selection():
    p = plan(); state = complete(p)
    frozen = copy.deepcopy(state['selection'])
    for r in state['records']:
        if r['stage'] == 'confirmation' and r['family'] == 'kda' and r['width'] == 256:
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(2.1)
    report = study.summarize(p, state)
    assert report['selection'] == frozen
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    state['selection']['winner'] = 'mha'
    with pytest.raises(ValueError, match='Frozen'):
        study.summarize(p, state)


@pytest.mark.parametrize('key,value', [('max_trials',31), ('trial_seconds',2401),
    ('session_seconds',21601), ('max_disk_mb',4097), ('returned_half_life',50),
    ('widths',[64,128,256]), ('seeds',[2241,2242]), ('confirmation_seeds',[2251,2252,2253]),
    ('attention_policy','unreviewed'), ('evaluation_targets',111537),
    ('learning_rates',[.0025,.005,.01]), ('max_model_evaluations',61)])
def test_frozen_grid_and_resource_contract_cannot_expand(key, value):
    p = plan(); p[key] = value
    with pytest.raises(ValueError):
        study.validate_plan(p)


@pytest.mark.parametrize('change', ['missing','duplicate','confirmation','wide'])
def test_screen_rejects_incomplete_or_out_of_scope_evidence(change):
    p = plan(); rows = screening(p)
    if change == 'missing': rows.pop()
    elif change == 'duplicate': rows.append(copy.deepcopy(rows[0]))
    elif change == 'confirmation': rows[0]['seed'] = study.CONFIRM_SEEDS[0]
    else: rows[0]['width'] = 256
    with pytest.raises(ValueError):
        study.freeze_selection(p, rows)
