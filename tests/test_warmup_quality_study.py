"""Synthetic controller evidence only; no model execution or real quality scores."""
import copy
import pytest

from src.research import warmup_quality_study as study
from src.research.team.suite import _request
from src.train.lr_schedule import multiplier, schedule_contract
from omegaconf import OmegaConf
from tests.test_kda_quality_study import plan as kda_plan
from tests.test_qknorm_study import row
from tests.test_schedule_study import coverage


def plan(attention='mha', lr=.005, decay=.00006):
    p = kda_plan(lr, decay)
    p.pop('attention_policy')
    p.update(study='warmup-quality-v1', families=study.FAMILIES,
        warmup_steps=study.WARMUP_STEPS, initial_ratio=.1,
        seeds=study.SCREEN_SEEDS, confirmation_seeds=study.CONFIRM_SEEDS,
        learning_rates=[lr], max_trials=24, max_model_evaluations=48,
        schedule_policy='fixed_peak_warmup100_300_cooldown2000_v1',
        recipe=dict(attention_family=attention, learning_rate=lr, weight_decay=decay),
        scope='Synthetic warmup controller fixture, not empirical evidence.')
    p['config']['model'], p['config']['training'] = study.backbone(p['recipe'])
    p['screen_jobs'] = [list(j) for j in study.local_jobs(p)]
    return study.validate_plan(p)


def complete(p, winner='warmup100'):
    means = dict(baseline=2., warmup100=1.97, warmup300=1.98)
    if winner == 'warmup300': means.update(warmup100=1.98, warmup300=1.97)
    if winner == 'baseline': means.update(baseline=1.96)
    screen = [row(j, ce=means[j[0]]) for j in study.local_jobs(p)]
    selection = study.freeze_selection(p, screen)
    conf = [row(j, 'confirmation', ce=means[j[0]], raw_ce=4. if j[0]==winner else 3.)
            for j in selection['confirmation']['jobs']]
    return dict(records=screen+conf, selection=selection)


@pytest.mark.parametrize('attention', ['mha','kda'])
@pytest.mark.parametrize('lr', study.kda.LRS)
@pytest.mark.parametrize('arm', study.FAMILIES)
def test_only_early_schedule_changes_with_width_and_family_preserved(attention, lr, arm, tmp_path):
    p = plan(attention, lr)
    spec = study.suite_spec(p, arm, lr)
    cfg = spec['candidate']['config']
    assert cfg['model'] == p['config']['model']
    t, original = copy.deepcopy(cfg['training']), copy.deepcopy(p['config']['training'])
    schedule, baseline = t.pop('lr_schedule'), original.pop('lr_schedule')
    assert t == original
    if arm == 'baseline': assert schedule == baseline
    else:
        expected = {**baseline, 'kind':'warmup_linear_cooldown',
                    'warmup_steps':study.WARMUP_STEPS[arm], 'initial_ratio':.1}
        assert schedule == expected
    validated = schedule_contract(OmegaConf.create(cfg['training']))
    assert multiplier(validated, 0) == (1. if arm=='baseline' else .1)
    assert multiplier(validated, 2999) == .1
    for width in study.WIDTHS:
        job = dict(width=width, n_heads=width//32, arm='candidate', seed=2281, stage='local')
        req = _request(spec, job, tmp_path)
        assert req['config']['model']['dim'] == width
        assert req['config']['model']['attn_type'] == attention
        assert 'dim_alpha' not in req['config']['model']
        assert req['config']['training']['max_iters']*8*128 == 3072000
        assert req['config']['training']['weight_averaging']['half_lives'] == [200]


@pytest.mark.parametrize('winner', study.FAMILIES)
def test_fixed_peak_no_alias_manifest_and_primary_only_decision(winner):
    p = plan(); state = complete(p, winner)
    report = study.summarize(p, state)
    assert report['physical_training_trajectories'] == len(state['records']) == 24
    assert report['model_evaluations'] == 48 and report['reference_aliases'] == {}
    assert len(study.local_jobs(p)) == 6
    assert len(state['selection']['confirmation']['jobs']) == 18
    assert report['quality_decision']['qualifies_for_incumbent_replacement'] == (winner != 'baseline')
    assert len(report['comparisons']) == 6 and len(report['fixed_lr_screen']) == 3
    assert not set(study.SCREEN_SEEDS) & set(study.CONFIRM_SEEDS)
    assert {r['lr'] for r in state['records']} == {.005}
    for c in report['comparisons']:
        if c['candidate'] == winner and c['reference'] == 'baseline':
            assert c['quality_improvement'] and c['raw_sensitivity'][0]['mean_delta'] > 0


def test_ties_prefer_baseline_then_shorter_warmup():
    p = plan()
    rows = [row(j, ce=2.) for j in study.local_jobs(p)]
    assert study.freeze_selection(p, rows)['winner'] == 'incumbent_unchanged'
    rows = [row(j, ce=2. if j[0]=='baseline' else 1.97) for j in study.local_jobs(p)]
    assert study.freeze_selection(p, rows)['winner'] == 'warmup100'


def test_best_bad_seed_does_not_fall_back_to_other_eligible_duration():
    p = plan()
    rows = []
    for j in study.local_jobs(p):
        v = 2. if j[0]=='baseline' else 1.98
        if j[0]=='warmup100': v = 1.9 if j[3]==2281 else 2.001
        rows.append(row(j, ce=v))
    s = study.freeze_selection(p, rows)
    assert s['best_selected_arm'] == 'warmup100'
    assert not s['candidate_eligible'] and s['winner'] == 'incumbent_unchanged'


@pytest.mark.parametrize('case', ['bad_seed','small_mean','conditional_upper'])
def test_each_confirmation_gate_blocks_adoption_without_reselection(case, monkeypatch):
    p = plan(); state = complete(p); frozen = copy.deepcopy(state['selection'])
    if case == 'conditional_upper':
        paired = study.paired_blocks
        def altered(a,b,n):
            result = paired(a,b,n)
            if n == 8192: result['conditional_text_bootstrap_95pct'][1] = .001
            return result
        monkeypatch.setattr(study, 'paired_blocks', altered)
    else:
        for r in state['records']:
            if r['stage']=='confirmation' and r['family']=='warmup100' and r['width']==256:
                v = (2.001 if r['seed']==2293 else 1.97) if case=='bad_seed' else 1.998
                r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(v)
    report = study.summarize(p,state)
    assert report['selection'] == frozen
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    assert any(c['quality_improvement'] for c in report['comparisons'] if
               c['candidate']=='warmup300' and c['reference']=='baseline') if case!='conditional_upper' else True


@pytest.mark.parametrize('key,value', [('max_trials',25),('trial_seconds',2401),
    ('session_seconds',21601),('max_disk_mb',4097),('returned_half_life',50),
    ('widths',[64,128,256]),('seeds',[2281,2291]),('confirmation_seeds',[2281,2292,2293]),
    ('schedule_policy','changed'),('initial_ratio',0.),('evaluation_targets',111537),
    ('learning_rates',[.0025,.005]),('max_model_evaluations',49),
    ('warmup_steps',{'baseline':0,'warmup100':100,'warmup300':500})])
def test_protocol_and_caps_reject_changes(key,value):
    p = plan(); p[key]=value
    with pytest.raises(ValueError): study.validate_plan(p)


@pytest.mark.parametrize('case', ['missing','duplicate','confirmation_seed','large_width','extra_output','raw_nonfinite'])
def test_screen_rejects_wrong_panel(case):
    p = plan(); rows = [row(j) for j in study.local_jobs(p)]
    if case=='missing': rows.pop()
    elif case=='duplicate': rows.append(copy.deepcopy(rows[0]))
    elif case=='confirmation_seed': rows[0]['seed']=2291
    elif case=='large_width': rows[0]['width']=256
    elif case=='extra_output': rows[0]['averaged_outputs']['50']=copy.deepcopy(rows[0]['averaged_outputs']['200'])
    else: rows[0]['coverage']['128']['mean_ce']=float('nan')
    with pytest.raises(ValueError): study.freeze_selection(p, rows)


def test_frozen_selection_and_fixed_peak_cannot_be_changed():
    p=plan(); s=complete(p); s['selection']['winner']='warmup300'
    with pytest.raises(ValueError): study.summarize(p,s)
    with pytest.raises(ValueError): study.suite_spec(p,'warmup100',.0025)
