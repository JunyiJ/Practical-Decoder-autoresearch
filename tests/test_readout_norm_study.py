"""Synthetic selection/lineage budgets; no training or corpus evaluations."""
import copy
import json
import pytest

from src.research import readout_norm_study as study, optimizer_study as engine
from src.research.runner import atomic_json
from tests.test_embedding_scale_study import plan as embedding_plan
from tests.test_qknorm_study import row
from tests.test_schedule_study import coverage


def plan(input_scale=1., incumbent_lr=.005):
    p = embedding_plan()
    for key in ('input_scales','input_scale_policy'):
        p.pop(key)
    p.update(study='readout-norm-quality-v1',families=study.FAMILIES,
        learning_rates=study.LRS,seeds=study.SCREEN_SEEDS,confirmation_seeds=study.CONFIRM_SEEDS,
        widths=study.WIDTHS,max_trials=42,max_model_evaluations=84,
        readout_norm_kinds=study.KINDS,readout_norm_policy='mha4_readout_norm',readout_norm_epsilon=1e-5,
        recipe=dict(warmup_steps=0,learning_rate=incumbent_lr,weight_decay=.00006,input_embedding_scale=input_scale))
    p['config']['model'],p['config']['training']=study.backbone(p['recipe'])
    p['screen_jobs']=[list(x) for x in study.local_jobs(p)]
    return study.validate_plan(p)


def screening(p, layer_affine_lr=None, best='rms'):
    layer_affine_lr = p['recipe']['learning_rate'] if layer_affine_lr is None else layer_affine_lr
    choices = dict(layer_affine=layer_affine_lr, layer_no_bias=.01, rms=.0025)
    means = dict(layer_affine=2., layer_no_bias=1.99, rms=1.97)
    if best == 'layer_no_bias':
        means.update(layer_no_bias=1.97, rms=1.99)
    elif best == 'layer_affine':
        means.update(layer_affine=1.97, layer_no_bias=2.03, rms=2.04)
    out = []
    for j in study.local_jobs(p):
        f, lr, _, seed = j
        value = means[f] + (0 if lr == choices[f] else .08)
        out.append(row(j, ce=value))
    return out


def complete(p, layer_affine_lr=None, best='rms'):
    local = screening(p, layer_affine_lr, best)
    selection = study.freeze_selection(p, local)
    vals = dict(layer_affine=2., layer_no_bias=1.98, rms=1.96, incumbent_reference=2.02)
    if best == 'layer_no_bias':
        vals.update(layer_no_bias=1.96, rms=1.98)
    if best == 'layer_affine':
        vals.update(layer_affine=1.96, layer_no_bias=1.98, rms=1.95)
    conf = [row(j, 'confirmation', ce=vals[j[0]], raw_ce=4. if j[0] == best else 3.) for j in selection['confirmation']['jobs']]
    return dict(records=local + conf, selection=selection)


@pytest.mark.parametrize('input_scale',[1.,.5,.25])
@pytest.mark.parametrize('incumbent_lr',[.0025,.005,.01])
@pytest.mark.parametrize('arm',study.FAMILIES+[study.REFERENCE])
def test_only_final_norm_and_peak_lr_change(input_scale,incumbent_lr,arm):
    p=plan(input_scale,incumbent_lr);c=study.suite_spec(p,arm,incumbent_lr)['candidate']['config']
    m=copy.deepcopy(c['model'])
    assert m.pop('readout_norm_width_policy')=='mha4_readout_norm'
    assert m.pop('readout_norm_kind')==('layer_affine' if arm==study.REFERENCE else arm)
    assert m.pop('readout_norm_epsilon')==1e-5
    assert m==p['config']['model'] and c['training']==p['config']['training']
    assert len(study.local_jobs(p))==18
    assert c['training']['max_iters']*8*128==3072000
    assert c['training']['weight_averaging']['half_lives']==[200]


@pytest.mark.parametrize('layer_affine_lr,count',[(.005,36),(.0025,42),(.01,42)])
def test_exact_reference_alias_and_complete_correlated_output_count(layer_affine_lr,count):
    incumbent_lr=.005;p=plan();s=complete(p,layer_affine_lr)
    r = study.summarize(p, s)
    assert len(s['records']) == r['physical_training_trajectories'] == count
    assert r['model_evaluations'] == 2 * count
    assert r['reference_aliases'] == ({study.REFERENCE:'layer_affine'} if incumbent_lr == layer_affine_lr else {})
    assert r['quality_decision']['qualifies_for_incumbent_replacement']
    assert len(r['comparisons']) == 12 and len(r['fixed_lr_screen']) == 9
    for c in r['comparisons']:
        if c['candidate'] == 'rms':
            assert c['quality_improvement'] and c['raw_sensitivity'][0]['mean_delta'] > 0
        if c['candidate'] == 'layer_affine' and c['reference'] == study.REFERENCE and incumbent_lr == layer_affine_lr:
            assert all(x['conditional_text_bootstrap_95pct'] == [0., 0.] for x in c['sensitivity'])


@pytest.mark.parametrize('best', study.FAMILIES)
def test_each_frozen_family_can_qualify_without_alternative_promotion(best):
    p = plan(); s = complete(p, .0025 if best == 'layer_affine' else .005, best)
    assert s['selection']['winner'] == best
    report = study.summarize(p, s)
    assert report['quality_decision'] == dict(frozen_screen_winner=best, qualifies_for_incumbent_replacement=True)
    for r in s['records']:
        if r['stage'] == 'confirmation' and r['family'] == best and r['width'] == 256 and r['seed'] == 2333:
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(2.1)
    report = study.summarize(p, s)
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']


def test_lower_lr_and_simpler_arm_win_exact_ties():
    p = plan(); rows = [row(j, ce=2.) for j in study.local_jobs(p)]
    sel = study.freeze_selection(p, rows)
    assert all(c['lr'] == .0025 for c in sel['choices'].values())
    assert sel['best_selected_arm'] == 'layer_affine' and sel['winner'] == 'incumbent_unchanged'


def test_bad_screen_seed_cannot_be_rescued_by_other_arm_or_confirmation():
    p = plan(); s = complete(p)
    for r in s['records'][:18]:
        if r['family'] == 'rms' and r['lr'] == .0025:
            value = 1.92 if r['seed'] == 2321 else 2.001
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(value)
    s['selection'] = study.freeze_selection(p, s['records'][:18])
    assert s['selection']['best_selected_arm'] == 'rms'
    assert s['selection']['winner'] == 'incumbent_unchanged'
    assert not study.summarize(p, s)['quality_decision']['qualifies_for_incumbent_replacement']


@pytest.mark.parametrize('reference', ['layer_affine', study.REFERENCE])
def test_nondefault_must_beat_both_references_on_both_widths(reference):
    p = plan(); s = complete(p, .0025)
    for r in s['records']:
        if r['stage'] == 'confirmation' and r['family'] == reference and r['width'] == 256 and r['seed'] == 2333:
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(1.95)
    assert not study.summarize(p, s)['quality_decision']['qualifies_for_incumbent_replacement']


def test_other_candidate_comparison_is_reported_but_not_a_hidden_adoption_gate():
    p = plan(); s = complete(p)
    for r in s['records']:
        if r['stage'] == 'confirmation' and r['family'] == 'layer_no_bias':
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(1.95)
    report = study.summarize(p, s)
    assert report['quality_decision']['qualifies_for_incumbent_replacement']
    assert not any(c['quality_improvement'] for c in report['comparisons'] if c['candidate']=='rms' and c['reference']=='layer_no_bias')


def test_text_interval_can_veto_negative_means_and_every_seed():
    p = plan(); s = complete(p)
    for r in s['records']:
        if r['stage'] != 'confirmation' or r['family'] != 'rms':continue
        v = coverage(1.994); counts = v['block_target_counts']
        offsets = [.3 if i < len(counts)//2 else -.3 for i in range(len(counts))]
        center = sum(x*c for x,c in zip(offsets, counts))/sum(counts)
        v['block_loss_sums'] = [(1.994+x-center)*c for x,c in zip(offsets, counts)]
        v['mean_ce'] = sum(v['block_loss_sums'])/sum(counts)
        r['averaged_outputs']['200']['coverage']['evaluations']['128'] = v
    report = study.summarize(p, s)
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']
    assert any(c['sensitivity'][0]['mean_delta'] < -.005 and max(c['sensitivity'][0]['paired_seed_deltas'].values()) < 0 and any(x['conditional_text_bootstrap_95pct'][1] > 0 for x in c['sensitivity']) for c in report['comparisons'])


@pytest.mark.parametrize('change', ['missing','duplicate','seed','stage','failed','extra_average','coverage'])
def test_incomplete_or_mislabelled_screen_evidence_rejected(change):
    p = plan(); rs = screening(p)
    if change == 'missing':rs.pop()
    elif change == 'duplicate':rs.append(rs[0])
    elif change == 'seed':rs[0]['seed'] = 2331
    elif change == 'stage':rs[0]['stage'] = 'confirmation'
    elif change == 'failed':rs[0]['status'] = 'failed'
    elif change == 'extra_average':rs[0]['averaged_outputs']['50'] = rs[0]['averaged_outputs']['200']
    else:rs[0]['coverage']['128']['target_count'] = 1
    with pytest.raises(ValueError):study.freeze_selection(p, rs)


@pytest.mark.parametrize('change', ['manifest','choices','eligibility','winner','alias','extra','missing'])
def test_confirmation_cannot_modify_choices_or_aliases(change):
    p = plan(); s = complete(p)
    if change == 'manifest':s['selection']['confirmation']['jobs'].pop()
    elif change == 'choices':s['selection']['choices']['layer_no_bias']['lr'] = .005
    elif change == 'eligibility':s['selection']['candidate_eligible'] = False
    elif change == 'winner':s['selection']['winner'] = 'layer_no_bias'
    elif change == 'alias':s['selection']['confirmation']['reference_aliases'] = {}
    elif change == 'extra':s['records'].append(s['records'][-1])
    else:s['records'].pop()
    with pytest.raises(ValueError):study.summarize(p, s)


@pytest.mark.parametrize('field,value', [('max_trials',43),('returned_half_life',50),('widths',[64,128,256]),
    ('bootstrap_draws',100),('min_improvement',0.),('max_model_evaluations',85),('trial_seconds',9999),
    ('readout_norm_policy','arbitrary'),('readout_norm_epsilon',1e-6),('seeds',[2321,2331])])
def test_protocol_extensions_rejected(field, value):
    p = plan(); p[field] = value
    with pytest.raises(ValueError):study.validate_plan(p)


@pytest.mark.parametrize('change', ['schedule','averaging','model','recipe','tokens'])
def test_unreviewed_backbone_rejected(change):
    p = plan()
    if change == 'schedule':p['config']['training']['lr_schedule']['start_step'] = 1990
    elif change == 'averaging':p['config']['training']['weight_averaging']['half_lives'] = [50]
    elif change == 'model':p['config']['model']['ffn_hidden_norm'] = 'rms'
    elif change == 'recipe':p['recipe']['input_embedding_scale'] = .75
    else:p['config']['training']['max_iters'] = 6000
    with pytest.raises(ValueError):study.validate_plan(p)


@pytest.mark.parametrize('layer_affine_lr', [.005,.0025])
def test_real_serial_engine_freezes_before_confirmation_and_stops_first_failure(tmp_path, monkeypatch, layer_affine_lr):
    p = plan(); s = complete(p, layer_affine_lr); fixtures = s['records']; calls = []
    monkeypatch.setattr(engine, '_preflight', lambda s:None)
    monkeypatch.setattr(engine, '_verify_checkpoints', lambda *a:None)
    monkeypatch.setattr(engine, '_fingerprint', lambda s:dict(runtime=dict(packages=dict(torch='x'))))
    monkeypatch.setattr(engine, '_request', lambda spec,r,d:dict(cpu_threads=4,config=dict(training=dict(device='mps'),model=dict(block_size=128))))
    monkeypatch.setattr(engine, 'validate_result', lambda raw,*a:raw['row'])
    def launch(request, output, *a, **kw):
        i = len(calls)
        if i >= 18:
            saved = json.loads((output/'selection.json').read_text())
            assert saved['choices'] == s['selection']['choices'] and saved['confirmation'] == s['selection']['confirmation']
        calls.append(i)
        atomic_json(request.parent/'result.json', dict(row=fixtures[i],runtime=dict(torch='x',cpu_threads=4,requested_device='mps')))
        return 'completed', None
    kw = dict(suite_builder=lambda *a:{}, job_builder=study.local_jobs, selection_builder=study.freeze_selection,
              report_builder=study.summarize, completion_jobs=lambda p,s:s['confirmation']['jobs'])
    result = engine.run(p, tmp_path/'pass', launcher=launch, **kw)
    assert result['status'] == 'completed' and len(calls) == len(fixtures)
    calls.clear()
    def fail(*a, **kw):calls.append(1); return 'failed', 'synthetic failure'
    with pytest.raises(RuntimeError):engine.run(p, tmp_path/'failed', launcher=fail, **kw)
    assert len(calls) == 1
    with pytest.raises(ValueError):engine.run(p, tmp_path/'failed', launcher=fail, **kw)
    assert len(calls) == 1
