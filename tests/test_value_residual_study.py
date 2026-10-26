"""Synthetic selection/lineage budgets; no training or corpus evaluations."""
import copy
import json
import pytest

from src.research import value_residual_study as study, optimizer_study as engine
from src.research.runner import atomic_json
from tests.test_readout_norm_study import plan as normalization_plan
from src.models.output_bias import build_training_prior
from src.models.value_residual import POLICY
from tests.test_qknorm_study import row
from tests.test_schedule_study import coverage


def plan(input_scale=1., incumbent_lr=.005):
    p = normalization_plan()
    for key in ('readout_norm_kinds','readout_norm_policy','readout_norm_epsilon'):
        p.pop(key)
    p.update(study='value-residual-quality-v1',families=study.FAMILIES,
        learning_rates=study.LRS,seeds=study.SCREEN_SEEDS,confirmation_seeds=study.CONFIRM_SEEDS,
        widths=study.WIDTHS,max_trials=42,max_model_evaluations=84,
        value_residuals=study.VALUES,value_residual_policy=POLICY,prior_pseudocount=1,
        recipe=dict(warmup_steps=0,learning_rate=incumbent_lr,weight_decay=.00006,input_embedding_scale=input_scale,readout_norm_kind='layer_affine',readout_norm_epsilon=1e-5,output_bias_kind='none',weight_tying_kind='legacy_untied',residual_topology_kind='serial',attention_logit_scale=1.,gated_activation_kind='silu',residual_gain_kind='unit'))
    p['output_bias_prior']=build_training_prior(b'aaabbbcccddd'*200,.8,.1)
    p['data_sha256']=p['output_bias_prior']['corpus_sha256']
    p['config']['model'],p['config']['training']=study.backbone(p['recipe'],p['output_bias_prior'])
    p['screen_jobs']=[list(x) for x in study.local_jobs(p)]
    return study.validate_plan(p)


def screening(p, none_lr=None, best='first_blend'):
    none_lr = p['recipe']['learning_rate'] if none_lr is None else none_lr
    choices = dict(local=none_lr, local_half=.01, first_blend=.0025)
    means = dict(local=2., local_half=1.99, first_blend=1.97)
    if best == 'local_half':
        means.update(local_half=1.97, first_blend=1.99)
    elif best == 'local':
        means.update(local=1.97, local_half=2.03, first_blend=2.04)
    out = []
    for j in study.local_jobs(p):
        f, lr, _, seed = j
        value = means[f] + (0 if lr == choices[f] else .08)
        out.append(row(j, ce=value))
    return out


def complete(p, none_lr=None, best='first_blend'):
    local = screening(p, none_lr, best)
    selection = study.freeze_selection(p, local)
    vals = dict(local=2., local_half=1.98, first_blend=1.96, incumbent_reference=2.02)
    if best == 'local_half':
        vals.update(local_half=1.96, first_blend=1.98)
    if best == 'local':
        vals.update(local=1.96, local_half=1.98, first_blend=1.95)
    conf = [row(j, 'confirmation', ce=vals[j[0]], raw_ce=4. if j[0] == best else 3.) for j in selection['confirmation']['jobs']]
    return dict(records=local + conf, selection=selection)


@pytest.mark.parametrize('input_scale',[1.,.5,.25])
@pytest.mark.parametrize('incumbent_lr',[.0025,.005,.01])
@pytest.mark.parametrize('arm',study.FAMILIES+[study.REFERENCE])
def test_only_gate_kind_and_peak_lr_change(input_scale,incumbent_lr,arm):
    p=plan(input_scale,incumbent_lr);c=study.suite_spec(p,arm,incumbent_lr)['candidate']['config']
    m=copy.deepcopy(c['model'])
    assert m.pop('value_residual_kind')==study.VALUES['local' if arm==study.REFERENCE else arm]
    expected=copy.deepcopy(p['config']['model']);expected.pop('value_residual_kind');expected['vocab_size']=1  # executor placeholder; trainer resolves corpus vocabulary
    assert m==expected and c['training']==p['config']['training']
    assert len(study.local_jobs(p))==18
    assert c['training']['max_iters']*8*128==3072000
    assert c['training']['weight_averaging']['half_lives']==[200]


@pytest.mark.parametrize('none_lr,count',[(.005,36),(.0025,42),(.01,42)])
def test_exact_reference_alias_and_complete_correlated_output_count(none_lr,count):
    incumbent_lr=.005;p=plan();s=complete(p,none_lr)
    r = study.summarize(p, s)
    assert len(s['records']) == r['physical_training_trajectories'] == count
    assert r['model_evaluations'] == 2 * count
    assert r['reference_aliases'] == ({study.REFERENCE:'local'} if incumbent_lr == none_lr else {})
    assert r['quality_decision']['qualifies_for_incumbent_replacement']
    assert len(r['comparisons']) == 12 and len(r['fixed_lr_screen']) == 9
    for c in r['comparisons']:
        if c['candidate'] == 'first_blend':
            assert c['quality_improvement'] and c['raw_sensitivity'][0]['mean_delta'] > 0
        if c['candidate'] == 'local' and c['reference'] == study.REFERENCE and incumbent_lr == none_lr:
            assert all(x['conditional_text_bootstrap_95pct'] == [0., 0.] for x in c['sensitivity'])


@pytest.mark.parametrize('best', study.FAMILIES)
def test_each_frozen_family_can_qualify_without_alternative_promotion(best):
    p = plan(); s = complete(p, .0025 if best == 'local' else .005, best)
    assert s['selection']['winner'] == best
    report = study.summarize(p, s)
    assert report['quality_decision'] == dict(frozen_screen_winner=best, qualifies_for_incumbent_replacement=True)
    for r in s['records']:
        if r['stage'] == 'confirmation' and r['family'] == best and r['width'] == 256 and r['seed'] == 2473:
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(2.1)
    report = study.summarize(p, s)
    assert not report['quality_decision']['qualifies_for_incumbent_replacement']


def test_lower_lr_and_simpler_arm_win_exact_ties():
    p = plan(); rows = [row(j, ce=2.) for j in study.local_jobs(p)]
    sel = study.freeze_selection(p, rows)
    assert all(c['lr'] == .0025 for c in sel['choices'].values())
    assert sel['best_selected_arm'] == 'local' and sel['winner'] == 'incumbent_unchanged'


def test_bad_screen_seed_cannot_be_rescued_by_other_arm_or_confirmation():
    p = plan(); s = complete(p)
    for r in s['records'][:18]:
        if r['family'] == 'first_blend' and r['lr'] == .0025:
            value = 1.92 if r['seed'] == 2461 else 2.001
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(value)
    s['selection'] = study.freeze_selection(p, s['records'][:18])
    assert s['selection']['best_selected_arm'] == 'first_blend'
    assert s['selection']['winner'] == 'incumbent_unchanged'
    assert not study.summarize(p, s)['quality_decision']['qualifies_for_incumbent_replacement']


@pytest.mark.parametrize('reference', ['local', study.REFERENCE])
def test_nondefault_must_beat_both_references_on_both_widths(reference):
    p = plan(); s = complete(p, .0025)
    for r in s['records']:
        if r['stage'] == 'confirmation' and r['family'] == reference and r['width'] == 256 and r['seed'] == 2473:
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(1.95)
    assert not study.summarize(p, s)['quality_decision']['qualifies_for_incumbent_replacement']


def test_other_candidate_comparison_is_reported_but_not_a_hidden_adoption_gate():
    p = plan(); s = complete(p)
    for r in s['records']:
        if r['stage'] == 'confirmation' and r['family'] == 'local_half':
            r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(1.95)
    report = study.summarize(p, s)
    assert report['quality_decision']['qualifies_for_incumbent_replacement']
    assert not any(c['quality_improvement'] for c in report['comparisons'] if c['candidate']=='first_blend' and c['reference']=='local_half')


def test_text_interval_can_veto_negative_means_and_every_seed():
    p = plan(); s = complete(p)
    for r in s['records']:
        if r['stage'] != 'confirmation' or r['family'] != 'first_blend':continue
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
    elif change == 'seed':rs[0]['seed'] = 2471
    elif change == 'stage':rs[0]['stage'] = 'confirmation'
    elif change == 'failed':rs[0]['status'] = 'failed'
    elif change == 'extra_average':rs[0]['averaged_outputs']['50'] = rs[0]['averaged_outputs']['200']
    else:rs[0]['coverage']['128']['target_count'] = 1
    with pytest.raises(ValueError):study.freeze_selection(p, rs)


@pytest.mark.parametrize('change', ['manifest','choices','eligibility','winner','alias','extra','missing'])
def test_confirmation_cannot_modify_choices_or_aliases(change):
    p = plan(); s = complete(p)
    if change == 'manifest':s['selection']['confirmation']['jobs'].pop()
    elif change == 'choices':s['selection']['choices']['local_half']['lr'] = .005
    elif change == 'eligibility':s['selection']['candidate_eligible'] = False
    elif change == 'winner':s['selection']['winner'] = 'local_half'
    elif change == 'alias':s['selection']['confirmation']['reference_aliases'] = {}
    elif change == 'extra':s['records'].append(s['records'][-1])
    else:s['records'].pop()
    with pytest.raises(ValueError):study.summarize(p, s)


@pytest.mark.parametrize('field,value', [('max_trials',43),('returned_half_life',50),('widths',[64,128,256]),
    ('bootstrap_draws',100),('min_improvement',0.),('max_model_evaluations',85),('trial_seconds',9999),
    ('value_residual_policy','arbitrary'),('prior_pseudocount',2),('seeds',[2461,2471])])
def test_protocol_extensions_rejected(field, value):
    p = plan(); p[field] = value
    with pytest.raises(ValueError):study.validate_plan(p)


@pytest.mark.parametrize('change', ['schedule','averaging','model','recipe','tokens'])
def test_unreviewed_backbone_rejected(change):
    p = plan()
    if change == 'schedule':p['config']['training']['lr_schedule']['start_step'] = 1990
    elif change == 'averaging':p['config']['training']['weight_averaging']['half_lives'] = [50]
    elif change == 'model':p['config']['model']['ffn_hidden_norm'] = 'first_blend'
    elif change == 'recipe':p['recipe']['input_embedding_scale'] = .75
    else:p['config']['training']['max_iters'] = 6000
    with pytest.raises(ValueError):study.validate_plan(p)


@pytest.mark.parametrize('none_lr', [.005,.0025])
def test_real_serial_engine_freezes_before_confirmation_and_stops_first_failure(tmp_path, monkeypatch, none_lr):
    p = plan(); s = complete(p, none_lr); fixtures = s['records']; calls = []
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


@pytest.mark.parametrize('norm',['layer_affine','layer_no_bias','rms'])
def test_final_normalization_is_shared_by_all_bias_arms(norm):
    p=plan();p['recipe']['readout_norm_kind']=norm
    p['config']['model'],p['config']['training']=study.backbone(p['recipe'],p['output_bias_prior'])
    for f in study.FAMILIES+[study.REFERENCE]:
        cfg=study.suite_spec(p,f,.005)['candidate']['config']
        assert cfg['model']['readout_norm_kind']==norm
        assert cfg['model']['output_bias_prior']==p['output_bias_prior']

@pytest.mark.parametrize('damage',['counts','initial_values','corpus','split','prior_digest','model_prior'])
def test_prior_binding_cannot_be_changed(damage):
    p=plan()
    if damage=='counts':p['output_bias_prior']['counts'][0]+=1
    elif damage=='initial_values':p['output_bias_prior']['initial_values'][0]+=.1
    elif damage=='corpus':p['data_sha256']='0'*64
    elif damage=='split':p['config']['data']['train_fraction']=.7
    elif damage=='prior_digest':p['output_bias_prior']['prior_sha256']='0'*64
    else:p['config']['model']['output_bias_prior']['counts'][0]+=1
    with pytest.raises(ValueError):study.validate_plan(p)

@pytest.mark.parametrize('bias',['none','learned_zero','learned_train_unigram'])
@pytest.mark.parametrize('norm',['layer_affine','layer_no_bias','rms'])
def test_inherited_bias_and_norm_are_shared_without_retuning(bias,norm):
    p=plan();p['recipe'].update(output_bias_kind=bias,readout_norm_kind=norm)
    p['config']['model'],p['config']['training']=study.backbone(p['recipe'],p['output_bias_prior'])
    for f in study.FAMILIES+[study.REFERENCE]:
        cfg=study.suite_spec(p,f,.005)['candidate']['config']
        assert cfg['model']['output_bias_kind']==bias
        assert cfg['model']['readout_norm_kind']==norm
        assert cfg['model']['output_bias_prior']==p['output_bias_prior']

@pytest.mark.parametrize('topology',['serial','parallel_separate_norms'])
@pytest.mark.parametrize('table',['legacy_untied','matched_untied','scaled_tied'])
def test_final_topology_and_table_remain_shared(topology,table):
    p=plan();p['recipe'].update(residual_topology_kind=topology,weight_tying_kind=table)
    p['config']['model'],p['config']['training']=study.backbone(p['recipe'],p['output_bias_prior'])
    for f in study.FAMILIES+[study.REFERENCE]:
        m=study.suite_spec(p,f,.005)['candidate']['config']['model']
        assert m['residual_topology_kind']==topology and m['weight_tying_kind']==table

@pytest.mark.parametrize('attention',[1.,.5,2.])
@pytest.mark.parametrize('topology',['serial','parallel_separate_norms'])
def test_final_attention_remains_fixed_across_all_gate_arms(attention,topology):
    p=plan();p['recipe'].update(attention_logit_scale=attention,residual_topology_kind=topology)
    p['config']['model'],p['config']['training']=study.backbone(p['recipe'],p['output_bias_prior'])
    for f in study.FAMILIES+[study.REFERENCE]:
        m=study.suite_spec(p,f,.005)['candidate']['config']['model']
        assert m['attention_logit_scale']==attention and m['residual_topology_kind']==topology

@pytest.mark.parametrize('attention',[True,0.,.25,3.,'half'])
def test_unreviewed_attention_inheritance_rejected(attention):
    p=plan();p['recipe']['attention_logit_scale']=attention
    with pytest.raises(ValueError):study.validate_plan(p)

@pytest.mark.parametrize('activation',['silu','gelu_exact','relu'])
@pytest.mark.parametrize('attention',[1.,.5,2.])
def test_final_activation_and_attention_shared_by_all_residual_arms(activation,attention):
    p=plan();p['recipe'].update(gated_activation_kind=activation,attention_logit_scale=attention)
    p['config']['model'],p['config']['training']=study.backbone(p['recipe'],p['output_bias_prior'])
    for f in study.FAMILIES+[study.REFERENCE]:
        m=study.suite_spec(p,f,.005)['candidate']['config']['model']
        assert m['gated_activation_kind']==activation and m['attention_logit_scale']==attention

@pytest.mark.parametrize('activation',[None,True,'gelu','swiglu','unknown'])
def test_unreviewed_activation_inheritance_rejected(activation):
    p=plan();p['recipe']['gated_activation_kind']=activation
    with pytest.raises(ValueError):study.validate_plan(p)

@pytest.mark.parametrize('gain',['unit','static_tenth','learned_tenth'])
@pytest.mark.parametrize('activation',['silu','gelu_exact','relu'])
def test_final_gain_and_activation_shared_by_all_value_arms(gain,activation):
    p=plan();p['recipe'].update(residual_gain_kind=gain,gated_activation_kind=activation)
    p['config']['model'],p['config']['training']=study.backbone(p['recipe'],p['output_bias_prior'])
    study.validate_plan(p)
    for f in study.FAMILIES+[study.REFERENCE]:
        m=study.suite_spec(p,f,p['recipe']['learning_rate'])['candidate']['config']['model']
        assert m['residual_gain_kind']==gain and m['gated_activation_kind']==activation

@pytest.mark.parametrize('gain',[None,True,'local','unknown'])
def test_invalid_inherited_gain_rejected(gain):
    p=plan();p['recipe']['residual_gain_kind']=gain
    with pytest.raises(ValueError):study.validate_plan(p)
