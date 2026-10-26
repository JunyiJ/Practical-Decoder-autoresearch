"""Synthetic complete predecessor adoption only; no model or corpus execution."""
import json

import pytest
from src.research import value_residual_study as normalization
from src.research.position_quality_predecessor import reviewed_value_residual
from src.research.protocol import digest, file_digest
from src.research.runner import atomic_json
from tests.test_value_residual_study import plan, complete
from tests.test_schedule_study import coverage


def predecessor(folder,family='local_half',none_lr=.005,adopt=True):
    p=plan();p['git_commit']='a'*40;s=complete(p,none_lr,family)
    if not adopt:
        for r in s['records'][18:]:
            if r['family']==family and r['width']==256 and r['seed']==2473:
                r['averaged_outputs']['200']['coverage']['evaluations']['128']=coverage(2.1)
    report=normalization.summarize(p,s);s['status']='completed'
    for name,value in [('plan',p),('selection',s['selection']),('report',report),('state',s)]:
        atomic_json(folder/f'{name}.json',value)
    review=dict(status='verified',distinct_trajectories=len(s['records']),distinct_checkpoints=2*len(s['records']),
        plan_sha256=digest(p),source_commit=p['git_commit'],source_sha256=p['source_sha256'],
        report_sha256=file_digest(folder/'report.json'),selection_sha256=file_digest(folder/'selection.json'),
        quality_decision=report['quality_decision'])
    path=folder/'review.json';atomic_json(path,review);return path


@pytest.mark.parametrize('family',['local','local_half','first_blend'])
@pytest.mark.parametrize('adopt',[False,True])
@pytest.mark.parametrize('none_lr',[.0025,.005,.01])
def test_only_final_confirmed_scale_lr_propagates(tmp_path,family,adopt,none_lr):
    path=predecessor(tmp_path,family,none_lr,adopt)
    report=json.loads((tmp_path/'report.json').read_text())
    yes=report['quality_decision']['qualifies_for_incumbent_replacement']
    r=reviewed_value_residual(path)
    expected_value=normalization.VALUES[family] if yes else 'local'
    expected_gain='unit'
    expected_activation='silu'
    expected_attention=1.
    expected_scale=1.
    expected_lr=report['selection']['choices'][family]['lr'] if yes else .005
    assert r['input_embedding_scale']==expected_scale and r['learning_rate']==expected_lr
    assert r['warmup_steps']==0 and r['weight_decay']==.00006
    assert r['config']['model']['input_embedding_scale']==expected_scale
    assert r['residual_topology_kind']==r['config']['model']['residual_topology_kind']=='serial'
    assert r['value_residual_kind']==r['config']['model']['value_residual_kind']==expected_value
    assert r['residual_gain_kind']==r['config']['model']['residual_gain_kind']==expected_gain
    assert r['gated_activation_kind']==r['config']['model']['gated_activation_kind']==expected_activation
    assert r['attention_logit_scale']==r['config']['model']['attention_logit_scale']==expected_attention
    assert r['weight_tying_kind']==r['config']['model']['weight_tying_kind']=='legacy_untied'
    assert r['output_bias_kind']==r['config']['model']['output_bias_kind']=='none'
    assert r['readout_norm_kind']==r['config']['model']['readout_norm_kind']=='layer_affine'
    assert r['output_bias_prior']==r['config']['model']['output_bias_prior']
    assert r['config']['model']['vocab_size']==len(r['output_bias_prior']['alphabet'])
    assert r['readout_norm_epsilon']==1e-5
    assert r['config']['training']['learning_rate']==expected_lr


@pytest.mark.parametrize('field,value',[
    ('status','partial_verified'),('distinct_trajectories',6),('distinct_checkpoints',12),
    ('plan_sha256','0'*64),('source_commit','b'*40),('source_sha256','0'*64),
    ('report_sha256','0'*64),('selection_sha256','0'*64)])
def test_partial_or_changed_review_rejected(tmp_path,field,value):
    path=predecessor(tmp_path);v=json.loads(path.read_text());v[field]=value;atomic_json(path,v)
    with pytest.raises(ValueError):reviewed_value_residual(path)


def test_running_state_cannot_borrow_a_review(tmp_path):
    path=predecessor(tmp_path);p=tmp_path/'state.json';s=json.loads(p.read_text());s['status']='running';atomic_json(p,s)
    with pytest.raises(ValueError,match='Completed'):reviewed_value_residual(path)


def test_rehashed_false_adoption_rejected(tmp_path):
    path=predecessor(tmp_path,adopt=False);rp=tmp_path/'report.json';v=json.loads(rp.read_text())
    v['quality_decision']['qualifies_for_incumbent_replacement']=True;atomic_json(rp,v)
    r=json.loads(path.read_text());r.update(report_sha256=file_digest(rp),quality_decision=v['quality_decision']);atomic_json(path,r)
    with pytest.raises(ValueError,match='complete evidence'):reviewed_value_residual(path)


def test_no_local_eligibility_means_no_confirmation_reselection(tmp_path):
    path=predecessor(tmp_path,'local',.005,True)
    r=reviewed_value_residual(path);assert r['input_embedding_scale']==1. and r['learning_rate']==.005
