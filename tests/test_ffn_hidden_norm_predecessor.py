"""Synthetic final-adoption propagation, with no models or training."""
import json
import pytest
from src.research import output_gate_study as gate
from src.research.ffn_hidden_norm_predecessor import reviewed_output_gate
from src.research.protocol import digest, file_digest
from src.research.runner import atomic_json
from tests.test_output_gate_study import plan, complete
from tests.test_schedule_study import coverage


def predecessor(folder, family='sigmoid_headwise_zero', adopt=True, alias=False):
    p = plan();p['git_commit'] = 'a'*40
    s = complete(p, .005 if alias else .0025, family)
    if not adopt:
        for r in s['records'][18:]:
            if r['family'] == family and r['width'] == 256 and r['seed'] == 2213:
                r['averaged_outputs']['200']['coverage']['evaluations']['128'] = coverage(2.1)
    report = gate.summarize(p, s)
    for name,value in [('plan',p),('selection',s['selection']),('report',report)]:
        atomic_json(folder/f'{name}.json', value)
    review = dict(status='verified',distinct_trajectories=len(s['records']),
        distinct_checkpoints=2*len(s['records']),plan_sha256=digest(p),
        source_commit=p['git_commit'],source_sha256=p['source_sha256'],
        report_sha256=file_digest(folder/'report.json'),
        selection_sha256=file_digest(folder/'selection.json'),
        quality_decision=report['quality_decision'])
    path=folder/'review.json';atomic_json(path,review)
    return path


@pytest.mark.parametrize('family',gate.FAMILIES)
@pytest.mark.parametrize('adopt',[False,True])
def test_only_final_adopted_gate_and_lr_propagates(tmp_path, family, adopt):
    r = reviewed_output_gate(predecessor(tmp_path,family,adopt))
    wanted = family if adopt else 'none'
    lr = {'none':.0025,'static_half':.0025,'sigmoid_headwise_zero':.00125}[family] if adopt else .005
    assert r['attention_output_gate']==wanted and r['learning_rate']==lr
    cfg=r['config'];m=cfg['model'];t=cfg['training']
    assert m.get('attention_output_gate','none')==wanted
    assert ('output_gate_width_policy' in m)==(wanted!='none')
    assert t['learning_rate']==lr and t['max_iters']==3000
    assert t['lr_schedule']['start_step']==2000
    assert t['weight_averaging']['half_lives']==[200]


@pytest.mark.parametrize('family',['static_half','sigmoid_headwise_zero'])
def test_prospective_aliases_remain_complete_evidence(tmp_path,family):
    path=predecessor(tmp_path,family,True,alias=True)
    assert json.loads(path.read_text())['distinct_trajectories']==36
    assert reviewed_output_gate(path)['attention_output_gate']==family


@pytest.mark.parametrize('field,value',[
    ('status','partial_verified'),('distinct_trajectories',18),('distinct_checkpoints',36),
    ('plan_sha256','0'*64),('source_commit','b'*40),('source_sha256','0'*64),
    ('report_sha256','0'*64),('selection_sha256','0'*64),
])
def test_partial_or_changed_review_rejected(tmp_path,field,value):
    path=predecessor(tmp_path);v=json.loads(path.read_text());v[field]=value;atomic_json(path,v)
    with pytest.raises(ValueError):reviewed_output_gate(path)


def test_rehashed_false_adoption_rejected(tmp_path):
    path=predecessor(tmp_path,adopt=False);rp=tmp_path/'report.json';v=json.loads(rp.read_text())
    v['quality_decision']['qualifies_for_incumbent_replacement']=True;atomic_json(rp,v)
    r=json.loads(path.read_text());r.update(report_sha256=file_digest(rp),quality_decision=v['quality_decision']);atomic_json(path,r)
    with pytest.raises(ValueError,match='complete evidence'):reviewed_output_gate(path)


def test_rehashed_missing_confirmation_rejected(tmp_path):
    path=predecessor(tmp_path);rp=tmp_path/'report.json';v=json.loads(rp.read_text());v['records'].pop();atomic_json(rp,v)
    r=json.loads(path.read_text());r['report_sha256']=file_digest(rp);atomic_json(path,r)
    with pytest.raises(ValueError):reviewed_output_gate(path)


def test_local_failure_cannot_be_rescued_by_better_confirmation(tmp_path):
    path=predecessor(tmp_path,family='none',adopt=True,alias=True)
    r=reviewed_output_gate(path)
    assert r['attention_output_gate']=='none' and r['learning_rate']==.005
