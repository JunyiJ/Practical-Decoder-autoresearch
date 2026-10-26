"""Synthetic complete predecessor adoption only; no model or corpus execution."""
import json

import pytest
from src.research import warmup_quality_study as warmup
from src.research.embedding_scale_predecessor import reviewed_warmup
from src.research.protocol import digest, file_digest
from src.research.runner import atomic_json
from tests.test_warmup_quality_study import plan, complete
from tests.test_schedule_study import coverage


def predecessor(folder,family='warmup100',adopt=True):
    p=plan();p['git_commit']='a'*40;s=complete(p,family)
    if not adopt:
        for r in s['records'][6:]:
            if r['family']==family and r['width']==256 and r['seed']==2293:
                r['averaged_outputs']['200']['coverage']['evaluations']['128']=coverage(2.1)
    report=warmup.summarize(p,s);s['status']='completed'
    for name,value in [('plan',p),('selection',s['selection']),('report',report),('state',s)]:
        atomic_json(folder/f'{name}.json',value)
    review=dict(status='verified',distinct_trajectories=24,distinct_checkpoints=48,
        plan_sha256=digest(p),source_commit=p['git_commit'],source_sha256=p['source_sha256'],
        report_sha256=file_digest(folder/'report.json'),selection_sha256=file_digest(folder/'selection.json'),
        quality_decision=report['quality_decision'])
    path=folder/'review.json';atomic_json(path,review);return path


@pytest.mark.parametrize('family',['warmup100','warmup300'])
@pytest.mark.parametrize('adopt',[False,True])
def test_only_final_confirmed_warmup_propagates(tmp_path,family,adopt):
    r=reviewed_warmup(predecessor(tmp_path,family,adopt));expected=int(family[6:]) if adopt else 0
    assert r['warmup_steps']==expected and r['learning_rate']==.005 and r['weight_decay']==.00006
    schedule=r['config']['training']['lr_schedule']
    assert schedule['kind']==('warmup_linear_cooldown' if adopt else 'linear_cooldown')
    assert schedule.get('warmup_steps',0)==expected
    assert r['config']['model']['attn_type']=='mha'


@pytest.mark.parametrize('field,value',[
    ('status','partial_verified'),('distinct_trajectories',6),('distinct_checkpoints',12),
    ('plan_sha256','0'*64),('source_commit','b'*40),('source_sha256','0'*64),
    ('report_sha256','0'*64),('selection_sha256','0'*64)])
def test_partial_or_changed_review_rejected(tmp_path,field,value):
    path=predecessor(tmp_path);v=json.loads(path.read_text());v[field]=value;atomic_json(path,v)
    with pytest.raises(ValueError):reviewed_warmup(path)


def test_running_state_cannot_borrow_a_review(tmp_path):
    path=predecessor(tmp_path);p=tmp_path/'state.json';s=json.loads(p.read_text());s['status']='running';atomic_json(p,s)
    with pytest.raises(ValueError,match='Completed'):reviewed_warmup(path)


def test_rehashed_false_adoption_rejected(tmp_path):
    path=predecessor(tmp_path,adopt=False);rp=tmp_path/'report.json';v=json.loads(rp.read_text())
    v['quality_decision']['qualifies_for_incumbent_replacement']=True;atomic_json(rp,v)
    r=json.loads(path.read_text());r.update(report_sha256=file_digest(rp),quality_decision=v['quality_decision']);atomic_json(path,r)
    with pytest.raises(ValueError,match='complete evidence'):reviewed_warmup(path)


def test_no_local_eligibility_means_no_confirmation_reselection(tmp_path):
    path=predecessor(tmp_path,'baseline',True)
    r=reviewed_warmup(path);assert r['warmup_steps']==0
