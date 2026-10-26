"""Synthetic predecessor adoption checks; no model execution or measured scores."""
import json
import pytest

from src.research import kda_quality_study as kda
from src.research.warmup_quality_predecessor import reviewed_kda
from src.research.protocol import digest, file_digest
from tests.test_kda_quality_study import plan, complete
from tests.test_schedule_study import coverage


def evidence(tmp_path, winner='kda', qualifies=True):
    p=plan(); p.update(git_commit='1'*40, source_sha256='2'*64)
    state=complete(p, mha_lr=.0025, best=winner)
    if not qualifies:
        for r in state['records']:
            if r['stage']=='confirmation' and r['family']==winner and r['width']==256:
                r['averaged_outputs']['200']['coverage']['evaluations']['128']=coverage(2.5)
    report=kda.summarize(p,state)
    for name,value in [('plan',p),('report',report),('selection',state['selection'])]:
        (tmp_path/f'{name}.json').write_text(json.dumps(value))
    review=dict(status='verified',distinct_trajectories=len(state['records']),
        distinct_checkpoints=2*len(state['records']),plan_sha256=digest(p),
        source_commit=p['git_commit'],source_sha256=p['source_sha256'],
        report_sha256=file_digest(tmp_path/'report.json'),
        selection_sha256=file_digest(tmp_path/'selection.json'),quality_decision=report['quality_decision'])
    path=tmp_path/'review.json'; path.write_text(json.dumps(review))
    return path,p,state,report


@pytest.mark.parametrize('winner',['mha','kda'])
@pytest.mark.parametrize('qualifies',[False,True])
def test_only_final_frozen_decision_resolves_incumbent(tmp_path,winner,qualifies):
    path,p,state,report=evidence(tmp_path,winner,qualifies)
    assert report['quality_decision']['qualifies_for_incumbent_replacement']==qualifies
    result=reviewed_kda(path)
    assert result['attention_family']==(winner if qualifies else 'mha')
    assert result['learning_rate']==(.0025 if qualifies else .005)
    assert result['weight_decay']==.00006
    cfg=result['config']
    assert cfg['model']['attn_type']==result['attention_family']
    assert cfg['training']['weight_averaging']['half_lives']==[200]
    assert cfg['training']['lr_schedule']['kind']=='linear_cooldown'


@pytest.mark.parametrize('key,value',[('status','partial'),('distinct_trajectories',12),
    ('distinct_checkpoints',59),('source_commit','changed'),('source_sha256','changed'),
    ('plan_sha256','changed'),('report_sha256','changed'),('selection_sha256','changed'),
    ('quality_decision',{'frozen_screen_winner':'mha','qualifies_for_incumbent_replacement':True})])
def test_partial_or_altered_review_rejected(tmp_path,key,value):
    path,_,_,_=evidence(tmp_path)
    review=json.loads(path.read_text()); review[key]=value
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError): reviewed_kda(path)


def test_report_rehash_does_not_hide_modified_adoption(tmp_path):
    path,_,_,report=evidence(tmp_path)
    report['quality_decision']['qualifies_for_incumbent_replacement']=False
    (tmp_path/'report.json').write_text(json.dumps(report))
    review=json.loads(path.read_text()); review['report_sha256']=file_digest(tmp_path/'report.json')
    review['quality_decision']=report['quality_decision']; path.write_text(json.dumps(review))
    with pytest.raises(ValueError): reviewed_kda(path)
