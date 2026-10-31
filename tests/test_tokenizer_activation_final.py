"""Final decision tests with synthetic endpoints; no new scientific result."""
import copy
import math

import pytest

from src.research import tokenizer_activation_final as f
from src.research import tokenizer_activation_inference_entry as inference
from src.research.tokenizer_activation_session import reference
from src.research.tokenizer_activation_rules import confirmation_specs
from tests.test_tokenizer_activation_session import ledger,publish
from tests.test_tokenizer_activation_inference import ready
from tests.test_tokenizer_activation_timing import SELECTION


def score(ce):
    counts=[min(2048,34278-i) for i in range(0,34278,2048)]
    return dict(context=128,eval_batch_size=8,block_tokens=2048,target_count=34278,target_bytes=111536,
        window_count=math.ceil(34278/128),parameters=f.PARAMETERS,block_target_counts=counts,
        block_loss_sums=[n*ce for n in counts],mean_ce=ce,bits_per_byte=ce*34278/111536/math.log(2),perplexity=math.exp(ce))


def rows(deltas=(-.03,-.03,-.03)):
    return [dict(spec=s,raw=score(5.05),h200=score(5.+(deltas[f.FRESH_SEEDS.index(s['seed'])] if s['family']=='gelu_exact' else 0)),
        training_seconds=100.,worker_seconds=150.,peak_rss_bytes=1000000000,peak_mps_driver_bytes=2000000000)
        for s in confirmation_specs(SELECTION)]


@pytest.mark.parametrize('deltas,passed',[((-.03,-.03,-.03),True),((-.01,-.01,-.01),False),((-.1,-.1,.01),False),((0.,0.,0.),False)])
def test_frozen_strict_rule_has_no_neutral_cost_route(deltas,passed):
    result=f.analyze(rows(deltas),SELECTION)
    assert result['strict_quality'] is passed and result['adopted'] is False
    assert result['statistics']['seed_deltas']==pytest.approx(deltas)
    assert len(result['fresh_endpoints'])==6


@pytest.mark.parametrize('case',['missing','order','coverage','score','duration','rss','driver','extra'])
def test_incomplete_invalid_or_changed_fresh_panel_rejected(case):
    r=rows()
    if case=='missing':r.pop()
    elif case=='order':r[0],r[1]=r[1],r[0]
    elif case=='coverage':r[0]['h200']['block_target_counts'][-1]-=1
    elif case=='score':r[0]['h200']['mean_ce']-=.1
    elif case=='duration':r[0]['training_seconds']=151
    elif case=='rss':r[0]['peak_rss_bytes']=4*1024**3
    elif case=='driver':r[0]['peak_mps_driver_bytes']=6*1024**3
    elif case=='extra':r[0]['claim']='cost adoption'
    with pytest.raises(ValueError):f.analyze(r,SELECTION)


@pytest.fixture
def terminal(ready,monkeypatch):
    led,launches,binding=ready
    for _ in range(4):inference.run_once(led)
    monkeypatch.setattr(f,'bind',lambda *a,**k:None);monkeypatch.setattr(f,'idle',lambda *a:None)
    monkeypatch.setattr(f,'quality_binding',binding)
    monkeypatch.setattr(led,'inspect',lambda:dict(next={'kind':'review_final'},selection=SELECTION))
    for row in rows():
        b=binding(led,row['spec']['seed'])['bindings'][row['spec']['family']]
        stem=str(__import__('pathlib').Path(b['request']['path']).parent)
        measured={role:dict(row[role],checkpoint=str(led.root/b['weights'][role]['path']),checkpoint_sha256=b['weights'][role]['sha256']) for role in ('raw','h200')}
        publish(led,stem+'/measurements.json',measured)
        publish(led,stem+'/training-summary.json',dict(training_time_sec=row['training_seconds']))
        publish(led,stem+'/resource-audit.json',dict(samples=[dict(elapsed_seconds=row['worker_seconds'],peak_rss_bytes=row['peak_rss_bytes'],mps_driver_bytes=row['peak_mps_driver_bytes'])]))
    return led


def test_final_publication_pins_all_six_pairs_and_fixed_representative(terminal):
    ref=f.complete(terminal);done=terminal.document(ref);decision=terminal.document(done['decision'])
    assert done['status']=='closed' and done['adopted'] is True
    assert len(decision['fresh_provenance'])==6 and decision['retained_fresh_pairs']==6
    assert decision['representative_recipe']['seed']==3611
    assert decision['representative_recipe']['family']=='gelu_exact'
    for row in decision['fresh_provenance']:
        for weight in row['weights'].values():terminal.verify(weight)
    with pytest.raises(ValueError,match='cannot retry'):f.complete(terminal)


def test_final_failure_is_permanent_and_never_publishes_decision(terminal,monkeypatch):
    def fail(*args):raise ValueError('injected final audit failure')
    monkeypatch.setattr(f,'build_report',fail)
    with pytest.raises(ValueError,match='injected'):f.complete(terminal)
    assert not (terminal.root/'evidence/meta/activation-final-decision.json').exists()
    assert (terminal.root/'evidence/meta/activation-final-failure.json').exists()
    with pytest.raises(ValueError,match='cannot retry'):f.complete(terminal)


def test_final_rechecks_measurement_bytes_after_arithmetic(terminal,monkeypatch):
    original=f.analyze
    def changed(*args):
        value=original(*args)
        p=terminal.root/'evidence/quality/06/measurements.json';p.write_bytes(p.read_bytes()+b' ')
        return value
    monkeypatch.setattr(f,'analyze',changed)
    with pytest.raises(ValueError,match='metadata changed'):f.build_report(terminal)
