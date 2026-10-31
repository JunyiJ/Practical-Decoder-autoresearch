"""Independent first-pair replay on disposable v1 producer fixtures only."""
import json

import pytest
from research.activation_inference_recovery_v1.test_execution import ready
from research.activation_inference_recovery_v1 import controller as entry
from research.activation_inference_recovery_v2 import carried_pair as carried


def test_first_pair_replayed_without_original_failed_root_entry(ready,monkeypatch):
    r=ready;accepted=entry.run_once(r.ctx)
    monkeypatch.setattr(carried,'_group_exists',lambda pid:False)
    result=carried._verify_first(r.ctx,r.source,r.imported,accepted)
    assert result['acceptance']==accepted and result['result']['seed']==3611
    assert result['binding']==r.imported['pair_bindings']['3611']
    assert r.launches==[3611]  # one disposable producer; import launches nothing


@pytest.mark.parametrize('case',['acceptance','request','model','result','guard','worker_identity','extra_file','review'])
def test_changed_carried_first_evidence_rejected(ready,monkeypatch,case):
    r=ready;accepted=entry.run_once(r.ctx);monkeypatch.setattr(carried,'_group_exists',lambda pid:False)
    names={'acceptance':'accepted.json','request':'request.json','model':'models.json','result':'result.json',
           'worker_identity':'attempt.json','review':'review.json'}
    if case=='guard':p=r.ctx.root/'control/recovery-worker-inference-3511.json'
    elif case=='extra_file':p=r.ctx.root/'evidence/inference/3511/unknown.json'
    else:p=r.ctx.root/'evidence/inference/3511'/names[case]
    p.write_text('{}')
    with pytest.raises((ValueError,KeyError)):carried._verify_first(r.ctx,r.source,r.imported,accepted)
    assert r.launches==[3611]


def test_live_prior_process_is_not_carried(ready,monkeypatch):
    r=ready;accepted=entry.run_once(r.ctx);monkeypatch.setattr(carried,'_group_exists',lambda pid:True)
    with pytest.raises(ValueError,match='remains live'):carried._verify_first(r.ctx,r.source,r.imported,accepted)


def test_wrong_seed_path_is_not_first_pair(ready):
    with pytest.raises(ValueError,match='Only original first'):
        carried._verify_first(ready.ctx,ready.source,ready.imported,{'path':'evidence/inference/3512/accepted.json'})


def test_anchor_hash_tampering_rejected(tmp_path,monkeypatch):
    p=tmp_path/'audit.json';p.write_text('{}')
    monkeypatch.setattr(carried,'ROOT',tmp_path);monkeypatch.setattr(carried,'ANCHORS',{'prefix':('audit.json','0'*64)})
    with pytest.raises(ValueError,match='anchor changed'):carried._anchor('prefix')


def test_frozen_carried_import_replayed_not_trusted(monkeypatch):
    calls=[];actual={'seed':3611,'files':{'first':'exact'},'remaining_seeds':[3612,3613]}
    def replay(*args):calls.append(True);return actual
    monkeypatch.setattr(carried,'collect_carried_pair',replay)
    assert carried.verify_carried_pair(3,None,None,actual)==actual
    with pytest.raises(ValueError,match='import differs'):
        carried.verify_carried_pair(3,None,None,dict(actual,seed=3612))
    assert calls==[True,True]
