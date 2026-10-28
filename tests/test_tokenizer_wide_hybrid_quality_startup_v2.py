"""Full synthetic pilot receipts -> first quality request, no model or dispatch."""
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_startup_v2 as start
from src.research import tokenizer_wide_hybrid_pilot_boundary_v2 as boundary
from src.research import tokenizer_wide_hybrid_pilot_startup_v2 as pilot_start
from src.research import tokenizer_wide_hybrid_storage_v2 as storage
from src.research.tokenizer_wide_hybrid_plan_v2 import request_options
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_pilot_boundary_v2 import prepared


def setup(writer,monkeypatch):
    disk=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3))
    for module in (start,pilot_start,storage):monkeypatch.setattr(module,'shutil',disk)
    args=prepared(writer,monkeypatch);completion=boundary.cleanup_pilot_boundary(**args)
    old=pilot_start.bind_campaign
    def campaign(*a):
        bound=old(*a);bound['requests']['quality']=request_options(writer.root)['quality'];return bound
    monkeypatch.setattr(start,'bind_campaign',campaign)
    monkeypatch.setattr(start,'PROJECTION',pilot_start.PROJECTION)
    return dict(args,pilot_completion_ref=completion)


def load(writer,ref):return json.loads((writer.root/'evidence'/ref['path']).read_text())


def history(writer,args):
    complete=load(writer,args['pilot_completion_ref'])
    return boundary.verify_pilot_history(writer,args['plan_sha256'],args['pilot_completion_ref'],
        current_inventory=pilot_start._inventory(writer.root/'evidence'),stages=[complete['chain_stage']])


def test_first_quality_storage_gate_precedes_request_and_predispatch_is_readonly(writer,monkeypatch):
    args=setup(writer,monkeypatch);root=writer.root/'evidence';before=pilot_start._inventory(root)
    old=start.bind_storage;calls=[]
    def gate(*a,**kw):
        assert not (root/'quality').exists() and len(kw['stages'])==1 and kw['records']==[]
        assert len(kw['checkpoint_pairs'])==3 and not list(root.rglob('*.pt'))
        calls.append('full storage');return old(*a,**kw)
    monkeypatch.setattr(start,'bind_storage',gate)
    ref=start.prepare_first_quality_request(**args);report=load(writer,ref)
    assert calls==['full storage'] and not report['dispatch_authorized']
    assert report['storage']['required_free_bytes']==30_631_281_412
    assert load(writer,report['request'])==request_options(writer.root)['quality']['0'][0]
    saved=pilot_start._inventory(root)
    checked=start.verify_first_quality_request(writer,ref)
    assert checked['first_quality_request_verified'] and not checked['dispatch_authorized']
    assert pilot_start._inventory(root)==saved
    assert all(saved[n]==v for n,v in before.items()) and not list(root.rglob('*.pt'))
    with pytest.raises(ValueError):start.prepare_first_quality_request(**args)


def test_history_accepts_appended_quality_metadata_without_weakening_strict_verifier(writer,monkeypatch):
    args=setup(writer,monkeypatch)
    writer.publish_json('evidence/meta/quality-new-observation.json',{'synthetic':True})
    writer.publish_json('evidence/quality/00/request.json',{'synthetic':True})
    report=history(writer,args)
    assert report['pilot_history_verified'] and not report['quality_audits_bound']
    assert not report['storage_verified'] and not report['dispatch_authorized']
    with pytest.raises(ValueError):boundary.verify_pilot_boundary(writer,args['plan_sha256'],args['pilot_completion_ref'])


@pytest.mark.parametrize('bad',['protected','journal','recreated','guard','extra','failure','partial','missing_chain'])
def test_history_cannot_excuse_changed_or_failed_past(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    if bad=='protected':(root/'pilots/00/result.json').write_text('{}')
    elif bad=='journal':(root/'.retention/pilot/deletions.jsonl').write_text('')
    elif bad=='recreated':(root/'pilots/00/checkpoints/raw.pt').write_bytes(b'recreated')
    elif bad=='guard':(writer.root/'control/pilot-entry-00.json').write_text('{}')
    elif bad=='extra':writer.publish_json('evidence/meta/unrelated.json',{})
    elif bad=='failure':writer.publish_json('evidence/meta/quality-failure.json',{})
    elif bad=='partial':(root/'meta/quality-new.json.tmp').write_text('{}')
    else:(root/'.retention/lr').mkdir()
    with pytest.raises(ValueError):history(writer,args)


@pytest.mark.parametrize('bad',['pilot_ref','source','projection','stop','quality_exists','storage'])
def test_preparation_failure_never_creates_request_or_allows_retry(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    if bad=='pilot_ref':args['pilot_completion_ref']['sha256']='0'*64
    elif bad=='source':(root/'meta/source.txt').write_text('changed')
    elif bad=='projection':args['projection_ref']['sha256']='0'*64
    elif bad=='stop':(writer.root/'STOP').touch()
    elif bad=='quality_exists':(root/'quality').mkdir()
    else:
        def fail(*a,**k):raise ValueError('synthetic full storage shortfall')
        monkeypatch.setattr(start,'bind_storage',fail)
    with pytest.raises(ValueError):start.prepare_first_quality_request(**args)
    assert not (root/'quality/00/request.json').exists()
    if bad!='quality_exists':assert (root/'meta/quality-first-startup-failure.json').exists()
    with pytest.raises(ValueError):start.prepare_first_quality_request(**args)


@pytest.mark.parametrize('bad',['request','guard','protected','new_process','new_weight','later_slot','stop','storage','report_total','attempt'])
def test_readonly_predispatch_rejects_changed_or_active_state(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);ref=start.prepare_first_quality_request(**args);root=writer.root/'evidence'
    if bad=='request':(root/'quality/00/request.json').write_text('{}')
    elif bad=='guard':(writer.root/'control/quality-first-startup.json').write_text('{}')
    elif bad=='protected':(root/'pilots/00/result.json').write_text('{}')
    elif bad=='new_process':writer.publish_json('evidence/quality/00/process/launch.json',{})
    elif bad=='new_weight':
        (root/'quality/00/checkpoints').mkdir();(root/'quality/00/checkpoints/raw.pt').write_bytes(b'unknown')
    elif bad=='later_slot':writer.publish_json('evidence/quality/01/request.json',{})
    elif bad=='stop':(writer.root/'STOP').touch()
    elif bad=='storage':monkeypatch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=30_631_281_411)))
    elif bad=='attempt':(root/'meta/quality-first-startup-attempt.json').write_text('{}')
    else:
        p=root/ref['path'];r=json.loads(p.read_text());r['storage']['required_free_bytes']=8*1024**3
        p.write_text(json.dumps(r));ref['sha256']=file_digest(p)
    before=pilot_start._inventory(root)
    with pytest.raises(ValueError):start.verify_first_quality_request(writer,ref)
    assert pilot_start._inventory(root)==before


@pytest.mark.parametrize('offset',[0,-1])
def test_full_panel_threshold_unchanged_after_pilot_deletion(writer,monkeypatch,offset):
    args=setup(writer,monkeypatch)
    disk=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=30_631_281_412+offset))
    monkeypatch.setattr(start,'shutil',disk);monkeypatch.setattr(storage,'shutil',disk)
    if offset==0:
        ref=start.prepare_first_quality_request(**args)
        assert start.verify_first_quality_request(writer,ref)['required_free_bytes']==30_631_281_412
    else:
        with pytest.raises(ValueError,match='remaining panel does not fit'):start.prepare_first_quality_request(**args)
        assert not (writer.root/'evidence/quality').exists()
