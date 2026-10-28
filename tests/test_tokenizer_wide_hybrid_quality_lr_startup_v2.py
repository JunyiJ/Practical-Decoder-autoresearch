"""Synthetic completed quality prefixes and pilot receipts; no model work."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_lr_startup_v2 as start
from src.research import tokenizer_wide_hybrid_quality_startup_v2 as first
from src.research import tokenizer_wide_hybrid_storage_v2 as storage
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_rules import lr_specs
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_startup_v2 import setup as pilot_setup
from test_tokenizer_wide_hybrid_retention_audit import fixture


def setup(writer,monkeypatch,slot=1):
    args=pilot_setup(writer,monkeypatch);root=writer.root/'evidence'
    old=first.bind_campaign
    requests=[dict(spec=s,config=dict(fixture_only=True,data=dict(path=str(root/'meta/corpus.txt'))),
        result_path=str(root/f'quality/{i:02d}/result.json')) for i,s in enumerate(lr_specs())]
    def campaign(*a):
        bound=old(*a);bound['requests']['quality']={str(i):[copy.deepcopy(r)] for i,r in enumerate(requests)}
        return bound
    monkeypatch.setattr(start,'bind_campaign',campaign)
    monkeypatch.setattr(start,'PROJECTION',first.PROJECTION)
    monkeypatch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    bound=campaign(root/'meta/plan.json',args['plan_sha256'])
    plan=json.loads((root/'meta/plan.json').read_text())
    fixture(root,producer_v2=True,prefix_count=slot,bindings=dict(context=bound['context'],
        projection_ref=args['projection_ref'],numerical_ref=plan['numerical_ref'],
        training_token_sha256=plan['training_token_sha256'],requests=requests))
    return dict(args,slot=slot)


@pytest.mark.parametrize('slot',[1,4,8])
def test_complete_prefix_and_full_storage_precede_next_request(writer,monkeypatch,slot):
    args=setup(writer,monkeypatch,slot);root=writer.root/'evidence';before=start._inventory(root)
    original=start.bind_storage;calls=[]
    def gate(*a,**kw):
        assert not (root/f'quality/{slot:02d}').exists()
        assert len(kw['records'])==slot and len(kw['checkpoint_pairs'])==slot+3
        assert kw['rates'] is None and kw['selection'] is None and len(kw['stages'])==1
        calls.append(slot);return original(*a,**kw)
    monkeypatch.setattr(start,'bind_storage',gate)
    ref=start.prepare_lr_request(**args);report=json.loads((root/ref['path']).read_text())
    assert calls==[slot] and len(report['completed_records'])==slot
    assert not report['dispatch_authorized'] and not report['deletion_authorized'] and report['selection'] is None
    assert report['storage']['required_free_bytes']>30_000_000_000
    assert report['storage']['checkpoint_logical_bytes']==sum(p.stat().st_size for p in root.rglob('*.pt'))
    assert len(list(root.rglob('*.pt')))==slot*2
    assert {p.name for p in (root/f'quality/{slot:02d}').iterdir()}=={'request.json'}
    assert json.loads((root/report['request']['path']).read_text())['spec']==lr_specs()[slot]
    assert all(start._inventory(root)[name]==entry for name,entry in before.items())
    guard=json.loads((writer.root/f'control/quality-lr-startup-{slot:02d}.json').read_text())
    assert guard['status']=='finished' and guard['limits']['timeout_seconds']==1800
    with pytest.raises(ValueError):start.prepare_lr_request(**args)


@pytest.mark.parametrize('bad',['weight','resource','tensor','numeric','source','corpus','pilot_journal',
    'pilot_guard','stop','wrong_request','later_slot','pending','failed','partial','extra','storage'])
def test_invalid_history_prefix_or_capacity_never_creates_request(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    paths={'weight':'quality/00/checkpoints/raw.pt','resource':'quality/00/resource-audit.json',
        'tensor':'quality/00/tensor-audit.json','numeric':'meta/numerical.json','source':'meta/source.txt',
        'corpus':'meta/corpus.txt','pilot_journal':'.retention/pilot/deletions.jsonl',
        'failed':'quality/00/worker-failure.json','partial':'quality/00/tensor-audit.json.tmp',
        'extra':'quality/00/unknown.json'}
    if bad in paths:(root/paths[bad]).write_text('{}')
    elif bad=='pilot_guard':(writer.root/'control/pilot-entry-00.json').write_text('{}')
    elif bad=='stop':(writer.root/'STOP').touch()
    elif bad=='wrong_request':
        original=start.bind_campaign
        def wrong(*a):
            bound=original(*a);bound['requests']['quality']['0'][0]['config']['fixture_only']=False;return bound
        monkeypatch.setattr(start,'bind_campaign',wrong)
    elif bad in ('later_slot','pending'):
        writer.publish_json('evidence/quality/'+('02' if bad=='later_slot' else '01')+'/request.json',{})
    else:monkeypatch.setattr(storage,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
    with pytest.raises(ValueError):start.prepare_lr_request(**args)
    if bad!='pending':
        assert not (root/'quality/01').exists()
        assert (root/'meta/quality-lr-startup-01-failure.json').exists()
    with pytest.raises(ValueError):start.prepare_lr_request(**args)


@pytest.mark.parametrize('when',['before_request','after_request'])
def test_source_change_during_preparation_retains_terminal_failure(writer,monkeypatch,when):
    args=setup(writer,monkeypatch);root=writer.root/'evidence'
    if when=='before_request':
        original=start.bind_storage
        def changed(*a,**kw):
            value=original(*a,**kw);(root/'meta/source.txt').write_text('changed');return value
        monkeypatch.setattr(start,'bind_storage',changed)
    else:
        original=writer.publish_json
        def changed(name,*a,**kw):
            value=original(name,*a,**kw)
            if name=='evidence/quality/01/request.json':(root/'meta/source.txt').write_text('changed')
            return value
        monkeypatch.setattr(writer,'publish_json',changed)
    with pytest.raises(ValueError):start.prepare_lr_request(**args)
    assert (root/'meta/quality-lr-startup-01-failure.json').exists()
    assert not (root/'meta/quality-lr-startup-01.json').exists()
    assert (root/'quality/01/request.json').exists()==(when=='after_request')
    with pytest.raises(ValueError):start.prepare_lr_request(**args)


@pytest.mark.parametrize('slot',[0,9,15,23,True])
def test_no_first_rank_fresh_or_boolean_slot_is_accepted(writer,slot):
    with pytest.raises(ValueError):start.prepare_lr_request(writer,slot,'0'*64,projection_ref={},pilot_completion_ref={})
    assert not (writer.root/'evidence/quality').exists()


@pytest.mark.parametrize('offset',[0,-1])
def test_exact_remaining_capacity_counts_occupied_weights_without_releasing_them(writer,monkeypatch,offset):
    args=setup(writer,monkeypatch,4);root=writer.root/'evidence'
    weights={p:file_digest(p) for p in root.rglob('*.pt')}
    occupied=sum(min(p.stat().st_size,p.stat().st_blocks*512) for p in weights)
    needed=30_631_281_412-occupied
    disk=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=needed+offset))
    monkeypatch.setattr(storage,'shutil',disk);monkeypatch.setattr(start,'shutil',disk)
    if offset==0:
        ref=start.prepare_lr_request(**args);report=json.loads((root/ref['path']).read_text())
        assert report['storage']['required_free_bytes']==needed
        assert report['storage']['occupied_checkpoint_capacity_bytes']==occupied
    else:
        with pytest.raises(ValueError,match='remaining panel does not fit'):start.prepare_lr_request(**args)
        assert not (root/'quality/04').exists()
    assert weights=={p:file_digest(p) for p in weights}
