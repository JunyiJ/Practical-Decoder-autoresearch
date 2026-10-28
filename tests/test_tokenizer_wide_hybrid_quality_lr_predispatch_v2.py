"""Synthetic request handoff only, preserving every input; no process launch."""
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_lr_startup_v2 as start
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_lr_startup_v2 import setup


def prepared(writer,monkeypatch,slot=1):
    args=setup(writer,monkeypatch,slot)
    return args,start.prepare_lr_request(**args)


@pytest.mark.parametrize('slot',[1,4,8])
def test_reaudits_complete_prefix_and_history_without_mutating(writer,monkeypatch,slot):
    args,ref=prepared(writer,monkeypatch,slot);root=writer.root/'evidence';before=start._inventory(root)
    report=json.loads((root/ref['path']).read_text())
    checked=start.verify_lr_request(writer,slot,ref)
    assert checked['required_free_bytes']==report['storage']['required_free_bytes']
    assert checked['lr_request_verified'] and checked['lr_prefix_verified'] and checked['pilot_history_verified']
    assert not checked['dispatch_authorized'] and start._inventory(root)==before
    assert len(list(root.rglob('*.pt')))==slot*2


@pytest.mark.parametrize('key,value',[
    ('required_free_bytes',8*1024**3),('occupied_checkpoint_capacity_bytes',0),
    ('checkpoint_logical_bytes',1),('cleanup_boundaries_verified',True),
    ('dispatch_authorized',True),('frozen_plan_verified',True),('audit_semantics_verified',True),
    ('storage_projection_fits',False),('current_phase_space_verified',False),
    ('current_inventory_sha256','0'*64),('projection',{})])
def test_saved_storage_cannot_replace_independent_full_panel_reconstruction(writer,monkeypatch,key,value):
    args,ref=prepared(writer,monkeypatch);root=writer.root/'evidence';path=root/ref['path']
    report=json.loads(path.read_text());report['storage'][key]=value
    path.write_text(json.dumps(report));ref['sha256']=file_digest(path);before=start._inventory(root)
    with pytest.raises(ValueError):start.verify_lr_request(writer,1,ref)
    assert start._inventory(root)==before


@pytest.mark.parametrize('bad',['request','prior_weight','prior_tensor','source','numeric','pilot_receipt',
    'guard','attempt','new_process','new_weight','later_slot','failed','partial','stop',
    'records','pairs','context','selection','slot','claim'])
def test_changed_failed_or_active_state_prevents_handoff(writer,monkeypatch,bad):
    args,ref=prepared(writer,monkeypatch);root=writer.root/'evidence'
    paths={'request':'quality/01/request.json','prior_weight':'quality/00/checkpoints/raw.pt',
        'prior_tensor':'quality/00/tensor-audit.json','source':'meta/source.txt','numeric':'meta/numerical.json',
        'pilot_receipt':'.retention/pilot/receipt.json','attempt':'meta/quality-lr-startup-01-attempt.json',
        'failed':'meta/quality-lr-startup-01-failure.json','partial':'meta/quality-lr-startup-01.json.tmp'}
    if bad in paths:(root/paths[bad]).write_text('{}')
    elif bad=='guard':(writer.root/'control/quality-lr-startup-01.json').write_text('{}')
    elif bad=='stop':(writer.root/'STOP').touch()
    elif bad=='new_process':writer.publish_json('evidence/quality/01/process/launch.json',{})
    elif bad=='new_weight':
        p=root/'quality/01/checkpoints/raw.pt';p.parent.mkdir();p.write_bytes(b'unknown')
    elif bad=='later_slot':writer.publish_json('evidence/quality/02/request.json',{})
    else:
        path=root/ref['path'];report=json.loads(path.read_text())
        if bad=='records':report['completed_records'][0]['ce']=4.
        elif bad=='pairs':report['checkpoint_pairs']={}
        elif bad=='context':report['context']['source_sha256']='0'*64
        elif bad=='selection':report['selection']={}
        elif bad=='slot':report['slot']=True
        else:report['dispatch_authorized']=True
        path.write_text(json.dumps(report));ref['sha256']=file_digest(path)
    before=start._inventory(root)
    with pytest.raises(ValueError):start.verify_lr_request(writer,1,ref)
    assert start._inventory(root)==before


@pytest.mark.parametrize('offset',[0,-1])
def test_actual_capacity_rechecked_at_exact_remaining_threshold(writer,monkeypatch,offset):
    args,ref=prepared(writer,monkeypatch);root=writer.root/'evidence'
    report=json.loads((root/ref['path']).read_text());needed=report['storage']['required_free_bytes']
    monkeypatch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=needed+offset)))
    if offset==0:assert start.verify_lr_request(writer,1,ref)['required_free_bytes']==needed
    else:
        with pytest.raises(ValueError,match='remaining panel no longer fits'):start.verify_lr_request(writer,1,ref)


def test_space_loss_at_final_handoff_recheck_is_rejected(writer,monkeypatch):
    args,ref=prepared(writer,monkeypatch);root=writer.root/'evidence'
    needed=json.loads((root/ref['path']).read_text())['storage']['required_free_bytes'];values=iter([needed,needed-1])
    monkeypatch.setattr(start,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=next(values))))
    with pytest.raises(ValueError,match='Remaining storage changed'):start.verify_lr_request(writer,1,ref)


def test_pending_request_never_gets_completed_or_checkpoint_credit(writer,monkeypatch):
    args,ref=prepared(writer,monkeypatch);seen=[];original=start.bind_lr_prefix
    def prefix(*a,**kw):
        assert len(kw['requests'])==1 and kw['pending_request']['spec']['family']=='gated_mla'
        result=original(*a,**kw)
        assert len(result['records'])==1 and set(result['checkpoint_pairs'])=={'quality/00'}
        seen.append(True);return result
    monkeypatch.setattr(start,'bind_lr_prefix',prefix)
    start.verify_lr_request(writer,1,ref)
    assert seen==[True]


@pytest.mark.parametrize('when',['entry','handoff'])
def test_exact_worker_locks_are_rechecked_before_handoff(writer,monkeypatch,when):
    args,ref=prepared(writer,monkeypatch);bad=(writer.lock_fds[0],writer.lock_fds[0])
    if when=='entry':monkeypatch.setattr(writer,'lock_fds',bad)
    else:
        original=start._frozen;calls=[]
        def change(*a,**kw):
            result=original(*a,**kw);calls.append(True)
            if len(calls)==2:monkeypatch.setattr(writer,'lock_fds',bad)
            return result
        monkeypatch.setattr(start,'_frozen',change)
    with pytest.raises(ValueError,match='lock descriptors'):start.verify_lr_request(writer,1,ref)
