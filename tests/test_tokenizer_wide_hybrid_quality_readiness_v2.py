"""Complete synthetic idle audits -> immutable seal; no process or model."""
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_readiness_v2 as ready
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_startup_v2 import setup as first_setup
from test_tokenizer_wide_hybrid_quality_lr_startup_v2 import setup as later_setup


def prepared(writer,monkeypatch,slot):
    if slot==0:
        args=first_setup(writer,monkeypatch)
        ref=ready.first.prepare_first_quality_request(**args)
    else:
        args=later_setup(writer,monkeypatch,slot)
        ref=ready.later.prepare_lr_request(**args)
    monkeypatch.setattr(ready,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3)))
    return ref


@pytest.mark.parametrize('slot',[0,1,4,8])
def test_full_idle_review_is_sealed_without_changing_prior_evidence(writer,monkeypatch,slot):
    startup=prepared(writer,monkeypatch,slot);root=writer.root/'evidence';before=ready._inventory(root)
    ref=ready.seal_quality_readiness(writer,slot,startup)
    report=json.loads((root/ref['path']).read_text());after=ready._inventory(root)
    assert all(after[n]==v for n,v in before.items()) and set(after)-set(before)=={ref['path']}
    assert report['inventory']==before and report['startup_ref']==startup
    assert report['predispatch_review']['pilot_history_verified']
    assert report['required_free_bytes']>8*1024**3 and report['output_limit_bytes']>1024**3
    assert report['control_inventory'][report['guard']['path']]['sha256']==report['guard']['sha256']
    assert report['frozen_files'] and not report['dispatch_authorized'] and not report['deletion_authorized']
    assert not (root/f'quality/{slot:02d}/process').exists()
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,slot,startup)
    assert ready._inventory(root)==after


@pytest.mark.parametrize('bad',['source','numeric','request','weight','pilot_receipt','startup_guard','process','stop','space'])
def test_failed_audit_preserves_inputs_and_blocks_retry(writer,monkeypatch,bad):
    startup=prepared(writer,monkeypatch,1);root=writer.root/'evidence'
    paths={'source':'meta/source.txt','numeric':'meta/numerical.json','request':'quality/01/request.json',
           'weight':'quality/00/checkpoints/raw.pt','pilot_receipt':'.retention/pilot/receipt.json'}
    if bad in paths:(root/paths[bad]).write_text('{}')
    elif bad=='startup_guard':(writer.root/'control/quality-lr-startup-01.json').write_text('{}')
    elif bad=='process':writer.publish_json('evidence/quality/01/process/launch.json',{})
    elif bad=='stop':(writer.root/'STOP').touch()
    else:monkeypatch.setattr(ready,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
    before=ready._inventory(root)
    with pytest.raises(ValueError):ready.seal_quality_readiness(writer,1,startup)
    after=ready._inventory(root)
    assert all(after[n]==v for n,v in before.items())
    assert set(after)-set(before)=={'meta/quality-readiness-01-failure.json'}
    assert not (root/'meta/quality-readiness-01.json').exists()
    with pytest.raises(ValueError,match='cannot be retried'):ready.seal_quality_readiness(writer,1,startup)
    assert ready._inventory(root)==after


@pytest.mark.parametrize('bad',['control','evidence','bound','claims','locks','stop'])
def test_changes_during_full_review_cannot_be_sealed(writer,monkeypatch,bad):
    startup=prepared(writer,monkeypatch,1);root=writer.root/'evidence';original=ready.later.verify_lr_request
    def change(*a,**kw):
        checked=original(*a,**kw)
        if bad=='control':(writer.root/'control/quality-lr-startup-01.json').write_text('{}')
        elif bad=='evidence':(root/'quality/00/result.json').write_text('{}')
        elif bad=='bound':
            frozen=ready.later._frozen
            def changed(*a,**kw):
                bound,requests=frozen(*a,**kw);bound['files']['fake']='0'*64;return bound,requests
            monkeypatch.setattr(ready.later,'_frozen',changed)
        elif bad=='claims':checked['lr_prefix_verified']=False
        elif bad=='locks':monkeypatch.setattr(writer,'lock_fds',(writer.lock_fds[0],writer.lock_fds[0]))
        else:(writer.root/'STOP').touch()
        return checked
    monkeypatch.setattr(ready.later,'verify_lr_request',change)
    with pytest.raises(ValueError):ready.seal_quality_readiness(writer,1,startup)
    assert not (root/'meta/quality-readiness-01.json').exists()
    # An invalid inherited lock can also prevent failure publication; the guard
    # itself already exists and still makes another readiness attempt impossible.
    assert (writer.root/'control/quality-readiness-01.json').exists()


@pytest.mark.parametrize('offset',[0,-1])
def test_full_future_storage_exact_boundary(writer,monkeypatch,offset):
    startup=prepared(writer,monkeypatch,1);root=writer.root/'evidence'
    needed=json.loads((root/startup['path']).read_text())['storage']['required_free_bytes']
    monkeypatch.setattr(ready,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=needed+offset)))
    if offset==0:assert ready.seal_quality_readiness(writer,1,startup)['sha256']
    else:
        with pytest.raises(ValueError,match='remaining panel'):ready.seal_quality_readiness(writer,1,startup)


def test_late_space_loss_and_finished_guard_are_rechecked(writer,monkeypatch):
    startup=prepared(writer,monkeypatch,0);root=writer.root/'evidence'
    needed=json.loads((root/startup['path']).read_text())['storage']['required_free_bytes'];values=iter([needed,needed-1])
    monkeypatch.setattr(ready,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=next(values))))
    with pytest.raises(ValueError,match='before sealing'):ready.seal_quality_readiness(writer,0,startup)
    assert not (root/'meta/quality-readiness-00.json').exists()
    guard=json.loads((writer.root/'control/quality-readiness-00.json').read_text())
    assert guard['status']=='finished' and (root/'meta/quality-readiness-00-failure.json').exists()
