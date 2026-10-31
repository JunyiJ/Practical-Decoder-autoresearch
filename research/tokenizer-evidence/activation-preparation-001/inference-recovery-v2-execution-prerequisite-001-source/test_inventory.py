"""Real parent/child monitor concurrency and disposable immutable evidence faults."""
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research import tokenizer_wide_hybrid_metadata as metadata
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
from src.research.tokenizer_activation_timing import slot,CONTRACT
from research.activation_inference_recovery_v2 import inventory as inv


@pytest.fixture
def writer(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir();(root/'.lock').touch()
    global_lock=tmp_path/'global.lock';global_lock.touch();monkeypatch.setattr(cleanup,'GLOBAL_LOCK',global_lock)
    with global_lock.open('rb') as g,(root/'.lock').open('rb') as c:
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(c,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield MetadataWriter(root,(g.fileno(),c.fileno()))


def seed_prior(writer,seed):
    stem=f'evidence/inference/{slot(seed)}'
    writer.publish_json(stem+'/result.json',{'synthetic':'unchanged'})
    writer.publish_json(stem+'/accepted.json',{'synthetic':'not scientific acceptance'})
    writer.publish_json(f'control/recovery-worker-inference-{slot(seed)}.json',{'synthetic':'completed'})


@pytest.mark.parametrize('prior,current',[(3611,3612),(3612,3613)])
@pytest.mark.parametrize('case',['allowed','same_size_prior','prior_control','extra_prior','unknown_control'])
def test_actual_later_child_prior_review_during_parent_monitor_updates(writer,prior,current,case):
    seed_prior(writer,prior);physical=slot(current)
    ack=f'control/recovery-entry-inference-{physical}.json';writer.publish_json(ack,{'ack':False})
    faulted=[]
    def parent_check():
        log=writer.root/f'evidence/inference/{physical}/process/worker.log'
        if faulted or not log.exists() or log.stat().st_size<8193:return
        # Inject changes only after the child captured its prior-slot proof.
        if case=='same_size_prior':
            p=writer.root/f'evidence/inference/{slot(prior)}/result.json'
            old=p.read_bytes();new=old.replace(b'unchanged',b'corrupted');assert len(new)==len(old);p.write_bytes(new)
        elif case=='prior_control':
            writer.publish_json(f'control/recovery-worker-inference-{slot(prior)}.json',{'synthetic':'changed'},replace=True)
        elif case=='extra_prior':writer.publish_json(f'evidence/inference/{slot(prior)}/unexpected.json',{})
        elif case=='unknown_control':writer.publish_json('control/rogue.json',{})
        writer.publish_json(ack,{'ack':True},replace=True);faulted.append(True)
    result=run_process(writer,'inference',physical,[sys.executable,'-m','research.activation_inference_recovery_v2.concurrent_probe',
        str(writer.root),str(cleanup.GLOBAL_LOCK),*map(str,writer.lock_fds),str(prior),str(current),case],
        Path(__file__).resolve().parents[2],required_free_bytes=8*1024**3,output_limit_bytes=5*1024**2,
        timeout_seconds=15,poll_seconds=.01,parent_check=parent_check)
    assert result['status']=='completed' and result['returncode']==0 and faulted==[True]
    lines=(writer.root/f'evidence/inference/{physical}/process/worker.log').read_text().splitlines()
    reports=[json.loads(line.removeprefix('PROBE_RESULT ')) for line in lines if line.startswith('PROBE_RESULT ')]
    assert len(reports)==1
    report=reports[0];assert report['real_parent_monitor_changed'] is True
    if case=='allowed':
        assert report['status']=='passed' and report['old_whole_root_equality'] is False and report['immutable_slot_equal'] is True
    else:assert report['status']=='rejected_expected'


@pytest.mark.parametrize('case',['unknown','temporary','failure','stop','symlink','external_alias','checkpoint','cap','files','disk'])
def test_global_guards_not_exempted_by_scoped_proof(writer,tmp_path,monkeypatch,case):
    seed_prior(writer,3612)
    if case=='unknown':writer.publish_json('evidence/inference/3513/unexpected.json',{})
    elif case=='temporary':(writer.root/'control/pending.json.tmp').write_bytes(b'partial')
    elif case=='failure':writer.publish_json('evidence/inference/3513/dispatch-failure.json',{'retry_allowed':False})
    elif case=='stop':(writer.root/'STOP').touch()
    elif case=='symlink':(writer.root/'control/linked').symlink_to(tmp_path)
    elif case=='external_alias':os.link(writer.root/'control/.metadata.lock',tmp_path/'external')
    elif case=='checkpoint':
        p=writer.root/'evidence/quality/00/checkpoints/raw.pt';p.parent.mkdir(parents=True);p.write_bytes(b'forbidden')
    elif case=='cap':
        with (writer.root/'control/large').open('wb') as f:f.truncate(metadata.budgets()['control']+1)
    elif case=='files':monkeypatch.setattr(metadata,'MAX_FILES',2)
    else:monkeypatch.setattr(inv.shutil,'disk_usage',lambda p:SimpleNamespace(free=CONTRACT['reserve_bytes']-1))
    with pytest.raises((ValueError,OSError)):inv.slot_inventory(writer,3612)


def test_budget_and_content_proof_do_not_skip_distinct_temporary_bytes(writer):
    seed_prior(writer,3612)
    p=writer.root/'evidence/inference/3513/result.json';p.parent.mkdir(parents=True)
    p.write_bytes(b'a'*17);p.with_name('result.json.tmp').write_bytes(b'b'*23)
    observed=metadata.snapshot(writer.root)
    assert observed['groups']['evidence/inference/3513/other']==40
    with pytest.raises(ValueError,match='partial'):inv.slot_inventory(writer,3612)


def test_inventory_elapsed_limit_and_remaining_capacity(writer,monkeypatch):
    seed_prior(writer,3612);seen=inv.global_observation(writer)
    needed=CONTRACT['reserve_bytes']+inv.METADATA-seen['metadata_bytes']
    monkeypatch.setattr(inv.shutil,'disk_usage',lambda p:SimpleNamespace(free=needed-1))
    inv.global_observation(writer)
    with pytest.raises(ValueError,match='allocation'):inv.global_observation(writer,capacity=True)
    clock=iter((0,inv.SECONDS));monkeypatch.setattr(inv,'time',SimpleNamespace(monotonic=lambda:next(clock)))
    with pytest.raises(ValueError,match='time limit'):inv.slot_inventory(writer,3612)


@pytest.mark.parametrize('seed',[True,3512,3614,None])
def test_invalid_or_empty_scope_not_accepted(writer,seed):
    with pytest.raises(ValueError):
        inv.slot_inventory(writer,seed)


def test_empty_valid_slot_is_not_completion_proof(writer):
    with pytest.raises(ValueError,match='No slot evidence'):inv.slot_inventory(writer,3612)
