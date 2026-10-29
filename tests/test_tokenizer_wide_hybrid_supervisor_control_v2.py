"""Real tiny supervisor production and synthetic parent checks; no training."""
import copy
import json

import pytest

from src.research import tokenizer_wide_hybrid_quality_acceptance_v2 as accept
from src.research.protocol import file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_process_v2 import run
from test_tokenizer_wide_hybrid_quality_acceptance_v2 import setup


def test_real_supervisor_snapshot_matches_terminal_and_rejects_false_state(writer):
    terminal=run(writer,"print('tiny supervisor proof')")
    terminal_path=writer.root/'evidence/quality/00/process/process.json'
    ref=dict(path='quality/00/process/process.json',sha256=file_digest(terminal_path))
    control_path=writer.root/'control/process-quality-00.json';raw=control_path.read_bytes()
    saved=json.loads(raw);bound=accept._supervisor(writer,0,terminal,ref)
    assert bound==dict(path='control/process-quality-00.json',sha256=file_digest(control_path))
    assert saved['elapsed_seconds']<=terminal['elapsed_seconds']
    assert saved['observation']['output_bytes']+terminal_path.stat().st_size==terminal['observation']['output_bytes']
    mutations=[('status','running'),('returncode',True),('parent_pid',terminal['parent_pid']+1),
        ('pid',terminal['pid']+1),('argv',['false']),('error','synthetic failure'),
        ('required_free_bytes',1),('reserve_bytes',1),('output_limit_bytes',1),('log_limit_bytes',1),
        ('timeout_seconds',1),('capture_kind','file'),('log_truncated',True),
        ('elapsed_seconds',True),('elapsed_seconds',-1),('elapsed_seconds',terminal['elapsed_seconds']+1),
        ('extra_field',True),('observation',{}),('observation',None)]
    for key,value in [('free_bytes',1),('free_bytes',True),('output_bytes',0),
                      ('output_bytes',terminal['observation']['output_bytes']),('log_bytes',0)]:
        mutations.append(('observation',dict(saved['observation'],**{key:value})))
    for key,value in mutations:
        changed=copy.deepcopy(saved);changed[key]=value;control_path.write_text(json.dumps(changed))
        try:
            with pytest.raises(ValueError):accept._supervisor(writer,0,terminal,ref)
        finally:control_path.write_bytes(raw)
    # Free-space observations can legitimately differ before terminal output.
    changed=copy.deepcopy(saved);changed['observation']['free_bytes']+=1
    control_path.write_text(json.dumps(changed))
    try:assert accept._supervisor(writer,0,terminal,ref)!=bound
    finally:control_path.write_bytes(raw)
    assert file_digest(terminal_path)==ref['sha256']


def test_parent_rechecks_supervisor_bytes_after_full_storage_audit(writer,monkeypatch):
    args=setup(writer,monkeypatch,slot=0);path=writer.root/'control/process-quality-00.json'
    raw=path.read_bytes();original=accept.bind_storage
    def late(*a,**kw):
        result=original(*a,**kw)
        changed=json.loads(raw);changed['observation']['free_bytes']+=1
        path.write_text(json.dumps(changed));return result
    monkeypatch.setattr(accept,'bind_storage',late)
    try:
        with pytest.raises(ValueError):accept.review_completed_lr(**args)
    finally:path.write_bytes(raw)
