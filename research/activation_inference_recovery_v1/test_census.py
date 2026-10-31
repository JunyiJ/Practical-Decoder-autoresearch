"""Disposable metadata, actual flock/atomic writer/pipe runner; no model loads."""
import fcntl
import json
import os
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from research.activation_inference_recovery_v1 import census
from src.research import tokenizer_wide_hybrid_metadata as metadata
from src.research import tokenizer_wide_hybrid_metadata_writer as writing
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_wide_hybrid_process_v2 import run_process


@pytest.fixture
def writer(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    root.mkdir()
    global_lock = tmp_path / 'global.lock'
    global_lock.touch()
    (root / '.lock').touch()
    monkeypatch.setattr(cleanup, 'GLOBAL_LOCK', global_lock)
    with global_lock.open('rb') as g, (root / '.lock').open('rb') as c:
        fcntl.flock(g, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(c, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield writing.MetadataWriter(root, (g.fileno(), c.fileno()))


def sparse(root, name, size):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.truncate(size)
    return path


def install_disappearance(writer, monkeypatch):
    final = writer.publish_json('control/process-inference-3511.json', {'n': 0})
    temporary = final.with_name(final.name + '.tmp')
    temporary.write_bytes(b'{"n":1}\n')
    original = Path.stat
    fired = []
    def racing_stat(path, *args, **kwargs):
        if path == temporary and not fired:
            fired.append(True)
            os.replace(temporary, final)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'stat', racing_stat)
    return temporary, fired


def test_reproduces_original_enumeration_stat_race(writer, monkeypatch):
    temporary, fired = install_disappearance(writer, monkeypatch)
    with pytest.raises(FileNotFoundError) as exc:
        metadata.snapshot(writer.root)
    assert exc.value.filename == str(temporary) and fired == [True]


def test_uncooperative_disappearance_is_not_ignored_or_retried(writer, monkeypatch):
    temporary, fired = install_disappearance(writer, monkeypatch)
    with pytest.raises(FileNotFoundError) as exc:
        census.synchronized_snapshot(writer)
    assert exc.value.filename == str(temporary) and fired == [True]


def test_two_readers_wait_for_actual_atomic_control_writer(writer, monkeypatch):
    writer.publish_json('control/process-inference-3511.json', {'n': 0})
    entered, release = threading.Event(), threading.Event()
    scanned = threading.Event()
    real_replace, real_snapshot = os.replace, census.snapshot
    def paused_replace(source, target):
        entered.set()
        assert release.wait(1.5), 'test handoff timeout'
        return real_replace(source, target)
    def observed_snapshot(root):
        scanned.set()
        return real_snapshot(root)
    monkeypatch.setattr(writing.os, 'replace', paused_replace)
    monkeypatch.setattr(census, 'snapshot', observed_snapshot)
    with ThreadPoolExecutor(max_workers=3) as pool:
        producer = pool.submit(writer.publish_json, 'control/process-inference-3511.json', {'n': 1}, replace=True)
        assert entered.wait(1)
        readers = [pool.submit(census.synchronized_snapshot, writer) for _ in range(2)]
        try:
            assert not scanned.wait(.05), 'reader entered census during writer publication'
        finally:
            release.set()
        producer.result(timeout=2)
        results = [future.result(timeout=2) for future in readers]
    assert results[0] == results[1]
    assert 'control/process-inference-3511.json.tmp' not in results[0]['files']
    assert json.loads((writer.root/'control/process-inference-3511.json').read_text()) == {'n': 1}


def test_actual_parent_runner_and_child_census_with_control_updates(writer):
    # One bounded metadata-only child. The original supervisor already locks
    # its own scans; the child explicitly uses the new synchronized reader.
    script = '''
import sys
from pathlib import Path
from research.activation_inference_recovery_v1.census import synchronized_snapshot
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
cleanup.GLOBAL_LOCK=Path(sys.argv[2])
w=MetadataWriter(Path(sys.argv[1]),tuple(map(int,sys.argv[3:])))
for i in range(30):
    w.publish_json('control/child.json',{'i':i},replace=i>0)
    synchronized_snapshot(w)
    print('metadata sample',i,flush=True)
'''
    result = run_process(writer, 'inference', 3511,
        [sys.executable, '-c', script, str(writer.root), str(cleanup.GLOBAL_LOCK), *map(str, writer.lock_fds)],
        Path(__file__).resolve().parents[2], required_free_bytes=8*1024**3,
        output_limit_bytes=5*1024**2, timeout_seconds=15, poll_seconds=.01,
        parent_check=lambda: census.synchronized_snapshot(writer))
    assert result['status'] == 'completed' and result['returncode'] == 0
    assert json.loads((writer.root/'control/child.json').read_text()) == {'i': 29}
    log = (writer.root/'evidence/inference/3511/process/worker.log').read_text()
    assert len(log.splitlines()) == 30
    observed = census.synchronized_snapshot(writer)
    assert not any(name.endswith('.tmp') for name in observed['files'])


def test_distinct_temporary_bytes_and_exact_publication_alias(writer):
    a = sparse(writer.root, 'control/distinct.json', 17)
    sparse(writer.root, 'control/distinct.json.tmp', 23)
    b = sparse(writer.root, 'control/linked.json.tmp', 31)
    os.link(b, b.with_name('linked.json'))
    observed = census.synchronized_snapshot(writer)
    assert observed['groups']['control'] == 17+23+31
    assert observed['files'][str(a.relative_to(writer.root))]['size'] == 17
    assert len(observed['files']) == 6  # both locks and all four publication names
    assert observed['dispatch_authorized'] is False
    assert observed['transient_producer_enforcement_verified'] is False


@pytest.mark.parametrize('bad', ['symlink', 'external_hardlink', 'wrong_alias', 'unknown', 'fifo', 'checkpoint'])
def test_original_rejections_preserved(writer, tmp_path, bad):
    if bad == 'symlink':
        (writer.root/'control/bad').symlink_to(tmp_path)
    elif bad == 'external_hardlink':
        p = sparse(writer.root, 'control/bad', 1)
        os.link(p, tmp_path/'outside')
    elif bad == 'wrong_alias':
        p = sparse(writer.root, 'control/bad', 1)
        os.link(p, writer.root/'control/other')
    elif bad == 'unknown':
        sparse(writer.root, 'undeclared', 1)
    elif bad == 'fifo':
        os.mkfifo(writer.root/'control/pipe')
    else:
        sparse(writer.root, 'evidence/inference/3511/checkpoints/raw.pt', 1)
    with pytest.raises(ValueError):
        census.synchronized_snapshot(writer)


def test_group_cap_and_file_cap_preserved(writer, monkeypatch):
    p = sparse(writer.root, 'control/large', metadata.budgets()['control']+1)
    with pytest.raises(ValueError, match='allowance'):
        census.synchronized_snapshot(writer)
    p.unlink()  # disposable synthetic fixture only
    monkeypatch.setattr(metadata, 'MAX_FILES', 1)
    with pytest.raises(ValueError, match='file-count'):
        census.synchronized_snapshot(writer)


def test_writer_still_reserves_old_plus_temporary_bytes(writer):
    cap = metadata.budgets()['control']
    final = writer.publish_json('control/large.json', {'v': 'x'*(cap//2)})
    before = final.read_bytes()
    with pytest.raises(ValueError, match='cap'):
        writer.publish_json('control/large.json', {'v': 'y'*(cap//2)}, replace=True)
    observed = census.synchronized_snapshot(writer)
    assert final.read_bytes() == before
    assert 'control/large.json.tmp' in observed['files']
    assert observed['groups']['control'] <= cap
    assert observed['groups']['control'] > len(before)


def test_lock_timeout_does_not_scan_or_mutate(writer, monkeypatch):
    monkeypatch.setattr(writing, 'LOCK_SECONDS', .03)
    def forbidden(root):
        pytest.fail('scan before lock ownership')
    monkeypatch.setattr(census, 'snapshot', forbidden)
    before = metadata.snapshot(writer.root)
    with (writer.root/writing.LOCK_NAME).open('rb') as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(TimeoutError, match='lock timeout'):
            census.synchronized_snapshot(writer)
    assert metadata.snapshot(writer.root) == before


def test_revalidation_rejects_change_without_retry(writer, monkeypatch):
    calls = []
    original = census.snapshot
    def changed(root):
        result = original(root)
        calls.append(True)
        if len(calls) == 1:
            sparse(root, 'control/uncooperative', 2)
        return result
    monkeypatch.setattr(census, 'snapshot', changed)
    with pytest.raises(RuntimeError, match='changed while serialized'):
        census.synchronized_snapshot(writer)
    assert len(calls) == 2


def test_census_elapsed_bound_includes_scans(writer, monkeypatch):
    clock = iter((0.0, census.CENSUS_SECONDS))
    # Replace only the helper's clock object, not original writer's real clock.
    from types import SimpleNamespace
    monkeypatch.setattr(census, 'time', SimpleNamespace(monotonic=lambda: next(clock)))
    with pytest.raises(TimeoutError, match='census time limit'):
        census.synchronized_snapshot(writer)


def test_invalid_writer_rejected():
    with pytest.raises(ValueError, match='bounded metadata writer'):
        census.synchronized_snapshot(object())
