"""Disposable tiny/sparse files and real advisory locks; no training or corpus."""
import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pytest

from src.research import tokenizer_wide_hybrid_metadata_writer as writing
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research import tokenizer_wide_hybrid_metadata as metadata


@pytest.fixture
def writer(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir()
    global_lock=tmp_path/'global.lock';global_lock.touch();(root/'.lock').touch()
    monkeypatch.setattr(cleanup,'GLOBAL_LOCK',global_lock)
    monkeypatch.setattr(writing.shutil,'disk_usage',lambda p:SimpleNamespace(free=64*1024**3))
    with global_lock.open('rb') as g,(root/'.lock').open('rb') as c:
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(c,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield writing.MetadataWriter(root,(g.fileno(),c.fileno()))


def sparse(root,name,n):
    path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as out:out.truncate(n)
    return path


def test_exclusive_publish_and_control_replacement(writer):
    p=writer.publish_json('evidence/quality/00/result.json',{'finite':1,'unicode':'你好'})
    assert json.loads(p.read_text())=={'finite':1,'unicode':'你好'}
    assert not p.with_name(p.name+'.tmp').exists()
    with pytest.raises(FileExistsError):writer.publish_json('evidence/quality/00/result.json',{'different':True})
    writer.publish_json('control/state.json',{'stage':1})
    p=writer.publish_json('control/state.json',{'stage':2},replace=True)
    assert json.loads(p.read_text())=={'stage':2}
    assert metadata.snapshot(writer.root)['metadata_bytes']==sum(p.stat().st_size for p in writer.root.rglob('*') if p.is_file())


@pytest.mark.parametrize('name',['evidence/quality/00/result.json','evidence/meta/plan.json','emergency/failure.json','STOP'])
def test_immutable_evidence_cannot_be_replaced(writer,name):
    p=writer.publish_json(name,{'before':True});before=p.read_bytes()
    with pytest.raises(ValueError):writer.publish_json(name,{'after':True},replace=True)
    assert p.read_bytes()==before


@pytest.mark.parametrize('name',['.lock','control/.metadata.lock','control/result.json.tmp',
    'evidence/quality/00/checkpoints/raw.pt','../outside','/absolute','evidence/quality/24/result.json',
    'evidence/inference/3511/checkpoints/raw.pt'])
def test_reserved_or_unknown_destinations(writer,name):
    with pytest.raises(ValueError):writer.publish_json(name,{'x':1})
    with pytest.raises(ValueError):writer.open_append(name)


def test_partial_failed_json_stays_and_blocks_reentry(writer):
    name='evidence/quality/00/result.json'
    with pytest.raises(ValueError):writer.publish_json(name,{'before':1,'bad':float('nan')})
    tmp=writer.root/(name+'.tmp');assert tmp.exists()
    before=tmp.read_bytes()
    assert not (writer.root/name).exists()
    with pytest.raises(FileExistsError):writer.publish_json(name,{'retry':True})
    assert tmp.read_bytes()==before


def test_atomic_overlap_counts_old_and_partial_new_payload(writer):
    path=sparse(writer.root,'control/state.json',2*1024**2)
    before=path.read_bytes()
    with pytest.raises(ValueError,match='cap'):writer.publish_json('control/state.json',{'large':'a'*(2*1024**2)},replace=True)
    assert path.read_bytes()==before
    assert (writer.root/'control/state.json.tmp').exists()
    assert metadata.snapshot(writer.root)['groups']['control']<=3*1024**2


def test_other_files_in_group_reduce_capacity(writer):
    cap=metadata.budgets()['evidence/quality/00/other']
    sparse(writer.root,'evidence/quality/00/request.json',cap-20)
    with pytest.raises(ValueError,match='cap'):writer.publish_json('evidence/quality/00/result.json',{'a':'x'*50})
    assert metadata.snapshot(writer.root)['groups']['evidence/quality/00/other']<=cap


def test_append_exact_boundary_and_one_byte_over_preserved(writer):
    name='evidence/quality/00/resource-events.jsonl';cap=metadata.budgets()['evidence/quality/00/events']
    stream=writer.open_append(name)
    assert stream.write(b'x'*cap)==cap
    with pytest.raises(ValueError,match='cap'):stream.write(b'!')
    assert stream.closed and (writer.root/name).stat().st_size==cap
    with pytest.raises(ValueError):stream.write(b'')
    with pytest.raises(FileExistsError):writer.open_append(name)


def test_multiple_streams_share_aggregate_group_cap(writer):
    cap=metadata.budgets()['evidence/quality/00/other']
    a=writer.open_append('evidence/quality/00/a.jsonl');b=writer.open_append('evidence/quality/00/b.jsonl')
    try:
        a.write(b'a'*(cap//2));b.write(b'b'*(cap//2))
        with pytest.raises(ValueError):a.write(b'a')
        with pytest.raises(ValueError):b.write(b'b')
    finally:a.close();b.close()
    assert metadata.snapshot(writer.root)['groups']['evidence/quality/00/other']==cap


@pytest.mark.parametrize('action',['append','json'])
def test_disk_reserve_checked_before_payload(writer,monkeypatch,action):
    if action=='append':
        stream=writer.open_append('evidence/pilots/00/process/worker.log')
        monkeypatch.setattr(writing.shutil,'disk_usage',lambda p:SimpleNamespace(free=writing.RESERVE))
        with pytest.raises(ValueError):stream.write(b'!')
        assert (writer.root/stream.name).stat().st_size==0
    else:
        monkeypatch.setattr(writing.shutil,'disk_usage',lambda p:SimpleNamespace(free=writing.RESERVE))
        with pytest.raises(ValueError):writer.publish_json('control/state.json',{'x':1})
        assert (writer.root/'control/state.json.tmp').stat().st_size==0
        assert not (writer.root/'control/state.json').exists()


@pytest.mark.parametrize('bad',['replace','grow','symlink','hardlink'])
def test_changed_open_stream_identity_or_size_refused(writer,bad):
    name='evidence/pilots/00/process/worker.log';stream=writer.open_append(name);path=writer.root/name
    if bad=='replace':path.unlink();path.write_bytes(b'')
    elif bad=='grow':path.write_bytes(b'external')
    elif bad=='symlink':path.unlink();path.symlink_to(writer.root/'.lock')
    else:os.link(path,path.with_name('alias'))
    with pytest.raises(ValueError):stream.write(b'must not be written')
    assert stream.closed


def test_symlink_ancestor_rejected(writer,tmp_path):
    (writer.root/'evidence').symlink_to(tmp_path,target_is_directory=True)
    with pytest.raises(ValueError):writer.publish_json('evidence/meta/plan.json',{})


def test_wrong_global_lock_descriptor_rejected(writer,tmp_path):
    p=tmp_path/'wrong';p.touch()
    with p.open('rb') as wrong:
        with pytest.raises(ValueError):writing.MetadataWriter(writer.root,(wrong.fileno(),writer.lock_fds[1]))


def test_metadata_lock_contention_times_out_without_payload(writer,monkeypatch):
    monkeypatch.setattr(writing,'LOCK_SECONDS',.02)
    with (writer.root/writing.LOCK_NAME).open('rb') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(TimeoutError):writer.publish_json('control/state.json',{})
    assert not (writer.root/'control/state.json.tmp').exists()


def test_thread_writers_serialize_group_accounting(writer):
    cap=metadata.budgets()['evidence/pilots/00/other']
    streams=[writer.open_append(f'evidence/pilots/00/stream{i}') for i in range(2)]
    def work(stream):
        try:stream.write(b'x'*(cap//2+1));return True
        except ValueError:return False
        finally:stream.close()
    with ThreadPoolExecutor(max_workers=2) as pool:result=list(pool.map(work,streams))
    assert sorted(result)==[False,True]
    assert metadata.snapshot(writer.root)['groups']['evidence/pilots/00/other']==cap//2+1


def test_file_count_gate_accounts_for_atomic_temporary_names(writer,monkeypatch):
    monkeypatch.setattr(writing,'MAX_FILES',3)  # global campaign .lock and metadata lock already count as two
    with pytest.raises(ValueError,match='file-count'):writer.publish_json('control/state.json',{})
    stream=writer.open_append('control/log');stream.close()
    with pytest.raises(ValueError,match='file-count'):writer.open_append('control/log2')


def test_partial_io_error_preserved_and_stream_closed(writer,monkeypatch):
    stream=writer.open_append('evidence/pilots/00/process/worker.log')
    real=writing._write
    def fail(fd,data):real(fd,data[:2]);raise OSError('synthetic interrupted write')
    monkeypatch.setattr(writing,'_write',fail)
    with pytest.raises(OSError):stream.write(b'abcdef')
    assert stream.closed and (writer.root/stream.name).read_bytes()==b'ab'


def test_serialization_failure_keeps_previous_control_state(writer):
    p=writer.publish_json('control/state.json',{'good':True});before=p.read_bytes()
    with pytest.raises(TypeError):writer.publish_json('control/state.json',{'bad':object()},replace=True)
    assert p.read_bytes()==before and (writer.root/'control/state.json.tmp').exists()


def test_child_writers_with_inherited_worker_locks_share_metadata_budget(writer):
    import subprocess
    import sys
    code='''
import sys
from pathlib import Path
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
cleanup.GLOBAL_LOCK=Path(sys.argv[2])
w=MetadataWriter(Path(sys.argv[1]),tuple(map(int,sys.argv[3:5])))
with w.open_append('evidence/pilots/00/child'+sys.argv[5]) as stream:
    try:stream.write(b'x'*(128*1024+1))
    except ValueError:raise SystemExit(3)
'''
    def child(index):
        return subprocess.run([sys.executable,'-c',code,str(writer.root),str(cleanup.GLOBAL_LOCK),
            *map(str,writer.lock_fds),str(index)],pass_fds=writer.lock_fds,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(child,range(2)))
    assert sorted(p.returncode for p in results)==[0,3],[(p.returncode,p.stderr) for p in results]
    assert metadata.snapshot(writer.root)['groups']['evidence/pilots/00/other']==128*1024+1


def test_bad_byte_type_closes_stream(writer):
    stream=writer.open_append('control/log')
    with pytest.raises(TypeError):stream.write('text is not bytes')
    assert stream.closed


def test_newline_accounted_at_exact_json_budget(writer):
    name='evidence/pilots/00/report.json';cap=metadata.budgets()['evidence/pilots/00/other']
    sparse(writer.root,'evidence/pilots/00/occupied',cap-2)
    with pytest.raises(ValueError,match='newline'):writer.publish_json(name,{})
    assert (writer.root/(name+'.tmp')).read_bytes()==b'{}'
    assert not (writer.root/name).exists()
    assert metadata.snapshot(writer.root)['groups']['evidence/pilots/00/other']==cap


def test_compact_encoding_preserves_all_fields_and_existing_bytes(writer):
    value={'z':[1,{'nested':'你好','escaped':'line\nnext'}],'a':False,'none':None}
    first=writer.publish_json('evidence/meta/first.json',value)
    encoded=json.dumps(value,sort_keys=True,allow_nan=False,ensure_ascii=True,separators=(',',':')).encode('ascii')+b'\n'
    assert first.read_bytes()==encoded and json.loads(encoded)==value
    second=writer.publish_json('evidence/meta/second.json',{'new':value})
    assert first.read_bytes()==encoded and json.loads(second.read_bytes())=={'new':value}
    assert metadata.snapshot(writer.root)['groups']['evidence/meta']==len(encoded)+second.stat().st_size
