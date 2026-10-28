"""Real tiny serial child processes; no training, model, corpus or MPS."""
import json
import os
import sys
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_process_v2 as process
from src.research import tokenizer_wide_hybrid_metadata as metadata
from test_tokenizer_wide_hybrid_metadata_writer import writer


def run(writer,code,**options):
    args=dict(required_free_bytes=process.RESERVE+1024**2,output_limit_bytes=65536,
              log_limit_bytes=4096,timeout_seconds=5,poll_seconds=.02)
    args.update(options)
    return process.run_process(writer,'quality',0,[sys.executable,'-c',code],writer.root,**args)


def report(writer):
    return json.loads((writer.root/'evidence/quality/00/process/process.json').read_text())


def log(writer):return writer.root/'evidence/quality/00/process/worker.log'


def test_binary_merged_capture_locks_census_and_exclusive_terminal(writer):
    code=f'import os,fcntl\nfor fd in {writer.lock_fds!r}:\n os.fstat(fd)\n fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)\nos.write(1,b"a\\x00b");os.write(2,b"error")'
    result=run(writer,code)
    assert result==report(writer)
    assert result['status']=='completed' and result['returncode']==0
    assert result['capture_kind']=='bounded_pipe_v2' and not result['log_truncated']
    assert log(writer).read_bytes()==b'a\x00berror'
    assert not process._group_exists(result['pid'])
    census=metadata.snapshot(writer.root)
    total=sum(v['size'] for k,v in census['files'].items() if k.startswith('evidence/quality/00/'))
    assert result['observation']['output_bytes']==total
    before={k:(writer.root/k).read_bytes() for k in census['files']}
    with pytest.raises(FileExistsError):run(writer,'raise AssertionError()')
    assert before=={k:(writer.root/k).read_bytes() for k in census['files']}


@pytest.mark.parametrize('stage,slot,timeout',[('pilots',2,600),('quality',23,7200),('inference',3513,600)])
def test_default_stage_profiles(writer,stage,slot,timeout):
    result=process.run_process(writer,stage,slot,[sys.executable,'-c','pass'],writer.root,
        required_free_bytes=process.RESERVE+1024**2,output_limit_bytes=65536,log_limit_bytes=4096)
    assert result['timeout_seconds']==timeout and result['status']=='completed'


@pytest.mark.parametrize('n',[4096,4097,200000])
def test_fast_output_exact_cap_and_burst_overflow(writer,n):
    code=f'import os;os.write(1,b"x"*{n})'
    if n==4096:
        assert run(writer,code)['status']=='completed'
    else:
        with pytest.raises(RuntimeError,match='log limit'):run(writer,code)
        assert report(writer)['log_truncated'] and report(writer)['status']=='failed'
        assert not process._group_exists(report(writer)['pid'])
        with pytest.raises(FileExistsError):run(writer,'pass')
    assert log(writer).read_bytes()==b'x'*4096


def test_end_of_stream_does_not_mean_process_finished(writer):
    with pytest.raises(TimeoutError):
        run(writer,'import os,time;os.close(1);os.close(2);time.sleep(30)',timeout_seconds=.15)
    assert report(writer)['status']=='failed'
    assert not process._group_exists(report(writer)['pid'])


@pytest.mark.parametrize('ignore_term',[False,True])
def test_wall_timeout_reaps_owned_group(writer,ignore_term):
    code='import signal,time;'+('signal.signal(signal.SIGTERM,signal.SIG_IGN);' if ignore_term else '')+'time.sleep(30)'
    with pytest.raises(TimeoutError):run(writer,code,timeout_seconds=.15)
    assert report(writer)['returncode']<0 and not process._group_exists(report(writer)['pid'])


def test_child_failure_keeps_log_and_immutable_failure(writer):
    with pytest.raises(RuntimeError,match='code 86'):
        run(writer,"import sys;print('synthetic failure');sys.exit(86)")
    assert report(writer)['status']=='failed' and report(writer)['returncode']==86
    assert log(writer).read_text()=='synthetic failure\n'


def test_remaining_campaign_space_before_spawn(writer,monkeypatch):
    monkeypatch.setattr(process.subprocess,'Popen',lambda *a,**k:pytest.fail('must not spawn'))
    with pytest.raises(RuntimeError,match='remaining allocations'):
        run(writer,'pass',required_free_bytes=2**80)
    assert report(writer)['pid'] is None and report(writer)['status']=='failed'
    assert not log(writer).exists()


def test_spawn_failure_is_terminal(writer,monkeypatch):
    def failed(*a,**k):raise OSError('synthetic spawn failure')
    monkeypatch.setattr(process.subprocess,'Popen',failed)
    with pytest.raises(OSError,match='synthetic'):run(writer,'pass')
    assert report(writer)['pid'] is None and log(writer).stat().st_size==0
    with pytest.raises(FileExistsError):run(writer,'pass')


def test_stop_while_running_terminates_group(writer):
    code="from pathlib import Path;import time;Path('STOP').write_text('stop');time.sleep(30)"
    with pytest.raises(RuntimeError,match='STOP'):run(writer,code)
    assert report(writer)['status']=='failed' and not process._group_exists(report(writer)['pid'])


def test_sibling_output_in_slot_counts(writer):
    code="from pathlib import Path;Path('evidence/quality/00/extra.bin').write_bytes(b'x'*60000)"
    with pytest.raises(RuntimeError,match='output limit'):run(writer,code)
    assert report(writer)['status']=='failed'


def test_terminal_document_capacity_reserved_before_spawn(writer,monkeypatch):
    monkeypatch.setattr(process.subprocess,'Popen',lambda *a,**k:pytest.fail('must not spawn'))
    with pytest.raises(RuntimeError,match='terminal-report reserve'):
        run(writer,'pass',output_limit_bytes=1000,log_limit_bytes=100)
    assert report(writer)['pid'] is None


def test_live_descendant_detected_even_after_leader_exit(writer):
    code="import subprocess,sys;subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])"
    with pytest.raises(RuntimeError,match='descendants'):run(writer,code)
    assert report(writer)['status']=='failed'
    # The helper kills the owned group, including the orphan. Process-group
    # existence may temporarily include an OS-reaping zombie on some systems.
    assert report(writer)['returncode']==0


def test_cleanup_failure_diagnostic_preserves_original(writer,monkeypatch):
    original=process._stop
    def cleanup(p):
        original(p);raise OSError('synthetic cleanup report failure')
    monkeypatch.setattr(process,'_stop',cleanup)
    with pytest.raises(RuntimeError,match='code 3'):run(writer,'import sys;sys.exit(3)')
    assert 'synthetic cleanup' in report(writer)['cleanup_error']
    assert 'code 3' in report(writer)['error']


def test_failed_diagnostic_write_does_not_mask_error(writer,monkeypatch):
    original=writer.publish_json
    def publish(name,value,**kw):
        if name.endswith('process/process.json'):raise OSError('synthetic publication failure')
        return original(name,value,**kw)
    monkeypatch.setattr(writer,'publish_json',publish)
    with pytest.raises(RuntimeError,match='code 7') as error:run(writer,'import sys;sys.exit(7)')
    assert any('publication failure' in note for note in error.value.__notes__)
    assert json.loads((writer.root/'control/process-quality-00.json').read_text())['status']=='failed'
    with pytest.raises(FileExistsError):run(writer,'pass')


@pytest.mark.parametrize('options',[dict(required_free_bytes=1),dict(output_limit_bytes=True),
    dict(log_limit_bytes=process.LOG_LIMIT+1),dict(timeout_seconds=7201),dict(timeout_seconds=0),
    dict(poll_seconds=10),dict(timeout_seconds=float('nan')),dict(poll_seconds=True)])
def test_invalid_bounds_do_not_mutate(writer,options):
    before=metadata.snapshot(writer.root)
    with pytest.raises(ValueError):run(writer,'pass',**options)
    assert metadata.snapshot(writer.root)==before


@pytest.mark.parametrize('stage,slot',[('pilots',3),('quality',24),('quality',True),
    ('inference',3510),('unknown',0)])
def test_invalid_slots_do_not_mutate(writer,stage,slot):
    before=metadata.snapshot(writer.root)
    with pytest.raises(ValueError):
        process.run_process(writer,stage,slot,[sys.executable,'-c','pass'],writer.root,
                            required_free_bytes=process.RESERVE,output_limit_bytes=65536)
    assert metadata.snapshot(writer.root)==before


def test_oversized_command_metadata_rejected_without_attempt(writer):
    before=metadata.snapshot(writer.root)
    with pytest.raises(ValueError,match='document cap'):run(writer,'#'+'x'*process.DOCUMENT_LIMIT)
    assert metadata.snapshot(writer.root)==before


def test_child_checkpoint_metadata_producers_share_parent_census_lock(writer):
    repo=Path(__file__).resolve().parents[1]
    code=f'''import sys,time
sys.path.insert(0,{str(repo)!r})
from pathlib import Path
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_checkpoint_writer import CheckpointWriter
cleanup.GLOBAL_LOCK=Path({str(writer.root.parent/'global.lock')!r})
w=MetadataWriter(Path.cwd(),{writer.lock_fds!r})
c=CheckpointWriter(w,'quality',0,2048)
with c.open('raw.pt') as stream:
 for i in range(20):
  stream.write(b'x'*32)
  print('progress',i,flush=True)
  time.sleep(.002)
 stream.publish()
w.publish_json('evidence/quality/00/result.json',{{'synthetic_only':True}})
'''
    result=run(writer,code)
    assert result['status']=='completed'
    assert (writer.root/'evidence/quality/00/checkpoints/raw.pt').read_bytes()==b'x'*640
    assert len(log(writer).read_text().splitlines())==20
    census=metadata.snapshot(writer.root)
    assert result['observation']['output_bytes']==sum(v['size'] for k,v in census['files'].items()
                                                    if k.startswith('evidence/quality/00/'))


def test_uncoordinated_log_growth_fails_closed(writer):
    code="from pathlib import Path;Path('evidence/quality/00/process/worker.log').write_bytes(b'bypass')"
    with pytest.raises(RuntimeError,match='log size changed'):run(writer,code)
    assert report(writer)['status']=='failed'
    assert log(writer).read_bytes()==b'bypass'


def test_parent_interrupt_terminates_child_and_records_failure(writer,monkeypatch):
    original=process.snapshot;raised=False
    def inspect(root):
        nonlocal raised
        control=writer.root/'control/process-quality-00.json'
        if not raised and control.exists() and json.loads(control.read_text())['status']=='running':
            raised=True;raise KeyboardInterrupt('synthetic interrupt')
        return original(root)
    monkeypatch.setattr(process,'snapshot',inspect)
    with pytest.raises(KeyboardInterrupt):run(writer,'import time;time.sleep(30)')
    assert report(writer)['status']=='failed' and not process._group_exists(report(writer)['pid'])


def test_preexisting_partial_control_blocks_new_launch(writer):
    writer.publish_json('control/process-quality-00.json',{'status':'running'})
    with pytest.raises(FileExistsError):run(writer,'pass')
    assert not (writer.root/'evidence/quality/00/process/launch.json').exists()


@pytest.mark.parametrize('argv',[[],[''],['python',None],'python -c pass'])
def test_invalid_argv_before_attempt(writer,argv):
    before=metadata.snapshot(writer.root)
    with pytest.raises(ValueError):
        process.run_process(writer,'quality',0,argv,writer.root,
                            required_free_bytes=process.RESERVE,output_limit_bytes=65536)
    assert metadata.snapshot(writer.root)==before


def test_pipe_setup_failure_records_spawned_pid_and_reaps_child(writer,monkeypatch):
    def fail(fd,blocking):raise OSError('synthetic pipe setup failure')
    monkeypatch.setattr(process.os,'set_blocking',fail)
    with pytest.raises(OSError,match='pipe setup'):run(writer,'import time;time.sleep(30)')
    assert type(report(writer)['pid']) is int
    assert report(writer)['status']=='failed' and not process._group_exists(report(writer)['pid'])
