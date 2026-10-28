"""Bounded pipe capture and immutable terminal process evidence for v2 callers.

This subprocess primitive is not a campaign dispatcher. Frozen request/source,
full future storage and numerical/stage gates remain caller responsibilities;
the child must independently enforce RSS/MPS limits. Closed v1 code is unchanged.
"""
import json
import math
import os
from pathlib import Path
import selectors
import shutil
import subprocess
import time

from .tokenizer_serial_process import _group_exists, _stop
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_retention import RESERVE

LOG_LIMIT=4*1024**2
READ_BYTES=65536
DOCUMENT_LIMIT=16384


def _encoded_size(value):
    return len((json.dumps(value,sort_keys=True,allow_nan=False,indent=2,ensure_ascii=True)+'\n').encode('ascii'))


def run_process(writer,stage,slot,argv,cwd,*,required_free_bytes,output_limit_bytes,
                log_limit_bytes=LOG_LIMIT,timeout_seconds=None,poll_seconds=.2):
    """Fresh single child, bounded merged stdout/stderr, no shell or retry.

    Shorter limits support isolated tests; scientific acceptance independently
    requires the exact frozen pilot/quality/inference limits. Partial logs and
    mutable control state survive hard interruption; terminal evidence is exclusive.
    """
    if not isinstance(writer,MetadataWriter):raise ValueError('Bounded campaign writer required')
    valid=(stage=='pilots' and type(slot) is int and 0<=slot<3 or
           stage=='quality' and type(slot) is int and 0<=slot<24 or
           stage=='inference' and type(slot) is int and slot in (3511,3512,3513))
    if not valid:raise ValueError('Exact declared process slot required')
    maximum=7200 if stage=='quality' else 600
    timeout_seconds=maximum if timeout_seconds is None else timeout_seconds
    if (not isinstance(argv,(tuple,list)) or not argv
            or any(type(a) is not str or not a for a in argv)):
        raise ValueError('Explicit nonempty argv required; no shell')
    for value in (required_free_bytes,output_limit_bytes,log_limit_bytes):
        if type(value) is not int or value<=0:raise ValueError('Positive exact byte limits required')
    if required_free_bytes<RESERVE or log_limit_bytes>LOG_LIMIT or log_limit_bytes>output_limit_bytes:
        raise ValueError('Reserve/output policy violated')
    if (any(type(v) not in (int,float) or not math.isfinite(v) or v<=0
            for v in (timeout_seconds,poll_seconds))
            or not poll_seconds<=timeout_seconds<=maximum):
        raise ValueError('Invalid time/poll bounds')
    cwd=Path(cwd).resolve(strict=True)
    if not cwd.is_dir():raise ValueError('Existing cwd directory required')
    slot_name=str(slot) if stage=='inference' else f'{slot:02d}'
    stem=f'evidence/{stage}/{slot_name}'
    terminal=stem+'/process/process.json'
    control=f'control/process-{stage}-{slot_name}.json'
    state=dict(status='preflight',argv=list(argv),parent_pid=os.getpid(),pid=None,returncode=None,error=None,
               required_free_bytes=required_free_bytes,reserve_bytes=RESERVE,
               output_limit_bytes=output_limit_bytes,log_limit_bytes=log_limit_bytes,
               timeout_seconds=timeout_seconds,observation=None,
               capture_kind='bounded_pipe_v2',log_truncated=False)
    if _encoded_size(state)>DOCUMENT_LIMIT//2:raise ValueError('Process argv/document cap exceeded')
    started=time.monotonic();process=None;stream=None;selector=None;control_created=False
    log_bytes=0

    def observe(initial=False):
        with writer._serialized():
            if (writer.root/'STOP').exists() or (writer.root/'STOP').is_symlink():
                raise RuntimeError('Campaign STOP requested')
            seen=snapshot(writer.root)
            output=sum(v['size'] for k,v in seen['files'].items() if k.startswith(stem+'/'))
            sample=dict(free_bytes=shutil.disk_usage(writer.root).free,output_bytes=output,log_bytes=log_bytes)
            state['observation']=sample
            log_entry=seen['files'].get(stem+'/process/worker.log')
            if log_entry is not None and log_entry['size']!=log_bytes:
                raise RuntimeError('Captured log size changed outside bounded writer')
            if sample['free_bytes']<(required_free_bytes if initial else RESERVE):
                raise RuntimeError('Insufficient disk for remaining allocations' if initial else 'Disk reserve crossed')
            # Reserve room for one terminal report before observing completion.
            if output+DOCUMENT_LIMIT>output_limit_bytes:
                raise RuntimeError('Worker output limit or terminal-report reserve crossed')
            if time.monotonic()-started>=timeout_seconds:raise TimeoutError('Worker wall time limit crossed')

    def save_control():
        nonlocal control_created
        state['elapsed_seconds']=time.monotonic()-started
        if _encoded_size(state)>DOCUMENT_LIMIT:raise ValueError('Process state document cap exceeded')
        writer.publish_json(control,state,replace=control_created)
        control_created=True

    def save_terminal():
        state['elapsed_seconds']=time.monotonic()-started
        # Immutable terminal report includes its own bytes in the final census.
        if state['observation'] is not None:
            base=state['observation']['output_bytes']
            for _ in range(10):
                size=_encoded_size(state)
                if state['observation']['output_bytes']==base+size:break
                state['observation']['output_bytes']=base+size
            else:raise ValueError('Terminal size did not converge')
        if _encoded_size(state)>DOCUMENT_LIMIT:raise ValueError('Terminal process document cap exceeded')
        writer.publish_json(terminal,state)

    # Fail before mutation if any earlier launch/log/report/control attempt exists.
    with writer._serialized():
        seen=snapshot(writer.root)
        if any(k.startswith(stem+'/process/') or k in (control,control+'.tmp') for k in seen['files']):
            raise FileExistsError('Existing process attempt is terminal; no retry')
    # This exclusive launch file is the durable attempt marker, including a
    # preflight/storage failure; a successful prior slot cannot be reused.
    writer.publish_json(stem+'/process/launch.json',dict(state))
    try:
        observe(initial=True);save_control()
        stream=writer.open_append(stem+'/process/worker.log')
        # Recheck after metadata publication, immediately before subprocess creation.
        observe(initial=True)
        process=subprocess.Popen(argv,cwd=cwd,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,bufsize=0,start_new_session=True,
            pass_fds=writer.lock_fds,close_fds=True,shell=False)
        state.update(status='running',pid=process.pid)
        os.set_blocking(process.stdout.fileno(),False)
        selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
        save_control()
        eof=False;last_save=time.monotonic()
        while True:
            observe()
            code=process.poll()
            if code is not None and _group_exists(process.pid):raise RuntimeError('Worker left live descendants')
            if code is not None and eof:break
            events=selector.select(min(poll_seconds,max(0,timeout_seconds-(time.monotonic()-started))))
            for key,_ in events:
                try:data=os.read(key.fd,READ_BYTES)
                except BlockingIOError:continue
                if not data:
                    selector.unregister(key.fileobj);eof=True;continue
                remaining=log_limit_bytes-log_bytes
                retained=data[:remaining]
                if retained:stream.write(retained);log_bytes+=len(retained)
                if len(data)>remaining:
                    state['log_truncated']=True
                    raise RuntimeError('Worker log limit reached; excess pipe bytes not persisted')
            if time.monotonic()-last_save>=poll_seconds:
                save_control();last_save=time.monotonic()
        state['returncode']=process.wait()
        stream.close();stream=None
        observe()
        if state['returncode']!=0:raise RuntimeError('Worker exited with code '+str(state['returncode']))
        state['status']='completed';save_control();observe();save_terminal()
        return state
    except BaseException as exc:
        if process is not None:
            try:_stop(process)
            except BaseException as cleanup_error:state['cleanup_error']=repr(cleanup_error)[:1000]
            state['returncode']=process.returncode
        if stream is not None:
            try:stream.close()
            except BaseException as close_error:state['log_close_error']=repr(close_error)[:1000]
        state.update(status='failed',error=repr(exc)[:1000])
        try:
            with writer._serialized():
                seen=snapshot(writer.root)
                state['observation']=dict(free_bytes=shutil.disk_usage(writer.root).free,
                    output_bytes=sum(v['size'] for k,v in seen['files'].items() if k.startswith(stem+'/')),
                    log_bytes=log_bytes)
        except BaseException as observation_error:
            state['failure_observation_error']=repr(observation_error)[:1000]
            if state['observation'] is not None:state['observation']['log_bytes']=log_bytes
        # If the filesystem/metadata guard itself prevents diagnostics, retain
        # launch/partial/control evidence and propagate the original failure.
        for publish in (save_control,save_terminal):
            try:publish()
            except BaseException as report_error:exc.add_note('Process evidence publication failed: '+repr(report_error)[:1000])
        raise
    finally:
        if selector is not None:selector.close()
        if process is not None and process.stdout is not None:process.stdout.close()
