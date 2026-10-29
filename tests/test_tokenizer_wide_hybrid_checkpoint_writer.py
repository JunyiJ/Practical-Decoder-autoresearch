"""Tiny synthetic serialization and failure injection; no model/corpus/MPS."""
import hashlib
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
import torch

from src.research import tokenizer_wide_hybrid_checkpoint_writer as checkpoints
from src.research import tokenizer_wide_hybrid_metadata as metadata
from src.research import tokenizer_wide_hybrid_resources as resources
from src.research import tokenizer_wide_hybrid_campaign_resources as campaign
from src.research.protocol import digest,file_digest
from test_tokenizer_wide_hybrid_metadata_writer import writer


def producer(writer,cap=1024**2):
    return checkpoints.CheckpointWriter(writer,'quality',0,cap)


@pytest.mark.parametrize('stage,slot',[('pilots',0),('pilots',2),('quality',0),('quality',23)])
def test_torch_stream_roundtrip_raw_and_export(writer,stage,slot):
    out=checkpoints.CheckpointWriter(writer,stage,slot,1024**2)
    tensor=torch.arange(64,dtype=torch.float32)
    payload={'model':{'weight':tensor},'optimizer':{'momentum':tensor*2},
             'ema':{'weight':tensor+1},'rng':torch.random.get_rng_state(),'step':20}
    raw=out.save('raw.pt',payload)
    export=out.save('h200.pt',{'model':payload['ema'],'step':20})
    for receipt in (raw,export):
        path=Path(receipt['path'])
        assert receipt['size']==path.stat().st_size
        assert receipt['sha256']==file_digest(path)
        assert not path.with_suffix('.pt.tmp').exists()
    loaded=torch.load(raw['path'],weights_only=True)
    assert torch.equal(loaded['model']['weight'],tensor)
    assert torch.equal(loaded['optimizer']['momentum'],tensor*2)
    assert torch.equal(loaded['rng'],payload['rng'])
    assert torch.equal(torch.load(export['path'],weights_only=True)['model']['weight'],tensor+1)
    assert metadata.snapshot(writer.root)['checkpoint_logical_bytes']==raw['size']+export['size']


@pytest.mark.parametrize('args',[('pilots',3,1),('quality',24,1),('quality',True,1),
    ('quality',-1,1),('quality','00',1),('inference',0,1),('quality',0,0),
    ('quality',0,-1),('quality',0,True),('quality',0,1.5)])
def test_invalid_contract(writer,args):
    with pytest.raises(ValueError):checkpoints.CheckpointWriter(writer,*args)


@pytest.mark.parametrize('name',['../raw.pt','/raw.pt','checkpoints/raw.pt','raw.pt.tmp',
    'raw.tmp.pt','raw','raw.json','.pt','..pt','x.pt/'])
def test_invalid_destination(writer,name):
    with pytest.raises(ValueError):producer(writer).open(name)


def test_exact_aggregate_cap_and_no_retry(writer):
    out=producer(writer,10)
    with out.open('raw.pt') as stream:
        assert stream.write(b'123456')==6
        assert stream.tell()==6
        raw=stream.publish()
    stream=out.open('h200.pt');stream.write(b'7890')
    with pytest.raises(ValueError,match='allowance'):stream.write(b'x')
    assert stream.closed
    assert metadata.snapshot(writer.root)['checkpoint_logical_bytes']==10
    assert file_digest(Path(raw['path']))==raw['sha256']
    for name in ('raw.pt','h200.pt','replacement.pt'):
        with pytest.raises(FileExistsError):out.open(name)
    with pytest.raises(ValueError):stream.publish()


def test_oversized_single_write_rejected_before_payload(writer):
    out=producer(writer,10);stream=out.open('raw.pt')
    with pytest.raises(ValueError):stream.write(b'x'*11)
    assert (writer.root/out.stem/'raw.pt.tmp').stat().st_size==0


def test_partial_io_failure_preserves_bytes(writer,monkeypatch):
    out=producer(writer);stream=out.open('raw.pt')
    def failing(fd,data):
        os.write(fd,data[:3]);raise OSError('injected IO failure')
    monkeypatch.setattr(checkpoints,'_write',failing)
    with pytest.raises(OSError,match='injected'):stream.write(b'abcdef')
    assert stream.closed
    assert (writer.root/out.stem/'raw.pt.tmp').read_bytes()==b'abc'
    with pytest.raises(FileExistsError):out.open('other.pt')


def test_failed_torch_serialization_keeps_attempt(writer,monkeypatch):
    out=producer(writer)
    def broken(payload,stream):
        stream.write(b'partial');raise RuntimeError('serializer failed')
    monkeypatch.setattr(torch,'save',broken)
    with pytest.raises(RuntimeError):out.save('raw.pt',{})
    assert (writer.root/out.stem/'raw.pt.tmp').read_bytes()==b'partial'
    assert not (writer.root/out.stem/'raw.pt').exists()


@pytest.mark.parametrize('operation',['write','flush','publish'])
def test_identity_mutation_closes_stream(writer,operation):
    out=producer(writer);stream=out.open('raw.pt');stream.write(b'a')
    path=writer.root/out.stem/'raw.pt.tmp';path.unlink();path.write_bytes(b'a')
    with pytest.raises(ValueError,match='identity'):
        getattr(stream,operation)(*([b'b'] if operation=='write' else []))
    assert stream.closed


@pytest.mark.parametrize('operation',['write','flush','publish'])
def test_stop_during_output(writer,operation):
    out=producer(writer);stream=out.open('raw.pt');stream.write(b'a')
    writer.publish_json('STOP',{'stop':True})
    with pytest.raises(ValueError,match='STOP'):
        getattr(stream,operation)(*([b'b'] if operation=='write' else []))
    assert stream.closed
    assert (writer.root/out.stem/'raw.pt.tmp').read_bytes()==b'a'


def test_free_space_rechecked_between_chunks(writer,monkeypatch):
    out=producer(writer);stream=out.open('raw.pt')
    monkeypatch.setattr(checkpoints,'CHUNK_BYTES',4)
    real=checkpoints._write
    def drain(fd,data):
        real(fd,data)
        monkeypatch.setattr(checkpoints.shutil,'disk_usage',lambda p:SimpleNamespace(free=checkpoints.RESERVE))
    monkeypatch.setattr(checkpoints,'_write',drain)
    with pytest.raises(ValueError,match='reserve'):stream.write(b'abcdefgh')
    assert (writer.root/out.stem/'raw.pt.tmp').read_bytes()==b'abcd'


def test_publication_never_overwrites_final(writer):
    out=producer(writer);stream=out.open('raw.pt');stream.write(b'new')
    final=writer.root/out.stem/'raw.pt';final.write_bytes(b'keep')
    with pytest.raises(FileExistsError):stream.publish()
    assert final.read_bytes()==b'keep'
    assert final.with_suffix('.pt.tmp').read_bytes()==b'new'


def test_invalid_buffer_and_closed_stream_rejected(writer):
    stream=producer(writer).open('raw.pt')
    with pytest.raises(TypeError):stream.write('not bytes')
    assert stream.closed
    for method,args in [('write',(b'a',)),('tell',()),('flush',()),('publish',())]:
        with pytest.raises(ValueError):getattr(stream,method)(*args)


def test_real_torch_writes_interleave_with_resource_recorder(writer,monkeypatch):
    monkeypatch.setattr(resources,'observation',lambda root:dict(
        peak_rss_bytes=1000,mps_driver_bytes=2000,free_bytes=40*1024**3))
    stem='evidence/quality/00'
    rq=writer.publish_json(stem+'/request.json',{'synthetic':True})
    rs=writer.publish_json(stem+'/result.json',dict(status='completed',holdout_consumed=False,
                                                 request_sha256=file_digest(rq)))
    out=producer(writer)
    with campaign.CampaignQualityResourceRecorder(writer,0,digest('plan'),file_digest(rq)) as recorder:
        with ThreadPoolExecutor(2) as pool:
            saves=pool.submit(lambda:[out.save(f'tiny-{i}.pt',{'x':torch.arange(16)}) for i in range(10)])
            samples=pool.submit(lambda:[recorder._sample() for _ in range(20)])
            receipts=saves.result();samples.result()
        recorder.mark_success(file_digest(rs))
    assert metadata.snapshot(writer.root)['checkpoint_logical_bytes']==sum(r['size'] for r in receipts)
    assert all(file_digest(Path(r['path']))==r['sha256'] for r in receipts)


def test_chunked_digest_and_namespace_lock_released(writer,monkeypatch):
    out=producer(writer);stream=out.open('raw.pt')
    monkeypatch.setattr(checkpoints,'CHUNK_BYTES',4)
    real=checkpoints._write;calls=[]
    def writing(fd,data):
        calls.append(len(data));real(fd,data)
    monkeypatch.setattr(checkpoints,'_write',writing)
    stream.write(b'0123456789');stream.flush();receipt=stream.publish()
    assert calls==[4,4,2]
    assert receipt['sha256']==hashlib.sha256(b'0123456789').hexdigest()
    writer.publish_json('control/after.json',{'done':True})


def test_checkpoint_handoff_admits_supervisor_and_resource_publishers(writer,monkeypatch):
    """Both competing publishers must finish while the checkpoint yields."""
    import threading
    monkeypatch.setattr(checkpoints,'CHUNK_BYTES',4)
    wake=threading.Event();handoffs=[]
    stream=producer(writer).open('raw.pt')
    events=writer.open_append('evidence/quality/00/resource-events.jsonl')
    def supervisor():
        assert wake.wait(2)
        writer.publish_json('control/supervisor.json',{'sample':1})
    def resource():
        assert wake.wait(2)
        events.write(b'{"sample":1}\n')
    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks=[pool.submit(supervisor),pool.submit(resource)]
        def handoff(seconds):
            assert seconds==.02
            handoffs.append(seconds);wake.set()
            for task in tasks:task.result(timeout=2)
        monkeypatch.setattr(checkpoints,'_handoff',handoff)
        stream.write(b'0123456789');receipt=stream.publish()
    events.close()
    assert len(handoffs)==2
    assert Path(receipt['path']).read_bytes()==b'0123456789'
    assert receipt['sha256']==hashlib.sha256(b'0123456789').hexdigest()
    assert json_load(writer.root/'control/supervisor.json')=={'sample':1}


def json_load(path):
    import json
    return json.loads(path.read_text())


def test_small_callbacks_accumulate_handoff_without_changing_bytes(writer,monkeypatch):
    monkeypatch.setattr(checkpoints,'CHUNK_BYTES',4)
    pauses=[];monkeypatch.setattr(checkpoints,'_handoff',pauses.append)
    stream=producer(writer).open('raw.pt')
    stream.write(b'abc');assert pauses==[]
    stream.write(b'def');assert pauses==[.02]
    stream.write(b'gh');assert pauses==[.02,.02]
    receipt=stream.publish()
    assert Path(receipt['path']).read_bytes()==b'abcdefgh'
    assert receipt['sha256']==hashlib.sha256(b'abcdefgh').hexdigest()


@pytest.mark.parametrize('buffer',[bytearray(b'abc'),memoryview(b'abc'),memoryview(bytearray(b'abc'))])
def test_buffer_protocol_support(writer,buffer):
    stream=producer(writer).open('raw.pt')
    assert stream.write(buffer)==3
    receipt=stream.publish()
    assert Path(receipt['path']).read_bytes()==b'abc'


def test_strided_buffer_rejected(writer):
    stream=producer(writer).open('raw.pt')
    with pytest.raises(TypeError):stream.write(memoryview(b'abcd')[::2])
    assert stream.closed


def test_real_torch_archive_byte_limit_failure_retains_partial(writer):
    out=producer(writer,100)
    with pytest.raises((ValueError,RuntimeError)):
        out.save('raw.pt',{'x':torch.arange(64)})
    partial=writer.root/out.stem/'raw.pt.tmp'
    assert partial.exists() and partial.stat().st_size<=100
    assert not partial.with_suffix('').exists()
    with pytest.raises(FileExistsError):out.save('raw.pt',{})


def test_internal_mutation_cannot_change_written_chunk_digest(writer,monkeypatch):
    out=producer(writer);stream=out.open('raw.pt');data=bytearray(b'abcd')
    real=checkpoints._write
    def mutate(fd,piece):
        data[:]=b'zzzz';real(fd,piece)
    monkeypatch.setattr(checkpoints,'_write',mutate)
    stream.write(data);receipt=stream.publish()
    assert Path(receipt['path']).read_bytes()==b'abcd'
    assert receipt['sha256']==hashlib.sha256(b'abcd').hexdigest()


@pytest.mark.parametrize('operation',['write','flush','publish'])
def test_same_size_mutation_rejected(writer,operation):
    out=producer(writer);stream=out.open('raw.pt');stream.write(b'abc')
    path=writer.root/out.stem/'raw.pt.tmp';path.write_bytes(b'xyz')
    st=path.stat();os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns+1000000))
    with pytest.raises(ValueError,match='identity'):
        getattr(stream,operation)(*([b'b'] if operation=='write' else []))
    assert stream.closed and not path.with_suffix('').exists()


def test_external_alias_rejected(writer,tmp_path):
    out=producer(writer);stream=out.open('raw.pt');stream.write(b'a')
    path=writer.root/out.stem/'raw.pt.tmp';os.link(path,tmp_path/'external.pt')
    with pytest.raises(ValueError):stream.publish()
    assert stream.closed and path.exists()


def test_exact_disk_reserve_boundary(writer,monkeypatch):
    stream=producer(writer).open('raw.pt')
    monkeypatch.setattr(checkpoints.shutil,'disk_usage',lambda p:SimpleNamespace(free=checkpoints.RESERVE+3))
    assert stream.write(b'abc')==3
    monkeypatch.setattr(checkpoints.shutil,'disk_usage',lambda p:SimpleNamespace(free=checkpoints.RESERVE-1))
    with pytest.raises(ValueError,match='reserve'):stream.publish()
    assert stream.closed


def test_existing_completed_weight_cannot_be_reentered(writer):
    out=producer(writer)
    with out.open('raw.pt') as stream:
        stream.write(b'original');receipt=stream.publish()
    with pytest.raises(FileExistsError):out.open('raw.pt')
    assert Path(receipt['path']).read_bytes()==b'original'


def test_file_count_limit_at_publication(writer,monkeypatch):
    out=producer(writer);stream=out.open('raw.pt');stream.write(b'a')
    monkeypatch.setattr(checkpoints,'MAX_FILES',len(metadata.snapshot(writer.root)['files']))
    with pytest.raises(ValueError,match='file-count'):stream.publish()
    assert stream.closed and (writer.root/out.stem/'raw.pt.tmp').exists()


def test_final_publication_failure_preserves_both_names(writer,monkeypatch):
    out=producer(writer);stream=out.open('raw.pt');stream.write(b'a')
    def fail(path):raise OSError('injected directory sync failure')
    monkeypatch.setattr(checkpoints,'_sync_directory',fail)
    with pytest.raises(OSError,match='injected'):stream.publish()
    final=writer.root/out.stem/'raw.pt'
    assert final.read_bytes()==b'a' and final.with_suffix('.pt.tmp').read_bytes()==b'a'
    assert stream.closed
    with pytest.raises(ValueError,match='alias'):metadata.snapshot(writer.root)
    with pytest.raises(ValueError):out.open('new.pt')
