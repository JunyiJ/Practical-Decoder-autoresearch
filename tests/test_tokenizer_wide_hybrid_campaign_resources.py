"""Synthetic observations and disposable files; never initializes MPS."""
import fcntl
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_campaign_resources as campaign
from src.research import tokenizer_wide_hybrid_resources as resources
from src.research import tokenizer_wide_hybrid_metadata as metadata
from src.research import tokenizer_wide_hybrid_metadata_writer as writing
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research.protocol import digest,file_digest


@pytest.fixture
def writer(tmp_path,monkeypatch):
    root=tmp_path/'campaign';root.mkdir();gpath=tmp_path/'global.lock';gpath.touch();(root/'.lock').touch()
    monkeypatch.setattr(cleanup,'GLOBAL_LOCK',gpath)
    monkeypatch.setattr(writing.shutil,'disk_usage',lambda p:SimpleNamespace(free=40*1024**3))
    monkeypatch.setattr(resources,'observation',lambda root:dict(peak_rss_bytes=1000,mps_driver_bytes=2000,free_bytes=40*1024**3))
    with gpath.open('rb') as g,(root/'.lock').open('rb') as c:
        fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(c,fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield writing.MetadataWriter(root,(g.fileno(),c.fileno()))


def setup(writer,stage,slot=0):
    stem=f'evidence/{stage}/{slot:02d}'
    rq=writer.publish_json(stem+'/request.json',{'synthetic':True})
    rs=writer.publish_json(stem+'/result.json',dict(status='completed',holdout_consumed=False,request_sha256=file_digest(rq)))
    cache=writer.publish_json(stem+'/audit.json',{'synthetic_cache':True})
    return stem,digest('plan'),file_digest(rq),file_digest(rs),file_digest(cache)


def complete(writer,stage,slot=0):
    stem,plan,rq,rs,cache=setup(writer,stage,slot)
    cls=campaign.CampaignPilotResourceRecorder if stage=='pilots' else campaign.CampaignQualityResourceRecorder
    with cls(writer,slot,plan,rq) as recorder:
        if stage=='pilots':recorder.mark_success(rs,cache)
        else:recorder.mark_success(rs)
    report=json.loads((writer.root/stem/'resource-audit.json').read_text())
    events=[json.loads(line) for line in (writer.root/stem/'resource-events.jsonl').read_text().splitlines()]
    return stem,report,events


@pytest.mark.parametrize('stage,slot',[('pilots',0),('pilots',2),('quality',0),('quality',23)])
def test_bounded_recording_and_report_agree(writer,stage,slot):
    stem,report,events=complete(writer,stage,slot)
    verify=resources.verify_pilot_events if stage=='pilots' else resources.verify_events
    verify(report,events)
    assert report['limits']['timeout_seconds']==(600 if stage=='pilots' else 7200)
    assert report['events']['sha256']==file_digest(writer.root/stem/'resource-events.jsonl')
    census=metadata.snapshot(writer.root)
    assert census['groups'][stem+'/events']==(writer.root/stem/'resource-events.jsonl').stat().st_size
    assert census['groups'][stem+'/resources']==(writer.root/stem/'resource-audit.json').stat().st_size


@pytest.mark.parametrize('cls,slot',[(campaign.CampaignPilotResourceRecorder,3),
    (campaign.CampaignQualityResourceRecorder,24),(campaign.CampaignPilotResourceRecorder,True),
    (campaign.CampaignQualityResourceRecorder,-1),(campaign.CampaignQualityResourceRecorder,'00')])
def test_invalid_slot_before_output(writer,cls,slot):
    before=set(writer.root.rglob('*'))
    with pytest.raises(ValueError):cls(writer,slot,digest('plan'),digest('request'))
    assert set(writer.root.rglob('*'))==before


def test_actual_bounded_writer_required():
    with pytest.raises(ValueError):campaign.CampaignPilotResourceRecorder(object(),0,digest('plan'),digest('request'))


@pytest.mark.parametrize('stage',['pilots','quality'])
def test_completed_or_partial_attempt_cannot_be_reentered(writer,stage):
    stem,_,_=complete(writer,stage)
    before={str(p):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()}
    cls=campaign.CampaignPilotResourceRecorder if stage=='pilots' else campaign.CampaignQualityResourceRecorder
    with pytest.raises(FileExistsError):
        with cls(writer,0,digest('plan'),file_digest(writer.root/stem/'request.json')):pass
    assert before=={str(p):p.read_bytes() for p in writer.root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('stage',['pilots','quality'])
def test_worker_exception_is_preserved_in_bounded_failed_report(writer,stage):
    stem,plan,rq,_,_=setup(writer,stage)
    cls=campaign.CampaignPilotResourceRecorder if stage=='pilots' else campaign.CampaignQualityResourceRecorder
    with pytest.raises(RuntimeError,match='synthetic'):
        with cls(writer,0,plan,rq):raise RuntimeError('synthetic audit failure')
    report=json.loads((writer.root/stem/'resource-audit.json').read_text())
    assert report['status']=='failed' and 'synthetic' in report['violations'][0]
    assert report['events']['sha256']==file_digest(writer.root/stem/'resource-events.jsonl')


def test_resource_report_uses_atomic_group_overlap_limit(writer,monkeypatch):
    stem,plan,rq,rs,cache=setup(writer,'pilots')
    # Reserve all but32 bytes of the resource group with a retained partial
    # producer artifact. This must never be silently overwritten or discounted.
    tmp=writer.root/stem/'resource-audit.json.tmp'
    with tmp.open('wb') as out:out.truncate(2*1024**2-32)
    recorder=campaign.CampaignPilotResourceRecorder(writer,0,plan,rq)
    # Avoid os._exit only in this in-process synthetic failure assertion.
    monkeypatch.setattr(recorder,'_fatal',lambda status,problem,code: (_ for _ in ()).throw(RuntimeError(str(problem))))
    with pytest.raises(RuntimeError):
        with recorder:recorder.mark_success(rs,cache)
    assert tmp.stat().st_size==2*1024**2-32
    assert not (writer.root/stem/'resource-audit.json').exists()
    assert (writer.root/stem/'resource-events.jsonl').exists()


def test_cooperating_checkpoint_create_publish_during_metadata_recording(writer):
    stem,plan,rq,rs,cache=setup(writer,'pilots')
    directory=writer.root/stem/'checkpoints'
    observed=[]
    def checkpoint_lifecycle():
        # Tiny synthetic bytes emulate namespace transitions only. Real weight
        # creation/output-cap and future-campaign gates are still separate.
        for i in range(20):
            with writer._serialized():
                directory.mkdir(exist_ok=True)
                tmp=directory/f'fixture{i}.pt.tmp';tmp.write_bytes(b'x'*20)
                observed.append(metadata.snapshot(writer.root)['checkpoint_logical_bytes'])
            with writer._serialized():
                os.replace(tmp,directory/f'fixture{i}.pt')
                observed.append(metadata.snapshot(writer.root)['checkpoint_logical_bytes'])
    with campaign.CampaignPilotResourceRecorder(writer,0,plan,rq) as recorder:
        with ThreadPoolExecutor(max_workers=1) as pool:
            worker=pool.submit(checkpoint_lifecycle)
            for _ in range(20):recorder._sample()
            worker.result(timeout=15)
        recorder.mark_success(rs,cache)
    assert observed==[20*(i+1) for i in range(20) for _ in range(2)]
    census=metadata.snapshot(writer.root)
    assert census['checkpoint_logical_bytes']==400 and len(list(directory.glob('*.pt')))==20
    assert not list(directory.glob('*.tmp'))
    report=json.loads((writer.root/stem/'resource-audit.json').read_text())
    assert report['status']=='completed' and len(report['samples'])>=22


def test_uncooperating_disappearing_checkpoint_fails_closed(writer,monkeypatch):
    path=writer.root/'evidence/pilots/00/checkpoints/transient.pt.tmp';path.parent.mkdir(parents=True);path.write_bytes(b'x')
    real=Path.stat
    def racing(p,*args,**kwargs):
        if p==path:raise FileNotFoundError('synthetic concurrent uncoordinated publication')
        return real(p,*args,**kwargs)
    monkeypatch.setattr(Path,'stat',racing)
    with pytest.raises(FileNotFoundError):writer.publish_json('control/state.json',{})
    assert not (writer.root/'control/state.json').exists()


@pytest.mark.parametrize('failure',['time','driver'])
def test_bounded_adapter_hard_failure_retains_diagnostics_under_captured_stdio(writer,failure):
    import subprocess
    import sys
    stem,plan,rq,_,_=setup(writer,'pilots')
    code='''
import os,sys
from pathlib import Path
from types import SimpleNamespace
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research import tokenizer_wide_hybrid_resources as resources
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_campaign_resources import CampaignPilotResourceRecorder
cleanup.GLOBAL_LOCK=Path(sys.argv[2])
w=MetadataWriter(Path(sys.argv[1]),tuple(map(int,sys.argv[3:5])))
fd=os.open(os.devnull,os.O_WRONLY);os.dup2(fd,1);os.dup2(fd,2)
resources.observation=lambda root:dict(peak_rss_bytes=1000,
    mps_driver_bytes=6*1024**3 if sys.argv[7]=='driver' else 2000,free_bytes=40*1024**3)
if sys.argv[7]=='time':
    ticks=iter((0.,600.,601.));resources.time=SimpleNamespace(monotonic=lambda:next(ticks))
with CampaignPilotResourceRecorder(w,0,sys.argv[5],sys.argv[6]):
    raise AssertionError('Expired/over-limit pilot body must never execute')
'''
    result=subprocess.run([sys.executable,'-c',code,str(writer.root),str(cleanup.GLOBAL_LOCK),
        *map(str,writer.lock_fds),plan,rq,failure],pass_fds=writer.lock_fds,
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
    assert result.returncode==90,result.stderr
    report=json.loads((writer.root/stem/'resource-audit.json').read_text())
    assert report['status']=='failed'
    assert report['violations']==['wall_time_limit' if failure=='time' else 'driver_limit']
    assert report['events']['sha256']==file_digest(writer.root/stem/'resource-events.jsonl')
    assert metadata.snapshot(writer.root)['groups'][stem+'/resources']>0
