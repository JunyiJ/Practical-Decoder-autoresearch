"""Synthetic observations and tiny child processes; never initializes MPS."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.research import tokenizer_wide_hybrid_resources as recording
from src.research import tokenizer_wide_hybrid_retention_audit as audit
from src.research.protocol import digest, file_digest


def sample(_):
    return dict(peak_rss_bytes=1000,mps_driver_bytes=1000,free_bytes=40*1024**3)


def setup(tmp_path):
    request=tmp_path/'request.json';request.write_text('{}')
    rqsha=file_digest(request)
    result=tmp_path/'result.json'
    result.write_text(json.dumps(dict(status='completed',holdout_consumed=False,request_sha256=rqsha)))
    return digest('plan'),rqsha,file_digest(result)


def run(tmp_path,monkeypatch):
    plan,rqsha,rssha=setup(tmp_path)
    monkeypatch.setattr(recording,'observation',sample)
    with recording.QualityResourceRecorder(tmp_path,plan,rqsha) as rec:
        rec.mark_success(rssha)
    return plan,rqsha,rssha,json.loads((tmp_path/'resource-audit.json').read_text())


def test_success_matches_resource_gate_and_durable_trace(tmp_path,monkeypatch):
    plan,rqsha,rssha,report=run(tmp_path,monkeypatch)
    events=[json.loads(line) for line in (tmp_path/'resource-events.jsonl').read_text().splitlines()]
    recording.verify_events(report,events)
    assert report['events']['sha256']==file_digest(tmp_path/'resource-events.jsonl')
    assert len(report['samples'])>=2
    process=dict(status='completed',returncode=0,error=None,reserve_bytes=8*1024**3,
        timeout_seconds=7200,output_limit_bytes=10000,log_limit_bytes=4*1024**2,
        elapsed_seconds=1,required_free_bytes=30_631_281_412,
        observation=dict(free_bytes=40*1024**3,output_bytes=5000,log_bytes=10))
    audit.resource_review(report,process,dict(plan_sha256=plan),rqsha,rssha,10000)
    assert not rec_running()


def rec_running():
    import threading
    return any(t.name=='wide-hybrid-resource-recorder' and t.is_alive() for t in threading.enumerate())


@pytest.mark.parametrize('mutation',['missing_success','changed_request','changed_result','bad_result','exception','no_mps'])
def test_incomplete_attempt_is_preserved_without_success(tmp_path,monkeypatch,mutation):
    plan,rqsha,rssha=setup(tmp_path)
    monkeypatch.setattr(recording,'observation',sample if mutation!='no_mps' else lambda root:dict(sample(root),mps_driver_bytes=0))
    with pytest.raises((ValueError,RuntimeError)):
        with recording.QualityResourceRecorder(tmp_path,plan,rqsha) as rec:
            if mutation=='exception': raise RuntimeError('synthetic worker failure')
            if mutation=='changed_request': (tmp_path/'request.json').write_text('changed')
            if mutation=='changed_result': (tmp_path/'result.json').write_text('changed')
            if mutation=='bad_result':
                (tmp_path/'result.json').write_text(json.dumps(dict(status='failed',holdout_consumed=False,request_sha256=rqsha)))
                rssha=file_digest(tmp_path/'result.json')
            if mutation!='missing_success': rec.mark_success(rssha)
    report=json.loads((tmp_path/'resource-audit.json').read_text())
    assert report['status']=='failed' and report['violations']
    assert (tmp_path/'resource-events.jsonl').exists()
    assert not rec_running()


def test_existing_attempt_cannot_be_overwritten(tmp_path,monkeypatch):
    plan,rqsha,_,_=run(tmp_path,monkeypatch)
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        with recording.QualityResourceRecorder(tmp_path,plan,rqsha): pass
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize('which',['rss','driver','late_driver','disk','time','observer','sample_cap','event_cap'])
def test_hard_failure_preserves_event_log_under_captured_output(tmp_path,which):
    plan,rqsha,_=setup(tmp_path)
    code='''
import os,sys,time
from pathlib import Path
from src.research import tokenizer_wide_hybrid_resources as r
capture=os.open(os.devnull,os.O_WRONLY);os.dup2(capture,1);os.dup2(capture,2)
kind=sys.argv[4]
observations=0
def observe(root):
    global observations
    observations+=1
    if kind=='observer': raise RuntimeError('synthetic observer failure')
    return dict(peak_rss_bytes=4*1024**3 if kind=='rss' else 1000,
        mps_driver_bytes=6*1024**3 if kind=='driver' or kind=='late_driver' and observations>1 else 1000,
        free_bytes=8*1024**3-1 if kind=='disk' else 40*1024**3)
r.observation=observe
if kind=='time':
    ticks=iter((0.,7200.,7201.,7202.))
    r.time.monotonic=lambda:next(ticks)
if kind=='sample_cap':r.MAX_SAMPLES=0
if kind=='event_cap':r.EVENT_LIMIT=1024
with r.QualityResourceRecorder(Path(sys.argv[1]),sys.argv[2],sys.argv[3]):
    time.sleep(2)
'''
    child=subprocess.run([sys.executable,'-c',code,str(tmp_path),plan,rqsha,which],
        cwd=Path(__file__).resolve().parents[1],stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
    assert child.returncode==(91 if which in ('observer','sample_cap','event_cap') else 90)
    report=json.loads((tmp_path/'resource-audit.json').read_text())
    assert report['status'] in ('failed','guard_error') and report['violations']
    events=[json.loads(line) for line in (tmp_path/'resource-events.jsonl').read_text().splitlines()]
    assert events[-1]['status']==report['status']
    assert report['events']['sha256']==file_digest(tmp_path/'resource-events.jsonl')


@pytest.mark.parametrize('mutation',['partial','extra','sample','header','terminal'])
def test_trace_summary_mismatch_rejected(tmp_path,monkeypatch,mutation):
    _,_,_,report=run(tmp_path,monkeypatch)
    events=[json.loads(line) for line in (tmp_path/'resource-events.jsonl').read_text().splitlines()]
    if mutation=='partial':events.pop()
    elif mutation=='extra':events.insert(1,copy.deepcopy(events[1]))
    elif mutation=='sample':events[1]['peak_rss_bytes']+=1
    elif mutation=='header':events[0]['request_sha256']='0'*64
    else:events[-1]['status']='failed'
    with pytest.raises(ValueError):recording.verify_events(report,events)


def test_report_and_trace_bounds_fit_fixed_sample_budget():
    # Maximum allowed byte observations and near-timeout sample timestamps.
    row=dict(elapsed_seconds=7199.999999,peak_rss_bytes=2**63-1,mps_driver_bytes=2**63-1,free_bytes=2**63-1)
    samples=[row]*recording.MAX_SAMPLES
    report=dict(kind='wide_hybrid_trial_resources_v2',status='completed',plan_sha256='0'*64,
        request_sha256='1'*64,result_sha256='2'*64,limits=audit.RESOURCE_LIMITS,violations=[],
        samples=samples,events=dict(path='resource-events.jsonl',sha256='3'*64))
    assert len((json.dumps(report,sort_keys=True,indent=2)+'\n').encode())<recording.REPORT_LIMIT
    events=[recording.header('0'*64,'1'*64)]+[dict(event='sample',**s) for s in samples]
    assert sum(len((json.dumps(e,sort_keys=True,separators=(',',':'))+'\n').encode()) for e in events)<recording.EVENT_LIMIT-4096
