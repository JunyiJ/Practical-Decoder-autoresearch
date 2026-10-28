"""Synthetic pilot resource observations; no model, corpus, or MPS execution."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from src.research import tokenizer_wide_hybrid_resources as r
from src.research import tokenizer_wide_hybrid_retention_audit as audit
from src.research.tokenizer_wide_hybrid_pilot_config import CONTRACT
from src.research.protocol import digest, file_digest


def observation(_):
    return dict(peak_rss_bytes=1000,mps_driver_bytes=1000,free_bytes=40*1024**3)


def write(path, value):
    path.write_text(json.dumps(value))
    return file_digest(path)


def inputs(root):
    rq=write(root/'request.json',{'synthetic':True})
    rs=write(root/'result.json',dict(status='completed',holdout_consumed=False,request_sha256=rq))
    write(root/'audit.json',{'synthetic_cache_audit':True})
    return digest('plan'),rq,rs


def process():
    return dict(status='completed',returncode=0,error=None,reserve_bytes=8*1024**3,
        timeout_seconds=600,output_limit_bytes=10000,log_limit_bytes=4*1024**2,
        elapsed_seconds=599.99,required_free_bytes=30_631_281_412,
        observation=dict(free_bytes=40*1024**3,output_bytes=5000,log_bytes=100))


@pytest.fixture
def completed(tmp_path,monkeypatch):
    plan,rq,rs=inputs(tmp_path)
    monkeypatch.setattr(r,'observation',observation)
    with r.PilotResourceRecorder(tmp_path,plan,rq) as recorder:
        recorder.mark_success(rs,file_digest(tmp_path/'audit.json'))
    report=json.loads((tmp_path/'resource-audit.json').read_text())
    events=[json.loads(s) for s in (tmp_path/'resource-events.jsonl').read_text().splitlines()]
    return report,events,process(),dict(plan_sha256=plan),rq,rs


def test_fixed_pilot_contract_and_both_reviews(completed):
    report,events,p,context,rq,rs=completed
    r.verify_pilot_events(report,events)
    audit.pilot_resource_review(report,p,context,rq,rs,10000)
    assert report['limits']['timeout_seconds']==CONTRACT['job_seconds']==600
    for key in ('rss_limit_bytes','driver_limit_bytes','reserve_bytes','log_limit_bytes'):
        assert report['limits'][key]==CONTRACT[key]
    assert r.PILOT_MAX_SAMPLES==603


def test_quality_cannot_accept_pilot_and_pilot_cannot_accept_quality(completed):
    report,events,p,context,rq,rs=completed
    with pytest.raises(ValueError):r.verify_events(report,events)
    with pytest.raises(ValueError):audit.resource_review(report,p,context,rq,rs,10000)
    report['kind']=r.QualityResourceRecorder.KIND
    report['limits']=audit.RESOURCE_LIMITS.copy()
    events[0]=r.header(context['plan_sha256'],rq)
    p['timeout_seconds']=7200
    report.pop('cache_audit_sha256');events[-1].pop('cache_audit_sha256')
    r.verify_events(report,events)
    audit.resource_review(report,p,context,rq,rs,10000)
    with pytest.raises(ValueError):r.verify_pilot_events(report,events)
    with pytest.raises(ValueError):audit.pilot_resource_review(report,p,context,rq,rs,10000)


@pytest.mark.parametrize('bad',['wall','rss','driver','disk','reverse','empty','oversample',
    'no_mps','kind','limits','violations','plan','request','result','process_time',
    'process_timeout','process_output','process_log','process_failure'])
def test_pilot_acceptance_rejects_resource_or_provenance_changes(completed,bad):
    report,events,p,context,rq,rs=copy.deepcopy(completed)
    if bad=='wall':report['samples'][-1]['elapsed_seconds']=600.
    elif bad=='rss':report['samples'][-1]['peak_rss_bytes']=4*1024**3
    elif bad=='driver':report['samples'][-1]['mps_driver_bytes']=6*1024**3
    elif bad=='disk':report['samples'][-1]['free_bytes']=8*1024**3-1
    elif bad=='reverse':report['samples'][0]['elapsed_seconds']=599.
    elif bad=='empty':report['samples']=[]
    elif bad=='oversample':report['samples']=[report['samples'][0]]*604
    elif bad=='no_mps':
        for s in report['samples']:s['mps_driver_bytes']=0
    elif bad=='kind':report['kind']=r.QualityResourceRecorder.KIND
    elif bad=='limits':report['limits']['timeout_seconds']=7200
    elif bad=='violations':report['violations']=['ignored failure']
    elif bad in ('plan','request','result'):report[bad+'_sha256']='0'*64
    elif bad=='process_time':p['elapsed_seconds']=600
    elif bad=='process_timeout':p['timeout_seconds']=7200
    elif bad=='process_output':p['observation']['output_bytes']=10001
    elif bad=='process_log':p['observation']['log_bytes']=4*1024**2+1
    elif bad=='process_failure':p['returncode']=90
    with pytest.raises(ValueError):audit.pilot_resource_review(report,p,context,rq,rs,10000)
    if not bad.startswith('process_'):
        with pytest.raises(ValueError):r.verify_pilot_events(report,events)


@pytest.mark.parametrize('bad',['partial','extra','header','terminal'])
def test_pilot_trace_must_match_summary(completed,bad):
    report,events,*_=completed
    if bad=='partial':events.pop()
    elif bad=='extra':events.insert(1,copy.deepcopy(events[1]))
    elif bad=='header':events[0]['limits']['timeout_seconds']=7200
    else:events[-1]['status']='failed'
    with pytest.raises(ValueError):r.verify_pilot_events(report,events)


@pytest.mark.parametrize('bad',['missing_success','request_changed','result_changed','exception'])
def test_failed_pilot_attempt_preserved_and_not_reusable(tmp_path,monkeypatch,bad):
    plan,rq,rs=inputs(tmp_path);monkeypatch.setattr(r,'observation',observation)
    with pytest.raises((ValueError,RuntimeError)):
        with r.PilotResourceRecorder(tmp_path,plan,rq) as rec:
            if bad=='exception':raise RuntimeError('synthetic cache audit failure')
            if bad=='request_changed':write(tmp_path/'request.json',{'changed':True})
            if bad=='result_changed':write(tmp_path/'result.json',{'changed':True})
            if bad!='missing_success':rec.mark_success(rs,file_digest(tmp_path/'audit.json'))
    report=json.loads((tmp_path/'resource-audit.json').read_text())
    assert report['status']=='failed' and report['violations']
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    with pytest.raises(FileExistsError):
        with r.PilotResourceRecorder(tmp_path,plan,rq):pass
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}


def test_pilot_deadline_hard_exit_at_600_with_independent_trace(tmp_path):
    plan,rq,_=inputs(tmp_path)
    code='''
import os,sys
from pathlib import Path
from src.research import tokenizer_wide_hybrid_resources as r
fd=os.open(os.devnull,os.O_WRONLY);os.dup2(fd,1);os.dup2(fd,2)
r.observation=lambda root:dict(peak_rss_bytes=1000,mps_driver_bytes=1000,free_bytes=40*1024**3)
ticks=iter((0.,600.,601.));r.time.monotonic=lambda:next(ticks)
with r.PilotResourceRecorder(Path(sys.argv[1]),sys.argv[2],sys.argv[3]):
    raise AssertionError('Must never enter expired pilot body')
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path),plan,rq],
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=15)
    assert result.returncode==90
    report=json.loads((tmp_path/'resource-audit.json').read_text())
    assert report['status']=='failed' and report['violations']==['wall_time_limit']
    assert report['events']['sha256']==file_digest(tmp_path/'resource-events.jsonl')
    events=[json.loads(line) for line in (tmp_path/'resource-events.jsonl').read_text().splitlines()]
    assert events[0]['limits']['timeout_seconds']==600
    assert events[-1]['reason']=='wall_time_limit'
    assert r.reason(dict(observation(None),elapsed_seconds=600.)) is None


@pytest.mark.parametrize('bad',['missing','changed','symlink'])
def test_cache_audit_must_exist_unchanged_before_pilot_resource_completion(tmp_path,monkeypatch,bad):
    plan,rq,rs=inputs(tmp_path);monkeypatch.setattr(r,'observation',observation)
    path=tmp_path/'audit.json';sha=file_digest(path)
    with pytest.raises(ValueError):
        with r.PilotResourceRecorder(tmp_path,plan,rq) as rec:
            rec.mark_success(rs,sha)
            if bad=='missing':path.unlink()
            elif bad=='changed':write(path,{'changed':True})
            else:
                copy=tmp_path/'copy.json';copy.write_bytes(path.read_bytes());path.unlink();path.symlink_to(copy)
    report=json.loads((tmp_path/'resource-audit.json').read_text())
    assert report['status']=='failed' and report['cache_audit_sha256']==sha
