"""Real parent metadata/audits, mocked scientific child; no MPS or training."""
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_dispatch_v2 as dispatch
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_quality_startup_v2 import setup as first_setup
from test_tokenizer_wide_hybrid_quality_acceptance_v2 import complete


def setup(writer,monkeypatch):
    args=first_setup(writer,monkeypatch);root=writer.root/'evidence'
    original=dispatch.readiness.first.bind_campaign
    def campaign(*a):
        bound=original(*a)
        for choices in bound['requests']['quality'].values():
            for request in choices:
                request['config']=dict(fixture_only=True,data=dict(path=str(root/'meta/corpus.txt')))
        return bound
    monkeypatch.setattr(dispatch.readiness.first,'bind_campaign',campaign)
    monkeypatch.setattr(dispatch.readiness.later,'bind_campaign',campaign)
    monkeypatch.setattr(dispatch.readiness.later,'PROJECTION',dispatch.readiness.first.PROJECTION)
    free=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3))
    for module in (dispatch,dispatch.readiness,dispatch.readiness.later,dispatch.acceptance):
        monkeypatch.setattr(module,'shutil',free)
    calls=[]
    def process(w,stage,slot,argv,cwd,**limits):
        assert w is writer and stage=='quality' and cwd==dispatch.ROOT
        assert limits['timeout_seconds']==7200 and limits['log_limit_bytes']==4*1024**2
        assert not (writer.root/f'control/quality-controller-review-{slot:02d}.json').exists()
        prepare=json.loads((writer.root/f'control/quality-controller-prepare-{slot:02d}.json').read_text())
        assert prepare['status']=='finished'
        startup=dict(path='meta/quality-first-startup.json' if slot==0 else f'meta/quality-lr-startup-{slot:02d}.json',sha256=argv[9])
        ready=dict(path=f'meta/quality-readiness-{slot:02d}.json',sha256=argv[15])
        seal=dispatch._document(root,ready)
        assert limits['required_free_bytes']==seal['required_free_bytes'] and limits['output_limit_bytes']==seal['output_limit_bytes']
        limits['parent_check']();calls.append(slot)
        return complete(writer,monkeypatch,slot,startup,ready)['process']
    monkeypatch.setattr(dispatch,'run_process',process)
    return args,calls


@pytest.mark.parametrize('count',[2,9])
def test_serial_slots_full_prefix_review_and_no_retry(writer,monkeypatch,count):
    args,calls=setup(writer,monkeypatch)
    first=dispatch.run_lr_once(slot=0,**args)
    before=dispatch.acceptance._inventory(writer.root/'evidence')
    with pytest.raises(ValueError,match='cannot be retried'):dispatch.run_lr_once(slot=0,**args)
    assert dispatch.acceptance._inventory(writer.root/'evidence')==before
    second=dispatch.run_lr_once(slot=1,**args)
    saved=dispatch._document(writer.root/'evidence',second)
    assert calls==[0,1] and saved['prior_dispatches']==[first]
    assert len(saved['review']['completed_records'])==2
    assert not saved['review']['next_stage_authorized'] and saved['review']['selection'] is None
    after=dispatch.acceptance._inventory(writer.root/'evidence')
    assert all(after[n]==v for n,v in before.items())
    refs=[first,second]
    for slot in range(2,count):refs.append(dispatch.run_lr_once(slot=slot,**args))
    assert calls==list(range(count))
    assert dispatch._prior(writer,count,args['plan_sha256'],args['projection_ref'],args['pilot_completion_ref'])==refs
    last=dispatch._document(writer.root/'evidence',refs[-1])
    assert len(last['review']['completed_records'])==count and last['review']['selection'] is None
    assert not last['review']['next_stage_authorized']
    assert not (writer.root/f'evidence/quality/{count:02d}').exists()
    # Historical parents are validated against their saved identities, not the
    # current process PID or its potentially different inherited FD numbers.
    monkeypatch.setattr(dispatch,'os',SimpleNamespace(getpid=lambda:1))
    assert dispatch._prior(writer,count,args['plan_sha256'],args['projection_ref'],args['pilot_completion_ref'])==refs


@pytest.mark.parametrize('phase',['prepare','seal','process','review','monitor','publish'])
def test_failure_keeps_evidence_and_permanently_blocks_retry(writer,monkeypatch,phase):
    args,calls=setup(writer,monkeypatch)
    def fail(*a,**kw):raise RuntimeError('synthetic '+phase)
    if phase=='prepare':monkeypatch.setattr(dispatch.readiness.first,'prepare_first_quality_request',fail)
    elif phase=='seal':monkeypatch.setattr(dispatch.readiness,'seal_quality_readiness',fail)
    elif phase=='process':monkeypatch.setattr(dispatch,'run_process',fail)
    elif phase=='review':monkeypatch.setattr(dispatch.acceptance,'review_completed_lr',fail)
    elif phase=='monitor':monkeypatch.setattr(dispatch._ChildMonitor,'check',fail)
    else:
        publish=writer.publish_json
        def publication(name,value,**kw):
            if name=='evidence/meta/quality-dispatch-00.json':fail()
            return publish(name,value,**kw)
        monkeypatch.setattr(writer,'publish_json',publication)
    with pytest.raises(RuntimeError,match='synthetic'):dispatch.run_lr_once(slot=0,**args)
    root=writer.root/'evidence'
    failed=json.loads((root/'meta/quality-dispatch-00-failure.json').read_text())
    assert failed['retry_allowed'] is False and not (root/'meta/quality-dispatch-00.json').exists()
    before=dispatch.acceptance._inventory(root)
    with pytest.raises(ValueError):dispatch.run_lr_once(slot=0,**args)
    assert dispatch.acceptance._inventory(root)==before


@pytest.mark.parametrize('bad',['receipt','prepare_guard','child_guard','review_guard','launch','entry','seal','startup','worker','missing'])
def test_prior_transport_receipts_gate_next_launch(writer,monkeypatch,bad):
    args,calls=setup(writer,monkeypatch);ref=dispatch.run_lr_once(slot=0,**args)
    root=writer.root/'evidence';saved=dispatch._document(root,ref)
    paths={'receipt':root/ref['path'],'launch':root/saved['review']['child']['launch']['path'],
        'entry':writer.root/saved['review']['entry_guard']['path'],
        'seal':root/saved['review']['readiness']['path'],'startup':root/saved['review']['startup']['path'],
        'worker':root/saved['review']['child']['worker']['path']}
    for phase in ('prepare','child','review'):paths[phase+'_guard']=writer.root/saved['guards'][phase]['path']
    if bad=='missing':(root/ref['path']).unlink()
    else:paths[bad].write_text('{}')
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):dispatch.run_lr_once(slot=1,**args)
    assert calls==[0] and not (root/'quality/01').exists()
    assert (root/'meta/quality-dispatch-01-failure.json').exists()


@pytest.mark.parametrize('slot',[True,-1,9,15,23,'0'])
def test_only_lr_slots_before_any_mutation(writer,monkeypatch,slot):
    args,calls=setup(writer,monkeypatch);before=dispatch.acceptance._inventory(writer.root/'evidence')
    with pytest.raises(ValueError):dispatch.run_lr_once(slot=slot,**args)
    assert calls==[] and dispatch.acceptance._inventory(writer.root/'evidence')==before


def test_parent_monitor_rss_failure_is_exception_not_hard_exit(writer,monkeypatch):
    monitor=dispatch._ChildMonitor(writer,0)
    monkeypatch.setattr(dispatch,'observation',lambda p:dict(peak_rss_bytes=4*1024**3+1,free_bytes=64*1024**3))
    with pytest.raises(RuntimeError,match='Parent child-supervision'):monitor.check()
    with pytest.raises(RuntimeError,match='Parent child-supervision'):monitor.finish()
    saved=json.loads((writer.root/'control/quality-controller-child-00.json').read_text())
    assert saved['status']=='failed' and saved['reason'] and saved['limits']['timeout_seconds']==7200


def test_child_success_followed_by_evidence_change_is_terminal(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);process=dispatch.run_process
    def change(*a,**kw):
        value=process(*a,**kw)
        (writer.root/'evidence/quality/00/result.json').write_text('{}')
        return value
    monkeypatch.setattr(dispatch,'run_process',change)
    with pytest.raises(ValueError):dispatch.run_lr_once(slot=0,**args)
    assert calls==[0] and not (writer.root/'evidence/meta/quality-dispatch-00.json').exists()
    assert (writer.root/'evidence/meta/quality-dispatch-00-failure.json').exists()


@pytest.mark.parametrize('key,value',[('timeout_seconds',600),('log_truncated',True),('elapsed_seconds',7200.),
    ('observation',dict(free_bytes=64*1024**3,output_bytes=0,log_bytes=0))])
def test_rewritten_prior_terminal_still_requires_limits_and_census(writer,monkeypatch,key,value):
    args,calls=setup(writer,monkeypatch);ref=dispatch.run_lr_once(slot=0,**args);root=writer.root/'evidence'
    saved=dispatch._document(root,ref);terminal=saved['review']['child']['process'];path=root/terminal['path']
    record=json.loads(path.read_text());record[key]=value;path.write_text(json.dumps(record))
    terminal['sha256']=dispatch.file_digest(path)
    (root/ref['path']).write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='Earlier terminal'):dispatch.run_lr_once(slot=1,**args)
    assert calls==[0] and not (root/'quality/01').exists()
