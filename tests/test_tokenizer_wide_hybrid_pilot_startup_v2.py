"""Synthetic idle-to-request transitions, real locks/storage/audits, no workers."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_pilot_startup_v2 as start
from src.research.protocol import file_digest,digest
from src.research.tokenizer_wide_hybrid_pilot_config import specs
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_pilot_audit_v2 import fixture,write


def setup(writer,monkeypatch,slot):
    root=writer.root/'evidence';args=fixture(root,count=slot)
    requests=list(args['requests'])
    for i in range(slot,3):
        requests.append(dict(spec=specs()[i],config={'data':{'path':str(root/'meta/corpus.txt')},'fixture_only':True},
                             result_path=str(root/f'pilots/{i:02d}/result.json')))
    plan={k:args[k] for k in ('numerical_ref','tokenizer','training_token_sha256')}
    plan_ref=write(root,'meta/plan.json',plan)
    source=write(root,'meta/source.txt',b'synthetic source identity')
    frozen={str(root/ref['path']):ref['sha256'] for ref in (plan_ref,args['projection_ref'],args['numerical_ref'],source)}
    bound=dict(context=args['context'],requests={'pilots':requests},files=frozen)
    def campaign(path,sha):
        assert path==root/'meta/plan.json'
        if sha!=plan_ref['sha256']:raise ValueError('Frozen plan differs')
        for name,value in frozen.items():
            if file_digest(start.Path(name))!=value:raise ValueError('Frozen file changed')
        return copy.deepcopy(bound)
    monkeypatch.setattr(start,'bind_campaign',campaign)
    monkeypatch.setattr(start,'PROJECTION',root/args['projection_ref']['path'])
    return dict(writer=writer,slot=slot,plan_sha256=plan_ref['sha256'],projection_ref=args['projection_ref'])


def load(writer,ref):return json.loads((writer.root/'evidence'/ref['path']).read_text())


def failure(writer,slot):return writer.root/f'evidence/meta/pilot-startup-{slot:02d}-failure.json'


@pytest.mark.parametrize('slot',[0,1,2])
def test_full_idle_review_before_request_and_read_only_predispatch(writer,monkeypatch,slot):
    args=setup(writer,monkeypatch,slot);original=start.bind_storage;calls=[]
    def gate(*a,**kw):
        assert not (writer.root/f'evidence/pilots/{slot:02d}').exists()
        assert len(kw['pilots'])==slot and len(kw['checkpoint_pairs'])==slot
        calls.append(slot);return original(*a,**kw)
    monkeypatch.setattr(start,'bind_storage',gate)
    ref=start.prepare_pilot_request(**args);report=load(writer,ref)
    assert calls==[slot] and not report['dispatch_authorized'] and not report['deletion_authorized']
    assert report['storage']['required_free_bytes']>30_000_000_000
    before={str(p):file_digest(p) for p in writer.root.rglob('*') if p.is_file()}
    verified=start.verify_before_process(writer,slot,ref)
    assert verified['required_free_bytes']==report['storage']['required_free_bytes']
    assert verified['startup_transition_verified'] and not verified['dispatch_authorized']
    assert before=={str(p):file_digest(p) for p in writer.root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError,match='cannot be retried'):start.prepare_pilot_request(**args)


@pytest.mark.parametrize('offset',[0,-1])
def test_whole_panel_threshold_not_just_next_pilot(writer,monkeypatch,offset):
    args=setup(writer,monkeypatch,0);needed=30_631_281_412
    monkeypatch.setattr(start.shutil,'disk_usage',lambda p:SimpleNamespace(free=needed+offset))
    if offset==0:
        ref=start.prepare_pilot_request(**args)
        assert start.verify_before_process(writer,0,ref)['required_free_bytes']==needed
    else:
        with pytest.raises(ValueError,match='remaining panel does not fit'):start.prepare_pilot_request(**args)
        assert not (writer.root/'evidence/pilots/00').exists()
        assert json.loads(failure(writer,0).read_text())['retry_allowed'] is False
        monkeypatch.setattr(start.shutil,'disk_usage',lambda p:SimpleNamespace(free=64*1024**3))
        with pytest.raises(ValueError,match='cannot be retried'):start.prepare_pilot_request(**args)


@pytest.mark.parametrize('bad',['weight','audit','numeric','later_slot','failed_worker','pending_slot','projection','source','stop'])
def test_no_request_without_complete_unchanged_prefix(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch,1);root=writer.root/'evidence'
    if bad=='weight':(root/'pilots/00/checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='audit':(root/'pilots/00/audit.json').unlink()
    elif bad=='numeric':(root/'meta/numerical.json').write_text('{}')
    elif bad=='later_slot':write(root,'pilots/02/request.json',{})
    elif bad=='failed_worker':write(root,'pilots/00/worker-failure.json',{'status':'failed'})
    elif bad=='pending_slot':write(root,'pilots/01/request.json',{})
    elif bad=='projection':args['projection_ref']={'path':'meta/projection.json','sha256':'0'*64}
    elif bad=='source':(root/'meta/source.txt').write_text('changed')
    else:(writer.root/'STOP').write_text('stop')
    with pytest.raises((ValueError,FileNotFoundError)):start.prepare_pilot_request(**args)
    if bad!='pending_slot':assert not (root/'pilots/01/request.json').exists()


@pytest.mark.parametrize('bad',['needed','occupied','logical','projection_state','completed','ce','pair','claims',
                                'inventory','context','projection_ref','attempt','request'])
def test_changed_receipt_semantics_rejected_even_with_updated_reference(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch,1);ref=start.prepare_pilot_request(**args);report=load(writer,ref)
    if bad=='needed':report['storage']['required_free_bytes']=8*1024**3
    elif bad=='occupied':report['storage']['occupied_checkpoint_capacity_bytes']+=1
    elif bad=='logical':report['storage']['checkpoint_logical_bytes']+=1
    elif bad=='projection_state':report['storage']['projection']['remaining_peak_bytes']=0
    elif bad=='completed':report['completed_pilots']=[]
    elif bad=='ce':report['completed_pilots'][0]['ce']=4.
    elif bad=='pair':report['checkpoint_pairs']['pilot/00']['raw']['size']+=1
    elif bad=='claims':report['dispatch_authorized']=True
    elif bad=='inventory':report['inventory'].pop('pilots/00/metrics.jsonl')
    elif bad=='context':report['context']['source_sha256']='0'*64
    elif bad=='projection_ref':report['projection_ref']['sha256']='0'*64
    elif bad=='attempt':report['attempt']['sha256']='0'*64
    else:report['request']['sha256']='0'*64
    ref=write(writer.root/'evidence',ref['path'],report)
    with pytest.raises((ValueError,KeyError)):start.verify_before_process(writer,1,ref)
    assert json.loads(failure(writer,1).read_text())['phase']=='predispatch'
    with pytest.raises(ValueError,match='Failed or partial'):start.verify_before_process(writer,1,ref)


@pytest.mark.parametrize('bad',['source','weight','new_file','process_attempt','stop','space','partial','hash'])
def test_predispatch_change_is_persistent_failure(writer,monkeypatch,bad):
    args=setup(writer,monkeypatch,1);ref=start.prepare_pilot_request(**args);root=writer.root/'evidence'
    if bad=='source':(root/'meta/source.txt').write_text('changed')
    elif bad=='weight':(root/'pilots/00/checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='new_file':write(root,'meta/extra.json',{})
    elif bad=='process_attempt':write(root,'pilots/01/process/attempt.json',{})
    elif bad=='stop':(writer.root/'STOP').write_text('stop')
    elif bad=='space':
        needed=load(writer,ref)['storage']['required_free_bytes']
        monkeypatch.setattr(start.shutil,'disk_usage',lambda p:SimpleNamespace(free=needed-1))
    elif bad=='partial':write(root,'meta/pilot-startup-01.json.tmp',{})
    else:ref['sha256']='0'*64
    with pytest.raises(ValueError):start.verify_before_process(writer,1,ref)
    assert failure(writer,1).exists()


def test_request_publication_failure_preserves_attempt_and_partial(writer,monkeypatch):
    args=setup(writer,monkeypatch,0);publish=writer.publish_json
    def fail(name,value,**kw):
        if name=='evidence/pilots/00/request.json':
            return publish(name,{'valid':True,'nonfinite':float('nan')})
        return publish(name,value,**kw)
    monkeypatch.setattr(writer,'publish_json',fail)
    with pytest.raises(ValueError):start.prepare_pilot_request(**args)
    assert (writer.root/'evidence/pilots/00/request.json.tmp').exists()
    assert failure(writer,0).exists()
    with pytest.raises(ValueError,match='cannot be retried'):start.prepare_pilot_request(**args)


def test_storage_drop_after_receipt_publication_is_recorded(writer,monkeypatch):
    args=setup(writer,monkeypatch,0);publish=writer.publish_json
    def drop(name,value,**kw):
        result=publish(name,value,**kw)
        if name=='evidence/meta/pilot-startup-00.json':
            monkeypatch.setattr(start.shutil,'disk_usage',lambda p:SimpleNamespace(free=30_631_281_411))
        return result
    monkeypatch.setattr(writer,'publish_json',drop)
    with pytest.raises(ValueError,match='after startup'):start.prepare_pilot_request(**args)
    assert failure(writer,0).exists() and (writer.root/'evidence/meta/pilot-startup-00.json').exists()


@pytest.mark.parametrize('slot',[0,1,2])
def test_empty_unreviewed_slot_directory_blocks_startup(writer,monkeypatch,slot):
    args=setup(writer,monkeypatch,slot)
    (writer.root/f'evidence/pilots/{slot:02d}').mkdir(parents=True)
    with pytest.raises(ValueError,match='must not exist'):start.prepare_pilot_request(**args)


def test_empty_later_directory_is_not_ignored_at_first_startup(writer,monkeypatch):
    args=setup(writer,monkeypatch,0)
    (writer.root/'evidence/pilots/02').mkdir(parents=True)
    with pytest.raises(ValueError,match='Unexpected pilot directory'):start.prepare_pilot_request(**args)
    assert failure(writer,0).exists()


@pytest.mark.parametrize('name',['pilots/00/process','pilots/02'])
def test_empty_process_or_later_directory_blocks_predispatch(writer,monkeypatch,name):
    args=setup(writer,monkeypatch,0);ref=start.prepare_pilot_request(**args)
    (writer.root/'evidence'/name).mkdir(parents=True)
    with pytest.raises(ValueError,match='Unexpected pilot/process'):start.verify_before_process(writer,0,ref)
    assert failure(writer,0).exists()


@pytest.mark.parametrize('drop',[False,True])
def test_startup_storage_handoff_to_real_bounded_process_helper(writer,monkeypatch,drop):
    import sys
    from src.research import tokenizer_wide_hybrid_process_v2 as process
    args=setup(writer,monkeypatch,0);ref=start.prepare_pilot_request(**args)
    checked=start.verify_before_process(writer,0,ref)
    if drop:
        monkeypatch.setattr(start.shutil,'disk_usage',lambda p:SimpleNamespace(free=checked['required_free_bytes']-1))
    def invoke():
        return process.run_process(writer,'pilots',0,[sys.executable,'-c','print("synthetic process only")'],writer.root,
            required_free_bytes=checked['required_free_bytes'],output_limit_bytes=65536,
            log_limit_bytes=4096,timeout_seconds=5,poll_seconds=.02)
    if drop:
        with pytest.raises(RuntimeError,match='Insufficient disk for remaining allocations'):invoke()
        report=json.loads((writer.root/'evidence/pilots/00/process/process.json').read_text())
        assert report['status']=='failed' and report['pid'] is None
    else:
        report=invoke()
        assert report['status']=='completed' and report['required_free_bytes']==30_631_281_412
        assert (writer.root/'evidence/pilots/00/process/worker.log').read_text()=='synthetic process only\n'
