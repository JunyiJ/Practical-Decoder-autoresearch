"""First fresh parent acceptance/serial transition, entirely synthetic children."""
import copy
import json
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_quality_dispatch_v2 as dispatch
from src.research import tokenizer_wide_hybrid_first_confirmation_startup_v2 as fresh
from test_tokenizer_wide_hybrid_metadata_writer import writer
from test_tokenizer_wide_hybrid_first_confirmation_startup_v2 import setup as initial
from test_tokenizer_wide_hybrid_quality_acceptance_v2 import complete

accept=dispatch.acceptance


def setup(writer,monkeypatch):
    args=initial(writer,monkeypatch);calls=[]
    free=SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=64*1024**3))
    for module in (dispatch,accept,dispatch.readiness):monkeypatch.setattr(module,'shutil',free)
    def process(w,stage,slot,argv,cwd,**limits):
        assert w is writer and stage=='quality' and slot==15 and cwd==dispatch.ROOT
        assert limits['timeout_seconds']==7200 and limits['log_limit_bytes']==4*1024**2
        guard=json.loads((writer.root/'control/quality-controller-prepare-15.json').read_text())
        assert guard['status']=='finished'
        startup=dict(path='meta/quality-confirm-startup-15.json',sha256=argv[9])
        ready=dict(path='meta/quality-readiness-15.json',sha256=argv[15])
        seal=dispatch._document(writer.root/'evidence',ready)
        assert limits['required_free_bytes']==seal['required_free_bytes']
        assert limits['output_limit_bytes']==seal['output_limit_bytes']
        limits['parent_check']();calls.append(slot)
        return complete(writer,monkeypatch,slot,startup,ready)['process']
    monkeypatch.setattr(dispatch,'run_process',process)
    return args,calls


def test_first_fresh_parent_independent_audit_storage_and_tampering(writer,monkeypatch):
    args,_=setup(writer,monkeypatch);root=writer.root/'evidence'
    start=fresh.prepare_first_confirmation_request(**args)
    seal=dispatch.readiness.seal_quality_readiness(writer,15,start)
    completed=complete(writer,monkeypatch,15,start,seal)
    before=accept._inventory(root);storage=accept.bind_storage;seen=[]
    def capacity(*a,**kw):
        assert len(kw['records'])==16 and len(kw['pilots'])==3 and len(kw['stages'])==3
        assert len(kw['checkpoint_pairs'])==19 and all(v['divisor']==8 for v in kw['selection'].values())
        seen.append(1);return storage(*a,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(accept,'bind_storage',capacity)
        report=accept.review_completed_first_confirmation(**completed)
    assert seen==[1] and report['kind']==accept.CONFIRM_KIND
    assert report['audited_indices']==[9,10,11,15] and len(report['checkpoint_pairs'])==4
    assert report['storage']['cleanup_boundaries_verified']==3 and report['selection'] is not None
    assert all(report[k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
    assert accept._inventory(root)==before and len(list(root.rglob('*.pt')))==8
    for name in ('meta/numerical.json','meta/rank-selection.json','.retention/rank/deletions.jsonl',
        'quality/09/checkpoints/final_001200_123456.pt','quality/15/checkpoints/h200.pt',
        'quality/15/result.json','quality/15/tensor-audit.json','quality/15/worker-outputs.json',
        'quality/15/resource-audit.json','quality/15/process/launch.json'):
        path=root/name;old=path.read_bytes();path.write_bytes(b'{}')
        try:
            with pytest.raises((ValueError,KeyError)):accept.review_completed_first_confirmation(**completed)
        finally:path.write_bytes(old)
    terminal=root/'quality/15/process/process.json';old=terminal.read_bytes()
    for key,value in [('returncode',True),('parent_pid',1),('pid',1),('log_truncated',True),('timeout_seconds',600)]:
        changed=copy.deepcopy(completed['process']);changed[key]=value;terminal.write_text(json.dumps(changed))
        try:
            with pytest.raises(ValueError):accept.review_completed_first_confirmation(**{**completed,'process':changed})
        finally:terminal.write_bytes(old)
    with monkeypatch.context() as patch:
        def late(*a,**kw):
            result=storage(*a,**kw)
            patch.setattr(accept,'shutil',SimpleNamespace(disk_usage=lambda p:SimpleNamespace(free=8*1024**3)))
            return result
        patch.setattr(accept,'bind_storage',late)
        with pytest.raises(ValueError):accept.review_completed_first_confirmation(**completed)
    assert accept._inventory(root)==before


def test_first_fresh_serial_transition_preserves_selection_and_stops_before_slot16(writer,monkeypatch):
    args,calls=setup(writer,monkeypatch);root=writer.root/'evidence';before=accept._inventory(root)
    original=accept.review_completed_first_confirmation;audited=[]
    def capture(*a,**kw):
        value=original(*a,**kw);audited.append(value);return value
    monkeypatch.setattr(accept,'review_completed_first_confirmation',capture)
    ref=dispatch.run_first_confirmation_once(**args);saved=dispatch._document(root,ref)
    assert calls==[15] and saved['status']=='first_confirmation_reviewed' and saved['prior_dispatches']==[]
    assert saved['rank_completion_ref']==args['rank_completion_ref']
    assert saved['review']['kind']==accept.CONFIRM_KIND and len(audited)==1
    assert saved['review']['representation']=='wide_hybrid_first_confirmation_acceptance_digest_v1'
    for key in ('completed_records','checkpoint_pairs','stages'):
        assert key not in saved['review'] and saved['review'][key+'_sha256']==dispatch.digest(audited[0][key])
    assert len(audited[0]['completed_records'])==16 and (root/ref['path']).stat().st_size<16000
    assert saved['review']['audited_indices']==[9,10,11,15]
    assert saved['review']['storage']['cleanup_boundaries_verified']==3
    after=accept._inventory(root);assert all(after[n]==v for n,v in before.items())
    assert len(list(root.rglob('*.pt')))==8 and not (root/'quality/16').exists()
    assert all(saved['review'][k] is False for k in ('adopted','dispatch_authorized','deletion_authorized','next_stage_authorized'))
    with pytest.raises(ValueError,match='cannot be retried'):dispatch.run_first_confirmation_once(**args)
    assert accept._inventory(root)==after
    census=dispatch.snapshot(writer.root)
    (writer.root.parent/'first-confirmation-dispatch-census.json').write_text(json.dumps(dict(
        groups=census['groups'],metadata_bytes=census['metadata_bytes'],file_count=len(census['files']),
        receipt_bytes=(root/ref['path']).stat().st_size,live_checkpoint_files=8,
        completed_synthetic_fresh=1,real_panel_fit_proven=False),sort_keys=True))


@pytest.mark.parametrize('phase',['process','review','source_after_review','evidence_after_review'])
def test_first_fresh_failure_retains_evidence_and_blocks_retry(writer,monkeypatch,phase):
    args,calls=setup(writer,monkeypatch)
    def fail(*a,**kw):raise RuntimeError('synthetic '+phase)
    if phase=='process':monkeypatch.setattr(dispatch,'run_process',fail)
    elif phase=='review':monkeypatch.setattr(accept,'review_completed_first_confirmation',fail)
    else:
        original=dispatch._PhaseGuard.__exit__
        def late(guard,*a,**kw):
            result=original(guard,*a,**kw)
            if guard.name=='control/quality-controller-review-15.json':
                name='meta/source.txt' if phase=='source_after_review' else 'quality/15/result.json'
                (writer.root/'evidence'/name).write_bytes(b'late mutation')
            return result
        monkeypatch.setattr(dispatch._PhaseGuard,'__exit__',late)
    with pytest.raises((RuntimeError,ValueError)):dispatch.run_first_confirmation_once(**args)
    root=writer.root/'evidence';failure=json.loads((root/'meta/quality-dispatch-15-failure.json').read_text())
    assert failure['retry_allowed'] is False and not (root/'meta/quality-dispatch-15.json').exists()
    assert calls==([] if phase=='process' else [15])
    before=accept._inventory(root)
    with pytest.raises(ValueError):dispatch.run_first_confirmation_once(**args)
    assert accept._inventory(root)==before and not (root/'quality/16').exists()


@pytest.mark.parametrize('slot',[0,9,14,16,23,24,True,15.0,'15'])
def test_fresh_parent_apis_reject_other_slots_before_writes(writer,slot):
    with pytest.raises(ValueError):dispatch.run_first_confirmation_once(writer,slot,'a'*64,projection_ref={},rank_completion_ref={})
    with pytest.raises(ValueError):accept.review_completed_first_confirmation(writer,slot,{},{},{})
    assert not (writer.root/'evidence').exists()
