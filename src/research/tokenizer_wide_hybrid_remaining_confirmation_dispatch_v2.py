"""One slot18/19 transition from an existing same-parent startup/readiness seal.

The caller owns both shared locks, inspects workers and first prepares and seals
the requested slot using separately bounded APIs. This controller never recreates
them.
A dedicated prepare guard encloses full replay of the sealed idle history; a
single supervised child and independently bounded acceptance follow. No CLI,
retry, cleanup, later slot or campaign launch is enabled.
"""
import json
import math
import os
import shutil
import time

from .protocol import ROOT,digest,file_digest
from .cpu_prerequisite_guard import violation
from .tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_retention_receipts import _document,_path
from .tokenizer_wide_hybrid_scores import _require
from .tokenizer_wide_hybrid_cleanup import _locks
from . import tokenizer_wide_hybrid_quality_dispatch_v2 as dispatch
from . import tokenizer_wide_hybrid_remaining_confirmation_startup_v2 as startup
from . import tokenizer_wide_hybrid_quality_entry_v2 as entry

KIND=dispatch.KIND
REPRESENTATION='wide_hybrid_confirmation_acceptance_digest_v1'
acceptance=dispatch.acceptance
readiness=dispatch.readiness
run_process=dispatch.run_process


def _prepare_control(writer,ref,slot=18):
    startup._names(slot)
    name=f'control/quality-controller-prepare-{slot:02d}.json'
    _require(ref['path']==name,'Exact supported controller prepare guard required')
    value=_document(writer.root,ref);elapsed=value.get('elapsed_seconds');sample=value.get('observation',{})
    limits=dict(timeout_seconds=1800,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3)
    _require(set(value)=={'kind','status','reason','pid','limits','elapsed_seconds','observation'}
        and value['kind']==KIND and value['status'] in ('starting','finished') and value['reason'] is None
        and type(value['pid']) is int and value['pid']==os.getpid() and value['limits']==limits
        and type(elapsed) in (int,float) and math.isfinite(elapsed) and 0<=elapsed<1800
        and type(sample) is dict and set(sample)=={'peak_rss_bytes','free_bytes'}
        and all(type(v) is int and v>0 for v in sample.values()) and violation(sample,elapsed,**limits) is None,
        'Bounded current-parent controller prepare guard required')
    return name,dict(size=_path(writer.root,name).stat().st_size,sha256=ref['sha256'])


def _stable_anchor(writer,readiness_ref,prepared):
    """Bind the exact original slot16 wire anchor as well as the expanded seal."""
    from .tokenizer_wide_hybrid_inventory_delta import MARKER,VERSION
    root=writer.root/'evidence'
    _document(root,readiness_ref)
    wire=json.loads(_path(root,readiness_ref['path']).read_text())
    prior=_document(root,prepared['prior_dispatch_ref'])
    anchors=[ref for ref in prior['prior_dispatches'] if ref['path']=='meta/quality-dispatch-16.json']
    _require(len(anchors)==1,'Exact stable original dispatch required')
    previous=anchors[0]
    anchor=_document(root,previous)['review']['readiness']
    _require(anchor['path']=='meta/quality-readiness-16.json'
        and wire.get(MARKER)==VERSION and wire['inventory']['base']==anchor,
        'Exact original slot16 inventory wire anchor required')


def _inventory(slot,seal,prepared,startup_ref):
    startup._names(slot)
    check=entry._slot18_inventory if slot==18 else entry._slot19_inventory
    return check(seal,prepared,startup_ref)


def verify_sealed_confirmation(writer,slot,startup_ref,readiness_ref,controller_guard_ref):
    """Full idle replay with exactly the completed seal and one prepare guard."""
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and slot in (18,19),'Exact sealed slot18 or slot19 required')
    _require(startup_ref['path']==startup._names(slot)['receipt'] and readiness_ref['path']==f'meta/quality-readiness-{slot:02d}.json',
             'Exact supported startup/readiness references required')
    root=writer.root/'evidence';seal=_document(root,readiness_ref);prepared=_document(root,startup_ref)
    _require(seal['kind']==readiness.KIND and seal['status']=='idle_review_sealed_pending_child_integration'
        and type(seal['slot']) is int and seal['slot']==slot and type(seal['parent_pid']) is int
        and seal['parent_pid']==os.getpid()>1 and seal['startup_ref']==startup_ref
        and seal['dispatch_authorized'] is False and seal['deletion_authorized'] is False,
        'Exact same-parent completed readiness required')
    _stable_anchor(writer,readiness_ref,prepared)
    _inventory(slot,seal,prepared,startup_ref)
    _require(seal['predispatch_review']==entry._expected_review(slot,startup_ref,seal)
        and all(seal[k]==prepared[k] for k in ('request','plan_sha256','projection_ref','context')),
        'Complete sealed idle review differs')
    acceptance._guard(writer,seal['guard'],readiness.KIND,os.getpid(),1800)
    _require(seal['guard']['path']==f'control/quality-readiness-{slot:02d}.json'
        and seal['control_inventory'][seal['guard']['path']]['sha256']==seal['guard']['sha256'],
        'Completed readiness guard binding differs')
    bound,request,allowance=readiness._bound(writer,slot,prepared)
    _require(bound['files']==seal['frozen_files'] and bound['context']==seal['context']
        and allowance==seal['output_limit_bytes'],'Sealed frozen inputs differ')
    preparation=dict(attempt_ref=prepared['attempt'],guard_ref=prepared['guard'],pending_request_ref=prepared['request'])
    history=startup.append.history
    if slot==19:
        from . import tokenizer_wide_hybrid_remaining_confirmation_history_v2 as history
    checked=history._verify(writer,slot-1,seal['plan_sha256'],prepared['prior_dispatch_ref'],
        projection_ref=seal['projection_ref'],rank_completion_ref=prepared['rank_completion_ref'],
        preparation=preparation,startup_ref=startup_ref,readiness_guard_ref=seal['guard'],
        readiness_ref=readiness_ref,controller_guard_ref=controller_guard_ref)
    _require(type(seal['required_free_bytes']) is int and seal['required_free_bytes']==checked['storage']['required_free_bytes'],
             'Independent full remaining storage differs')
    _prelaunch(writer,slot,startup_ref,readiness_ref,seal,controller_guard_ref,finished=False)
    return seal


def _prelaunch(writer,slot,startup_ref,readiness_ref,seal,guard_ref,*,finished=True):
    """Pin evidence, controls, independent frozen inputs, STOP and capacity again."""
    root=writer.root/'evidence';prepared=_document(root,startup_ref)
    _require(_document(root,readiness_ref)==seal,'Readiness changed before dispatch')
    inventory=_inventory(slot,seal,prepared,startup_ref)
    inventory[readiness_ref['path']]=dict(size=_path(root,readiness_ref['path']).stat().st_size,sha256=readiness_ref['sha256'])
    _require(acceptance._inventory(root)==inventory,'Unsealed evidence before dispatch')
    name,binding=_prepare_control(writer,guard_ref,slot)
    if finished:acceptance._guard(writer,guard_ref,KIND,os.getpid(),1800)
    _require(readiness._controls(writer,'not-a-control')==dict(seal['control_inventory'],**{name:binding}),
             'Exact sealed controls plus current prepare guard required')
    bound,_,allowance=readiness._bound(writer,slot,prepared)
    _require(bound['files']==seal['frozen_files'] and bound['context']==seal['context']
        and allowance==seal['output_limit_bytes'],'Frozen inputs changed before dispatch')
    _locks(root,writer.lock_fds)
    _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents dispatch')
    _require(shutil.disk_usage(writer.root).free>=seal['required_free_bytes'],'Full remaining storage changed before dispatch')


def run_sealed_confirmation_once(writer,slot,startup_ref,readiness_ref):
    """Consume one existing supported seal; never prepare, retry or advance."""
    _require(isinstance(writer,MetadataWriter) and type(slot) is int and slot in (18,19),'Exact dispatch slot18 or slot19 required')
    dispatch._fresh(writer,slot,24);root=writer.root/'evidence';base=f'meta/quality-dispatch-{slot:02d}'
    try:
        with dispatch._PhaseGuard(writer,slot,'prepare'):
            seal=verify_sealed_confirmation(writer,slot,startup_ref,readiness_ref,dispatch._guard_ref(writer,slot,'prepare'))
        prepare_ref=dispatch._guard_ref(writer,slot,'prepare')
        _prelaunch(writer,slot,startup_ref,readiness_ref,seal,prepare_ref)
        monitor=dispatch._ChildMonitor(writer,slot)
        try:
            process=run_process(writer,'quality',slot,entry.worker_argv(writer,slot,startup_ref,readiness_ref),ROOT,
                required_free_bytes=seal['required_free_bytes'],output_limit_bytes=seal['output_limit_bytes'],
                log_limit_bytes=4*1024**2,timeout_seconds=7200,parent_check=monitor.check)
        except BaseException as exc:
            try:monitor.finish(exc)
            except BaseException as error:exc.add_note('Parent monitor publication failed: '+repr(error)[:1000])
            raise
        monitor.finish()
        with dispatch._PhaseGuard(writer,slot,'review'):
            review=acceptance.review_completed_remaining_confirmation(writer,slot,startup_ref,readiness_ref,process)
        guards={phase:dispatch._guard_ref(writer,slot,phase) for phase in ('prepare','child','review')}
        for phase,ref in guards.items():acceptance._guard(writer,ref,KIND,os.getpid(),7200 if phase=='child' else 1800)
        prepared=_document(root,startup_ref);bound,_,allowance=readiness._bound(writer,slot,prepared)
        _require(bound['files']==seal['frozen_files'] and bound['context']==seal['context']
            and allowance==seal['output_limit_bytes'],'Frozen confirmation inputs changed before receipt')
        _require(digest(acceptance._inventory(root))==review['evidence_inventory_sha256'],
                 'Evidence changed after acceptance')
        expected=dict(seal['control_inventory'])
        for ref in [*guards.values(),review['entry_guard'],review['supervisor_ref']]:
            _document(writer.root,dict(path=ref['path'],sha256=ref['sha256']))
            expected[ref['path']]=dict(size=_path(writer.root,ref['path']).stat().st_size,sha256=ref['sha256'])
        _require(readiness._controls(writer,'not-a-control')==expected,'Unexpected or changed controls after acceptance')
        _locks(root,writer.lock_fds)
        _require(not (writer.root/'STOP').exists() and not (writer.root/'STOP').is_symlink(),'STOP prevents success publication')
        _require(shutil.disk_usage(writer.root).free>=review['storage']['required_free_bytes'],
                 'Full remaining storage changed before receipt')
        compact=dict(review,representation=REPRESENTATION)
        for key in ('completed_records','checkpoint_pairs','stages'):compact[key+'_sha256']=digest(compact.pop(key))
        prior=_document(root,prepared['prior_dispatch_ref'])
        result=dict(kind=KIND,status='confirmation_reviewed',slot=slot,parent_pid=os.getpid(),
            plan_sha256=seal['plan_sha256'],projection_ref=seal['projection_ref'],
            pilot_completion_ref=prior['pilot_completion_ref'],rank_completion_ref=prepared['rank_completion_ref'],
            prior_dispatches=[*prior['prior_dispatches'],prepared['prior_dispatch_ref']],guards=guards,review=compact)
        path=writer.publish_json('evidence/'+base+'.json',result)
        return dict(path=base+'.json',sha256=file_digest(path))
    except BaseException as exc:
        try:writer.publish_json('evidence/'+base+'-failure.json',dict(kind=KIND,status='failed',slot=slot,
            startup_ref=startup_ref,readiness_ref=readiness_ref,error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Dispatch failure publication failed: '+repr(error)[:1000])
        raise
