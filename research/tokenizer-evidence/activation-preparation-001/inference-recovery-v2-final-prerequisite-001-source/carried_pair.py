"""Read-only exact import of recovery001's valid first pair and later failure.

No v1 bind/budget/entry/producer is called on the failed root. This historical
reader independently reconstructs only the accepted first-pair provenance and
pins the entire now-idle failed root. It never grants a retry there.
"""
import json
from pathlib import Path
import shutil
import time

from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_activation_session import reference,require
from src.research.tokenizer_activation_entry import session_lock,MetadataWriter
from src.research.tokenizer_activation_timing import review_result
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.tokenizer_serial_process import _group_exists
from research.activation_inference_recovery_v1 import context as old,controller as entry,worker as outputs
from research.activation_inference_recovery_v1.source_binding import SOURCE_SHA256,verify_source_import

ANCHORS={
 'prefix':('research/tokenizer-evidence/activation-preparation-001/inference-recovery001-prefix-01-review.json',
           '3386b679511fef2873080f7f9154a387b92ea4874a7097e7b6bcce9e2bf52c27'),
 'failure':('research/tokenizer-evidence/activation-preparation-001/inference-recovery001-second-entry-failure-review.json',
            'f2e66d18e2c665e5aef52f87c0a848672c7286d5f5da2b9d4044193bd9724af2')}
V1_PLAN_SHA256='29b8265276d6bedb933bbb3f522b179ffa1b8403b18ac7f7b62b5b6638ee5dde'
SECONDS=180.0


def _anchor(name):
    path,sha=ANCHORS[name];ref=reference(ROOT,path)
    require(ref['sha256']==sha and ref['size']<=2*1024**2,'Preserved v1 audit anchor changed')
    return ref,json.loads((ROOT/path).read_text())


def _verify_first(ctx,source,imported,accepted):
    require(accepted['path']=='evidence/inference/3511/accepted.json','Only original first pair may be carried')
    done=ctx.document(accepted)
    rq=entry.request_check(ctx,3611,done['request'],[],imported)
    process,p=entry.process_review(ctx,3611,done['request'])
    require(done==dict(kind=old.KIND,status='accepted',seed=3611,request=done['request'],process=process,
        review=done['review'],entry_guard=done['entry_guard'],worker_guard=done['worker_guard'],
        review_guard=done['review_guard'],files=done['files']),'Complete original first acceptance required')
    require(not _group_exists(p['pid']),'Original first process group remains live')
    for phase,pid in [('entry',p['pid']),('worker',p['pid']),('review',rq['parent_pid'])]:
        ref=done[phase+'_guard']
        require(ref['path']==f'control/recovery-{phase}-inference-3511.json','Original first guard path differs')
        old.verify_guard(ctx,ref,pid)
    require(done['files']==entry.inventory(ctx,3611,exclude_acceptance=True),'Original first producer inventory changed')
    stem='evidence/inference/3511'
    refs={name:reference(ctx.root,stem+'/'+name+'.json') for name in ('request','attempt','models','result','worker-outputs')}
    require(refs['request']==done['request'],'Carried request differs')
    identity=dict(request=done['request'],pid=p['pid'],parent_pid=rq['parent_pid'],seed=3611)
    require(ctx.document(refs['attempt'])==identity,'Carried worker identity differs')
    require(ctx.document(refs['worker-outputs'])==dict(kind=outputs.OUTPUT_KIND,**identity,
        status='complete_pending_parent_review',models=refs['models'],result=refs['result']),
        'Carried worker completion differs')
    require(ctx.document(refs['models'])==outputs.expected_models(source,rq['binding']),
            'Carried original model provenance differs')
    result=ctx.document(refs['result'])
    reviewed=review_result(result,3611,imported['selection'],imported['inputs']['training96_sha256'])
    expected=dict(kind=outputs.REVIEW_KIND,status='reviewed_pending_transport',request=done['request'],
        original_import_sha256=digest(imported),binding_sha256=digest(rq['binding']),review=reviewed,files=list(refs.values()))
    require(done['review']['path']==stem+'/review.json' and ctx.document(done['review'])==expected,
            'Carried independent timing review differs')
    return dict(acceptance=accepted,request=done['request'],process=process,review=done['review'],
        guards={phase:done[phase+'_guard'] for phase in ('entry','worker','review')},
        binding=rq['binding'],result_ref=refs['result'],result=result,timing_review=reviewed)


def collect_carried_pair(global_fd,source,imported):
    started=time.monotonic()
    require(global_fd==source.writer.lock_fds[0],'Original shared worker ownership required')
    def check():
        require(time.monotonic()-started<SECONDS,'Carried-pair import time limit')
        require(shutil.disk_usage(ROOT).free>=old.CONTRACT['reserve_bytes'],'Carried-pair disk reserve')
    check();verify_source_import(source,imported)
    anchors={};audits={}
    for name in ANCHORS:anchors[name],audits[name]=_anchor(name)
    prior=audits['prefix'];failed=audits['failure']
    require(prior['status']=='recovery_prefix_independently_reviewed' and prior['accepted_seeds']==[3611]
        and failed['status']=='recovery001_first_pair_valid_second_entry_failed_no_retry'
        and failed['accepted_inference_pairs']==1 and failed['recovery001_retry_allowed'] is False
        and failed['formal_final_completed'] is False and failed['holdout_scored'] is False,
        'Exact accepted-first/failed-second history required')
    require(failed['prior_valid_review']==anchors['prefix'] and failed['first_accepted']==prior['accepted'][0]
        and prior['original_import_sha256']==digest(imported) and failed['source_import_sha256']==digest(imported)
        and failed['protected_original_checkpoints']==imported['protected_checkpoints'],'Original import or audit linkage differs')
    for audit in audits.values():
        guard=audit['guard'];require(reference(ROOT,guard['path'])==guard,'Original audit guard changed')
        record=json.loads((ROOT/guard['path']).read_text())
        require(record['status']=='finished' and record['reason'] is None,'Original audit guard not complete')
    with session_lock(old.RECOVERY_ROOT) as c:
        ctx=old.Recovery(MetadataWriter(old.RECOVERY_ROOT,(global_fd,c)),reference(old.RECOVERY_ROOT,old.PLAN_NAME))
        before=snapshot(ctx.root);plan=ctx.document(ctx.plan)
        require(ctx.plan['sha256']==V1_PLAN_SHA256 and prior['plan']==ctx.plan and failed['plan']==ctx.plan,
                'Exact frozen v1 plan required')
        require(plan['external_source']==old.external_source() and plan['original_source_sha256']==SOURCE_SHA256
            and digest(source_fingerprint())==SOURCE_SHA256 and plan['runtime']==runtime_identity(),
            'Frozen original/v1 source or runtime changed')
        require(plan['source_import']['sha256']==old.IMPORT_SHA256 and ctx.document(plan['source_import'])==imported,
                'V1 original import changed')
        expected={'.lock','control/.metadata.lock',old.PLAN_NAME,old.IMPORT_NAME}|set(prior['copied'])|set(failed['copied'])
        require(set(before['files'])==expected and before['checkpoint_logical_bytes']==0,
                'Failed v1 root inventory changed; no additional attempts allowed')
        for audit in audits.values():
            for ref in audit['copied'].values():check();ctx.verify(ref)
        pair=_verify_first(ctx,source,imported,prior['accepted'][0])
        failed_ref=reference(ctx.root,'evidence/inference/3512/process/process.json');process=ctx.document(failed_ref)
        require(process==failed['failed_process'] and process['status']=='failed' and process['returncode']==1
                and not _group_exists(process['pid']),'Exact failed second process required')
        files={name:reference(ctx.root,name) for name in sorted(before['files'])}
        require(snapshot(ctx.root)==before,'V1 failed root changed while importing')
        for ref in files.values():check();ctx.verify(ref)
    require(all(reference(ROOT,ref['path'])==ref for ref in anchors.values()),'Historical audit changed during import')
    check()
    return dict(kind='activation_carried_pair_v2',seed=3611,original_root=str(old.RECOVERY_ROOT),plan=ctx.plan,
        original_import_sha256=digest(imported),anchors=anchors,files=files,pair=pair,
        frozen_v1_external_source=plan['external_source'],failed_second_process=failed_ref,
        accepted_timing_pairs=1,remaining_seeds=[3612,3613],original_failure_retried=False,
        repeat_first_pair_allowed=False,dispatch_authorized=False)


def verify_carried_pair(global_fd,source,imported,expected):
    current=collect_carried_pair(global_fd,source,imported)
    require(type(expected) is dict and digest(current)==digest(expected),'Frozen carried-pair import differs')
    return current
