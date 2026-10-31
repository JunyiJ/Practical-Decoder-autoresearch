"""Read-only preservation of failed second entry; never call a failed-root entry."""
from datetime import datetime,timezone
import json
from pathlib import Path

from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock,MetadataWriter,reference
from src.research.tokenizer_activation_timing import review_result
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.tokenizer_serial_process import _group_exists
from research.activation_inference_recovery_v1 import context as c,controller as e,worker as w

base=ROOT/'research/tokenizer-evidence/activation-preparation-001';root=c.RECOVERY_ROOT
stem='inference-recovery001-second-entry-failure'
guard=base/(stem+'-guard.json');report=base/(stem+'-review.json');dest=base/(stem+'-portable')
if any(p.exists() for p in (guard,report,dest)):raise FileExistsError('Existing failure audit; no overwrite')
with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as r:
    idle()
    with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):
        ctx=c.Recovery(MetadataWriter(root,(g,r)),reference(root,c.PLAN_NAME));before=snapshot(root)
        assert ctx.plan['sha256']=='29b8265276d6bedb933bbb3f522b179ffa1b8403b18ac7f7b62b5b6638ee5dde'
        plan=ctx.document(ctx.plan)
        assert plan['external_source']==c.external_source() and plan['original_source_sha256']==digest(source_fingerprint())
        assert plan['runtime']==runtime_identity()
        imported=ctx.document(plan['source_import']);assert plan['source_import']['sha256']==c.IMPORT_SHA256
        with c.original_session(ctx) as source:
            c.verify_original(ctx,source,imported)
            prior_ref=reference(ROOT,'research/tokenizer-evidence/activation-preparation-001/inference-recovery001-prefix-01-review.json')
            prior=json.loads((ROOT/prior_ref['path']).read_text())
            assert prior['status']=='recovery_prefix_independently_reviewed' and prior['accepted_seeds']==[3611]
            assert prior['plan']==ctx.plan and prior['original_import_sha256']==digest(imported)
            pg=json.loads((ROOT/prior['guard']['path']).read_text())
            assert reference(ROOT,prior['guard']['path'])==prior['guard'] and pg['status']=='finished' and pg['reason'] is None
            for ref in prior['copied'].values():ctx.verify(ref)
            accepted=prior['accepted'][0];done=ctx.document(accepted)
            rq=e.request_check(ctx,3611,done['request'],[],imported)
            first_ref,first_process=e.process_review(ctx,3611,done['request'])
            assert done['process']==first_ref and done['files']==e.inventory(ctx,3611,exclude_acceptance=True)
            for phase,pid in [('entry',first_process['pid']),('worker',first_process['pid']),('review',rq['parent_pid'])]:
                c.verify_guard(ctx,done[phase+'_guard'],pid)
            assert not _group_exists(first_process['pid'])
            models=ctx.document(reference(root,'evidence/inference/3511/models.json'))
            assert models==w.expected_models(source,rq['binding'])
            first_timing=review_result(ctx.document(reference(root,'evidence/inference/3511/result.json')),3611,
                                      imported['selection'],imported['inputs']['training96_sha256'])
            saved_review=ctx.document(done['review']);assert saved_review['review']==first_timing
        process=ctx.document(reference(root,'evidence/inference/3512/process/process.json'))
        failure=ctx.document(reference(root,'evidence/inference/3512/dispatch-failure.json'))
        entry=ctx.document(reference(root,'control/recovery-entry-inference-3512.json'))
        assert process['status']=='failed' and process['returncode']==1 and failure['retry_allowed'] is False
        assert entry['status']=='failed' and 'Recovery output changed during review' in entry['reason']
        assert not _group_exists(process['pid'])
        assert not any(name.startswith('evidence/inference/3513/') for name in before['files'])
        for name in ('attempt.json','models.json','result.json','worker-outputs.json','accepted.json'):
            assert not (root/'evidence/inference/3512'/name).exists()
        log=(root/'evidence/inference/3512/process/worker.log').read_text()
        assert 'accepted_prefix' in log and 'budget(ctx)==before' in log and 'Recovery output changed during review' in log
        copied={}
        for name in before['files']:
            if name.startswith('evidence/inference/3512/') or name.startswith('control/') and name.endswith('inference-3512.json'):
                ref=reference(root,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as stream:stream.write((root/name).read_bytes())
                assert file_digest(target)==ref['sha256'];copied[name]=ref
        assert snapshot(root)==before
        idle()
value=dict(status='recovery001_first_pair_valid_second_entry_failed_no_retry',observed_at=datetime.now(timezone.utc).isoformat(),
    plan=ctx.plan,source_import_sha256=digest(imported),prior_valid_review=prior_ref,first_accepted=accepted,
    first_process=first_process,first_timing_review=first_timing,failed_process=process,failed_entry_guard=entry,
    protected_original_checkpoints=imported['protected_checkpoints'],copied=copied,
    guard=reference(ROOT,str(guard.relative_to(ROOT))),procedure_sha256=file_digest(Path(__file__)),
    diagnosis='Second child failed during accepted_prefix replay of the already completed first pair. worker.review_outputs compares the entire recovery census before/after a full original-source audit, while the parent updates the current second-entry process monitor. Synchronized individual snapshots do not make this multi-snapshot whole-root equality stable.',
    limitations='The exact differing census entries were not logged, so attributing the difference to a specific monitor byte is an inference. All first-pair artifacts/guards/model bindings/timing arithmetic and all sixteen original checkpoints reverified intact. The second child never reached produce/model loading; no second timing result exists.',
    next_action='Do not edit frozen v1 runtime, retry recovery001 or start3613 there. Prepare a separate versioned remaining-pair continuation: import first accepted pair byte-for-byte, compare completed-slot immutable inventories separately from active mutable control files, retain complete synchronized global quota/STOP/failure checks, and test actual concurrent parent monitor plus later-child prior review before a new freeze. No repeat of seed3611, no training or weight regeneration. New version must bind unchanged timing/quality rules and explicit source applicability.',
    original_campaign_retried=False,recovery001_retry_allowed=False,accepted_inference_pairs=1,
    formal_final_completed=False,holdout_scored=False)
with report.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({k:value[k] for k in ('status','accepted_inference_pairs','formal_final_completed')}))
