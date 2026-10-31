"""One explicit new-root inference freeze; never retry or mutate activation001."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock,MetadataWriter,Session,reference
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from research.activation_inference_recovery_v1 import context as c
from research.activation_inference_recovery_v1.source_binding import ORIGINAL_ROOT,SOURCE_SHA256,verify_source_import

base=ROOT/'research/tokenizer-evidence/activation-preparation-001'
root=c.RECOVERY_ROOT
stem='inference-recovery001-freeze'
guard=base/(stem+'-guard.json');review=base/(stem+'-review.json');portable=base/(stem+'-plan.json')
original_import=base/'inference-recovery-binding-prerequisite-001-import.json'
if root.exists() or any(p.exists() for p in (guard,review,portable)):
    raise FileExistsError('Existing freeze root or evidence is permanent; no retry')
error=None;plan_ref=None;capacity=None
try:
    with session_lock(GLOBAL_LOCK.parent) as g:
        idle()
        with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):
            assert digest(source_fingerprint())==SOURCE_SHA256
            assert file_digest(original_import)==c.IMPORT_SHA256
            imported=json.loads(original_import.read_text())
            before=snapshot(ORIGINAL_ROOT)
            with session_lock(ORIGINAL_ROOT) as s:
                source=Session(MetadataWriter(ORIGINAL_ROOT,(g,s)),reference(ORIGINAL_ROOT,'evidence/meta/activation-plan.json'))
                verify_source_import(source,imported)
            assert snapshot(ORIGINAL_ROOT)==before
            prerequisites={name:reference(ROOT,path) for name,path in c.PREREQUISITES.items()}
            for name,ref in prerequisites.items():
                report=json.loads((ROOT/ref['path']).read_text())
                assert report['status']=='passed' and report['pytest_exit_code']==0
                assert all(t['outcome']=='passed' for t in report['tests']) and report['tests']
                for path,sha in report['code_sha256'].items():
                    p=Path(path);p=p if p.is_absolute() else ROOT/p
                    assert file_digest(p)==sha
                guard_path=(ROOT/report['guard']['path'] if isinstance(report.get('guard'),dict)
                    else ROOT/report['outputs']['guard']['path'] if 'outputs' in report
                    else base/('inference-recovery-execution-prerequisite-002-guard.json' if name in ('execution','final')
                               else 'inference-recovery-census-prerequisite-001-guard.json'))
                guard_record=json.loads(guard_path.read_text())
                assert guard_record['status']=='finished' and guard_record['reason'] is None
                assert guard_record['limits']==dict(reserve_bytes=8*1024**3,rss_limit_bytes=4*1024**3,timeout_seconds=180)
                if report.get('guard_sha256'):assert file_digest(guard_path)==report['guard_sha256']
                if name=='quality':assert reference(ROOT,report['outputs']['guard']['path'])==report['outputs']['guard']
                if name=='binding':assert reference(ROOT,report['guard']['path'])==report['guard']
            assert shutil.disk_usage(ROOT).free>=c.STORAGE['full_required_free_bytes']
            external=c.external_source()
            root.mkdir(exist_ok=False)
            with session_lock(root) as r:
                writer=MetadataWriter(root,(g,r))
                # Preserve the exact previously reviewed import bytes/hash;
                # re-encoding JSON would invalidate its fixed identity.
                with writer.open_append(c.IMPORT_NAME) as stream:stream.write(original_import.read_bytes())
                source_import=reference(root,c.IMPORT_NAME)
                assert source_import['sha256']==c.IMPORT_SHA256
                plan=dict(kind=c.KIND,status='frozen',root=str(root),external_source=external,
                    original_source_sha256=SOURCE_SHA256,runtime=runtime_identity(),source_import=source_import,
                    timing=c.CONTRACT,final=c.FINAL,storage=c.STORAGE,prerequisites=prerequisites)
                writer.publish_json(c.PLAN_NAME,plan);plan_ref=reference(root,c.PLAN_NAME)
                ctx=c.Recovery(writer,plan_ref)
                assert c.bind(ctx,capacity=True)==imported
                seen=c.budget(ctx,capacity=True)
                assert set(seen['files'])=={'.lock','control/.metadata.lock',c.IMPORT_NAME,c.PLAN_NAME}
                capacity=dict(free_bytes=shutil.disk_usage(root).free,metadata_bytes=seen['metadata_bytes'],
                    required_remaining_bytes=c.CONTRACT['reserve_bytes']+c.METADATA-seen['metadata_bytes'],
                    checkpoint_allocation_bytes=0)
                with portable.open('xb') as stream:stream.write((root/c.PLAN_NAME).read_bytes())
                assert file_digest(portable)==plan_ref['sha256']
                assert c.external_source()==external and digest(source_fingerprint())==SOURCE_SHA256
            assert snapshot(ORIGINAL_ROOT)==before
            idle()
except Exception as exc:error=repr(exc)
value=dict(status='passed' if error is None else 'failed',error=error,observed_at=datetime.now(timezone.utc).isoformat(),
    root=str(root),plan=plan_ref,capacity=capacity,guard=reference(ROOT,str(guard.relative_to(ROOT))) if guard.exists() else None,
    portable_plan=reference(ROOT,str(portable.relative_to(ROOT))) if portable.exists() else None,
    procedure_sha256=file_digest(Path(__file__)),
    applicability=dict(original_source_unchanged=True if error is None else None,
        reused_original_functions=['tokenizer_wide_hybrid_metadata_writer.MetadataWriter','tokenizer_wide_hybrid_process_v2.run_process',
            'tokenizer_activation_inference.load_pair','tokenizer_activation_inference.expected_models','tokenizer_activation_inference.callbacks',
            'tokenizer_activation_inference.strict_mps','tokenizer_activation_timing.measure_pair','tokenizer_activation_timing.review_result',
            'tokenizer_activation_timing.summarize','tokenizer_activation_final.analyze'],
        excluded_original_entries=['tokenizer_activation_inference.produce','tokenizer_activation_inference.budget',
            'tokenizer_activation_inference.quality_binding','tokenizer_activation_inference_entry','tokenizer_activation_final.complete'],
        original_scope='Read-only source Session; checkpoints remain at original paths with original lineage. No retraining, rescoring or cleanup.',
        new_scope='External source hash set replaces orchestration only. New-root synchronized budget outside timed kernels; original parent supervisor scans already serialized. New-root worker/receipt/final contracts explicitly identify preserved original failure.',
        source_lock_handoff='Parent releases original-root lock before dispatch; child independently acquires it with inherited global ownership. Parent monitors only new root; reacquires original after child exit.',
        panel='Exactly seeds3611/3612/3613, unchanged two-family/batch1/8 workloads and resource limits; one pair per step; failures permanently latch; final requires all three accepted pairs.'),
    inference_launched=False,training_launched=False,holdout_scored=False,commit_and_push_before_dispatch=True)
with review.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({k:value[k] for k in ('status','error','root','plan','capacity')}))
raise SystemExit(0 if error is None else 1)
