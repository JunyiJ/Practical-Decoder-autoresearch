"""One finite remaining-pair freeze; neither closed root may be retried."""
from datetime import datetime,timezone
import json,shutil
from pathlib import Path
from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock,MetadataWriter,Session,reference
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from research.activation_inference_recovery_v1.source_binding import ORIGINAL_ROOT,SOURCE_SHA256
from research.activation_inference_recovery_v1.context import RECOVERY_ROOT as CLOSED_ROOT
from research.activation_inference_recovery_v2.carried_pair import verify_carried_pair
from research.activation_inference_recovery_v2 import context as c

base=ROOT/'research/tokenizer-evidence/activation-preparation-001';root=c.RECOVERY_ROOT
stem='inference-recovery002-freeze'
guard=base/(stem+'-guard.json');review=base/(stem+'-review.json');portable=base/(stem+'-plan.json')
original_path=base/'inference-recovery-binding-prerequisite-001-import.json'
carried_path=base/'inference-recovery-v2-carried-pair-prerequisite-001-import.json'
if root.exists() or any(p.exists() for p in (guard,review,portable)):raise FileExistsError('Permanent attempt; no retry')
error=None;plan_ref=None;capacity=None
try:
    with session_lock(GLOBAL_LOCK.parent) as g:
        idle()
        with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):
            assert digest(source_fingerprint())==SOURCE_SHA256
            assert file_digest(original_path)==c.IMPORT_SHA256 and file_digest(carried_path)==c.CARRIED_SHA256
            imported=json.loads(original_path.read_text());carried=json.loads(carried_path.read_text())
            before={str(p):snapshot(p) for p in (ORIGINAL_ROOT,CLOSED_ROOT)}
            with session_lock(ORIGINAL_ROOT) as s:
                source=Session(MetadataWriter(ORIGINAL_ROOT,(g,s)),reference(ORIGINAL_ROOT,'evidence/meta/activation-plan.json'))
                assert verify_carried_pair(g,source,imported,carried)==carried
            prerequisites={name:reference(ROOT,path) for name,path in c.PREREQUISITES.items()}
            for name,ref in prerequisites.items():
                report=json.loads((ROOT/ref['path']).read_text())
                assert report['status']=='passed' and report['pytest_exit_code']==0
                assert report['tests'] and all(t['outcome']=='passed' for t in report['tests'])
                for name_or_path,sha in report['code_sha256'].items():assert file_digest(ROOT/name_or_path)==sha
                guard_path=ROOT/ref['path'].replace('-review.json','-guard.json')
                log_path=ROOT/ref['path'].replace('-review.json','.log')
                if 'outputs' in report:
                    assert reference(ROOT,str(guard_path.relative_to(ROOT)))==report['outputs']['guard']
                    assert reference(ROOT,str(log_path.relative_to(ROOT)))==report['outputs']['log']
                else:
                    assert file_digest(guard_path)==report['guard_sha256'] and file_digest(log_path)==report['log_sha256']
                gr=json.loads(guard_path.read_text())
                assert gr['status']=='finished' and gr['reason'] is None
                assert gr['limits']==dict(reserve_bytes=8*1024**3,rss_limit_bytes=4*1024**3,timeout_seconds=180)
            assert shutil.disk_usage(ROOT).free>=c.STORAGE['full_required_free_bytes']
            external=c.external_source();root.mkdir(exist_ok=False)
            with session_lock(root) as r:
                writer=MetadataWriter(root,(g,r))
                for target,path in ((c.IMPORT_NAME,original_path),(c.CARRIED_NAME,carried_path)):
                    with writer.open_append(target) as stream:stream.write(path.read_bytes())
                plan=dict(kind=c.KIND,status='frozen',root=str(root),external_source=external,
                    original_source_sha256=SOURCE_SHA256,runtime=runtime_identity(),
                    source_import=reference(root,c.IMPORT_NAME),carried_import=reference(root,c.CARRIED_NAME),remaining_seeds=list(c.REMAINING_SEEDS),
                    timing=c.CONTRACT,final=c.FINAL,storage=c.STORAGE,prerequisites=prerequisites)
                writer.publish_json(c.PLAN_NAME,plan);plan_ref=reference(root,c.PLAN_NAME);ctx=c.Recovery(writer,plan_ref)
                assert c.bind(ctx,capacity=True)==imported and c.carried_import(ctx)==carried
                with c.original_session(ctx) as source:assert c.verify_original(ctx,source,imported)==carried
                seen=c.budget(ctx,capacity=True)
                assert set(seen['files'])=={'.lock','control/.metadata.lock',c.IMPORT_NAME,c.CARRIED_NAME,c.PLAN_NAME}
                capacity=dict(free_bytes=shutil.disk_usage(root).free,metadata_bytes=seen['metadata_bytes'],
                    required_remaining_bytes=c.CONTRACT['reserve_bytes']+c.METADATA-seen['metadata_bytes'],checkpoint_allocation_bytes=0)
                with portable.open('xb') as stream:stream.write((root/c.PLAN_NAME).read_bytes())
                assert file_digest(portable)==plan_ref['sha256']
                assert c.external_source()==external and digest(source_fingerprint())==SOURCE_SHA256
            assert all(snapshot(Path(path))==value for path,value in before.items());idle()
except Exception as exc:error=repr(exc)
value=dict(status='passed' if error is None else 'failed',error=error,observed_at=datetime.now(timezone.utc).isoformat(),
    root=str(root),plan=plan_ref,capacity=capacity,guard=reference(ROOT,str(guard.relative_to(ROOT))) if guard.exists() else None,
    portable_plan=reference(ROOT,str(portable.relative_to(ROOT))) if portable.exists() else None,
    procedure_sha256=file_digest(Path(__file__)),
    applicability=dict(original_and_v1_source_unchanged=error is None,closed_roots_unchanged=error is None,
        original_training_source_and_16_protected_checkpoints_reverified=error is None,carried_root_files=27,
        reused_original_functions=['MetadataWriter','run_process','load_pair','expected_models','callbacks','strict_mps',
            'measure_pair','review_result','summarize','analyze'],
        old_bind_budget_entry_produce_or_final_called_on_failed_roots=False,
        source_lock_handoff='Parent releases original root before child; child borrows global/new root locks and independently locks original and historical root for read-only audits. Parent monitor touches only new root.',
        scope='Only new3612/3613 measurements; exact accepted3611 result imported, never repeated. Final assembles all3 with unchanged numerical/cache/timing/statistical/resource rules; new orchestration only.',
        prerequisite_limits='Execution40 plus final7 disposable tests. Four real child processes exercise later-entry/prefix/output review during parent monitor updates. Source audit/GPU/models are doubles in tests; actual source/import/checkpoint applicability is independently verified here.'),
    inference_launched=False,training_launched=False,holdout_scored=False,commit_and_push_before_dispatch=True)
with review.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
print(json.dumps({k:value[k] for k in ('status','error','root','plan','capacity')}))
raise SystemExit(0 if error is None else 1)
