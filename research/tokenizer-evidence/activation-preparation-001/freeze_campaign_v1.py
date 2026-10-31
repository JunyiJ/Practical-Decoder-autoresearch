"""Freeze one fully reviewed activation campaign; never dispatch training."""
import json
import shutil
from datetime import datetime,timezone

from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_activation_entry import (LIMITS,design,bind,idle,session_lock,GLOBAL_LOCK,
    MetadataWriter,Session,reference,require,require_capacity)

BASE=ROOT/'research/tokenizer-evidence/activation-preparation-001'
CAP=BASE/'capacity-applicability-v1.json'


def main():
    capacity=json.loads(CAP.read_text());root=ROOT/'research_runs/tokenizer-activation-001'
    require(capacity['status']=='passed' and capacity['complete_executable_review'] is True,'Completed audit required')
    require(capacity['source_sha256']==digest(source_fingerprint()) and capacity['runtime']==runtime_identity(),
            'Audited source/runtime changed')
    require(capacity['design_sha256']==digest(design()) and capacity['root']==str(root),'Audited design/root changed')
    require(file_digest(BASE/'freeze_preflight_v1.py')==capacity['procedure_sha256'],'Audit procedure changed')
    require(not root.exists(),'Existing campaign root cannot be reused')
    prerequisites=dict(capacity['prerequisites'],capacity=reference(ROOT,str(CAP.relative_to(ROOT))))
    plan=dict(kind='activation_campaign_v1',status='frozen',root=str(root),source=source_fingerprint(),
        runtime=runtime_identity(),design=design(),limits=LIMITS,inputs=capacity['inputs'],
        prerequisites=prerequisites,storage=capacity['storage'])
    journal=BASE/'freeze-campaign-v1-guard.json'
    with CpuPrerequisiteGuard(journal,ROOT,timeout_seconds=180):
        with session_lock(GLOBAL_LOCK.parent) as g:
            idle()
            require(shutil.disk_usage(ROOT).free>=capacity['storage']['full_required_free_bytes'],'Full panel no longer fits')
            root.mkdir()
            with session_lock(root) as c:
                writer=MetadataWriter(root,(g,c))
                writer.publish_json('evidence/meta/activation-plan.json',plan)
                plan_ref=reference(root,'evidence/meta/activation-plan.json');ledger=Session(writer,plan_ref)
                require(bind(ledger,capacity=True)==plan,'Actual executable plan binding differs')
                observed=require_capacity(ledger)
                require(ledger.inspect()['next']['kind']=='pilot','Fresh campaign not at first pilot')
                require(not list(root.rglob('*.pt')),'Freeze must not create checkpoints')
                saved=BASE/'campaign001-frozen-plan.json'
                with saved.open('xb') as out:out.write((root/plan_ref['path']).read_bytes())
                require(file_digest(saved)==plan_ref['sha256'],'Published plan copy differs')
    guard=json.loads(journal.read_text());require(guard['status']=='finished' and guard['reason'] is None,'Freeze guard incomplete')
    review=dict(status='frozen_ready_for_guarded_pilots',observed_at=datetime.now(timezone.utc).isoformat(),
        root=str(root),plan=reference(ROOT,str((root/plan_ref['path']).relative_to(ROOT))),
        saved_plan=reference(ROOT,str(saved.relative_to(ROOT))),source_sha256=digest(plan['source']),
        capacity=reference(ROOT,str(CAP.relative_to(ROOT))),observed_capacity=observed,
        guard=reference(ROOT,str(journal.relative_to(ROOT))),procedure_sha256=file_digest(__file__),
        finite_trajectories=dict(pilots=2,screens=6,fresh=6),protected_final_pairs=8,
        research_training_launched=False,holdout_scored=False,quality_dispatch_requires_accepted_pilots=True,
        next_action='Commit/push frozen source/protocol/evidence; run exactly one guarded pilot step under both locks with pinned tokenizer cache. Review actual process/dispatch/acceptance before the second pilot; no retry after any failure.')
    with (BASE/'campaign001-freeze-review.json').open('x') as out:json.dump(review,out,indent=2);out.write('\n')
    print(json.dumps(review))



if __name__=='__main__':main()
