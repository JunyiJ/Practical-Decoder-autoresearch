"""Independent read-only verification after freezer receipt-publication failure.

Never reruns the freezer, modifies its plan, starts a worker or loads a checkpoint.
"""
import json
from pathlib import Path
from datetime import datetime,timezone

from src.research.protocol import ROOT,digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import (bind,idle,session_lock,GLOBAL_LOCK,MetadataWriter,
    Session,reference,require,require_capacity)
from src.research.tokenizer_wide_hybrid_metadata import snapshot

BASE=ROOT/'research/tokenizer-evidence/activation-preparation-001'


def main():
    root=ROOT/'research_runs/tokenizer-activation-001';journal=BASE/'freeze-independent-v1-guard.json'
    failure_path=BASE/'freeze-campaign-v1-publication-failure.json';failure=json.loads(failure_path.read_text())
    require(file_digest(BASE/'freeze_campaign_v1.py')==failure['procedure_sha256'],'Original failed procedure changed')
    require(file_digest(BASE/'freeze-campaign-v1-guard.json')==failure['guard_sha256'],'Original guard changed')
    original_guard=json.loads((BASE/'freeze-campaign-v1-guard.json').read_text())
    require(original_guard['status']=='finished' and original_guard['reason'] is None,'Original freeze did not complete guarded work')
    with CpuPrerequisiteGuard(journal,ROOT,timeout_seconds=180):
        with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:
            idle();before=snapshot(root);writer=MetadataWriter(root,(g,c))
            plan_ref=reference(root,'evidence/meta/activation-plan.json');ledger=Session(writer,plan_ref)
            require(plan_ref['sha256']==failure['plan_sha256']==failure['saved_plan_sha256']
                ==file_digest(BASE/'campaign001-frozen-plan.json'),'Original and saved plan identities differ')
            plan=bind(ledger,capacity=True);capacity=require_capacity(ledger);state=ledger.inspect()
            require(state['next']['kind']=='pilot' and len(state['events'])==0 and not list(root.rglob('*.pt')),
                    'Campaign must still be empty before first pilot')
            require(snapshot(root)==before,'Independent verification changed campaign contents')
    guard=json.loads(journal.read_text());require(guard['status']=='finished' and guard['reason'] is None,'Independent guard incomplete')
    value=dict(status='frozen_ready_for_guarded_pilots',observed_at=datetime.now(timezone.utc).isoformat(),
        root=str(root),plan=reference(ROOT,str((root/plan_ref['path']).relative_to(ROOT))),
        saved_plan=reference(ROOT,str((BASE/'campaign001-frozen-plan.json').relative_to(ROOT))),
        source_sha256=digest(plan['source']),capacity=plan['prerequisites']['capacity'],observed_capacity=capacity,
        guard=reference(ROOT,str(journal.relative_to(ROOT))),procedure_sha256=file_digest(Path(__file__)),
        preserved_publication_failure=reference(ROOT,str(failure_path.relative_to(ROOT))),
        review_scope='Independent actual executable source/runtime/design/input/prerequisite/capacity bind and empty immutable campaign census; original freeze was not repeated.',
        finite_trajectories=dict(pilots=2,screens=6,fresh=6),protected_final_pairs=8,
        research_training_launched=False,holdout_scored=False,quality_dispatch_requires_accepted_pilots=True,
        next_action='Commit/push frozen source/protocol/evidence, then one guarded pilot step under both locks with pinned tokenizer cache. Inspect actual process/dispatch/acceptance before second pilot; no failed attempt retry.')
    with (BASE/'campaign001-freeze-review.json').open('x') as out:json.dump(value,out,indent=2);out.write('\n')
    print(json.dumps(value))


if __name__=='__main__':main()
