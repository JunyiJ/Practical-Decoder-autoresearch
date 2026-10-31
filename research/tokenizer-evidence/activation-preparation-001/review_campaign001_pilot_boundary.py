"""Read-only boundary verification and exact portable nonweight evidence copy."""
import json
from pathlib import Path
from datetime import datetime,timezone
from src.research.protocol import ROOT,digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import bind,idle,session_lock,GLOBAL_LOCK,MetadataWriter,Session,reference,require,require_capacity,prefix
from src.research.tokenizer_activation_retention import verify_boundaries
from src.research.tokenizer_wide_hybrid_metadata import snapshot
base=ROOT/'research/tokenizer-evidence/activation-preparation-001'
root=ROOT/'research_runs/tokenizer-activation-001'
dest=base/'campaign001-pilots-portable'
guard=base/'campaign001-pilot-boundary-review-guard.json'
with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):
    with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:
        idle();before=snapshot(root)
        ledger=Session(MetadataWriter(root,(g,c)),reference(root,'evidence/meta/activation-plan.json'))
        plan=bind(ledger,capacity=True);state=ledger.inspect();prefix(ledger,state);verify_boundaries(ledger,state)
        require(len(state['pilots'])==2 and not state['records'] and state['next']['kind']=='quality','Audited pilot boundary required')
        require(not list(root.rglob('*.pt')),'Only completed pilot weights should have existed')
        copied={}
        for name,entry in before['files'].items():
            require(not entry['group'].endswith('/weights'),'No weight copies')
            ref=reference(root,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:f.write((root/name).read_bytes())
            require(file_digest(target)==ref['sha256'],'Portable copy hash differs');copied[name]=ref
        require(snapshot(root)==before,'Copy changed campaign evidence')
        cap=require_capacity(ledger)
        execution=ledger.document(reference(root,'evidence/.retention/pilot/execution.json'))
value=dict(status='pilot_boundary_verified_ready_for_quality',observed_at=datetime.now(timezone.utc).isoformat(),plan=ledger.plan,
    source_sha256=digest(plan['source']),capacity=cap,execution=execution,portable_directory=str(dest.relative_to(ROOT)),
    copied=copied,guard=reference(ROOT,str(guard.relative_to(ROOT))),procedure_sha256=file_digest(Path(__file__)),
    deleted_weights_cannot_be_reaudited=True,holdout_scored=False,next_action='Commit/push pilot and retention evidence; dispatch only quality slot00 under frozen guards.')
with (base/'campaign001-pilot-boundary-reviewed.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in value.items() if k!='copied'}))
