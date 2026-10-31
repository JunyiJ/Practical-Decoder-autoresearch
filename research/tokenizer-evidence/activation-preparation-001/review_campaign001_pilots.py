"""Read-only locked acceptance/capacity review of both real activation pilots."""
import json
from pathlib import Path
from datetime import datetime, timezone
from src.research.protocol import ROOT, digest, file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import bind, idle, session_lock, GLOBAL_LOCK, MetadataWriter, Session, reference, require, require_capacity, prefix
from src.research.tokenizer_wide_hybrid_metadata import snapshot

base=ROOT/'research/tokenizer-evidence/activation-preparation-001'
root=ROOT/'research_runs/tokenizer-activation-001'
guard_path=base/'campaign001-pilots-review-guard.json'
with CpuPrerequisiteGuard(guard_path,ROOT,timeout_seconds=180):
    with session_lock(GLOBAL_LOCK.parent) as g, session_lock(root) as c:
        idle(); before=snapshot(root)
        ledger=Session(MetadataWriter(root,(g,c)),reference(root,'evidence/meta/activation-plan.json'))
        plan=bind(ledger,capacity=True); state=ledger.inspect(); dispatches=prefix(ledger,state)
        require(len(state['pilots'])==2 and not state['records'] and state['pending'] is None
                and state['next']['kind']=='review_pilot_boundary','Both accepted pilots and no quality required')
        rows=[]
        for i,family in enumerate(('silu','gelu_exact')):
            stem=f'evidence/pilots/{i:02d}'
            def read(name): return ledger.document(reference(root,stem+'/'+name))
            process=read('process/process.json'); resources=read('resources-audit.json')
            train=read('training-summary.json'); measured=read('measurements.json')
            rows.append(dict(family=family,completed_steps=train['completed_steps'],elapsed_seconds=process['elapsed_seconds'],
                training_seconds=train['training_time_sec'],resources=resources['detail'],cache=measured['cache'],
                raw_ce=measured['raw']['mean_ce'],h200_ce=measured['h200']['mean_ce'],dispatch=dispatches[i]))
        capacity=require_capacity(ledger)
        require(snapshot(root)==before,'Read-only pilot review changed campaign')
        inventory={name:reference(root,name) for name,entry in before['files'].items() if not entry['group'].endswith('/weights')}
value=dict(status='both_pilots_accepted',observed_at=datetime.now(timezone.utc).isoformat(),plan=ledger.plan,
    source_sha256=digest(plan['source']),pilots=rows,capacity=capacity,non_checkpoint_inventory=inventory,
    guard=reference(ROOT,str(guard_path.relative_to(ROOT))),procedure_sha256=file_digest(Path(__file__)),
    holdout_scored=False,quality_conclusion=False,next_action='Execute frozen audited pilot retention boundary before quality slot00.')
with (base/'campaign001-pilots-reviewed.json').open('x') as f: json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in value.items() if k!='non_checkpoint_inventory'}))
