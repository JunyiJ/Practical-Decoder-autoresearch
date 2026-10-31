"""Read-only failure preservation: no inference retry, scoring or checkpoint load."""
import json
from pathlib import Path
from datetime import datetime,timezone
from src.research.protocol import ROOT,digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import bind,idle,session_lock,GLOBAL_LOCK,MetadataWriter,Session,reference,require,require_capacity,prefix,_group_exists
from src.research.tokenizer_activation_retention import verify_boundaries
from src.research.tokenizer_wide_hybrid_metadata import snapshot
base=ROOT/'research/tokenizer-evidence/activation-preparation-001';root=ROOT/'research_runs/tokenizer-activation-001'
guard=base/'campaign001-inference-failure-guard.json';dest=base/'campaign001-inference-failure-portable'
with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):
 with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:
  idle();before=snapshot(root);ledger=Session(MetadataWriter(root,(g,c)),reference(root,'evidence/meta/activation-plan.json'))
  plan=bind(ledger,capacity=True);state=ledger.inspect();prefix(ledger,state);verify_boundaries(ledger,state)
  require(len(state['records'])==12 and state['next']['kind']=='review_final','Complete training evidence required')
  stem='evidence/inference/3511';process=ledger.document(reference(root,stem+'/process/process.json'))
  failure=ledger.document(reference(root,stem+'/dispatch-failure.json'))
  require(process['status']=='failed' and process['returncode']==1 and failure['retry_allowed'] is False,'Exact permanent failure required')
  require(not _group_exists(process['pid']),'Owned inference process group remains live')
  require(not (root/stem/'accepted.json').exists() and not (root/stem/'result.json').exists(),'Unexpected accepted/complete timing result')
  log=(root/stem/'process/worker.log').read_text()
  require('process-inference-3511.json.tmp' in log and 'FileNotFoundError' in log and 'snapshot' in log,'Diagnosis differs from traceback')
  protected={r['path']:r for pair in state['pairs'] for r in pair if (root/r['path']).exists()}
  require(len(protected)==16,'Two selected plus six fresh raw/H200 pairs required')
  for ref in protected.values():ledger.verify(ref)
  copied={}
  for name,entry in before['files'].items():
   if name.startswith('evidence/inference/') or name.startswith('control/') and 'inference' in name:
    require(not entry['group'].endswith('/weights'),'No weight copies');ref=reference(root,name);target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as f:f.write((root/name).read_bytes())
    require(file_digest(target)==ref['sha256'],'Failure copy mismatch');copied[name]=ref
  capacity=require_capacity(ledger);require(snapshot(root)==before,'Audit changed frozen failed campaign')
value=dict(status='training_complete_inference_failed_no_retry',observed_at=datetime.now(timezone.utc).isoformat(),plan=ledger.plan,source_sha256=digest(plan['source']),
 process=process,protected_checkpoints=protected,copied=copied,capacity=capacity,guard=reference(ROOT,str(guard.relative_to(ROOT))),procedure_sha256=file_digest(Path(__file__)),
 diagnosis='The active output census enumerated control/process-inference-3511.json.tmp, then parent atomic replacement removed that temporary name before Path.stat. Worker budget() propagates FileNotFoundError; bounded parent records exit1. No accepted timing result exists.',
 limitations='Traceback establishes the census race failure location; no complete timing samples were published and no latency conclusion is available. Training quality conclusions remain supported by all12 accepted cells. Formal frozen final wrapper remains unavailable because it requires all3 accepted inference pairs.',
 recovery='Do not retry or patch activation001. Preserve its frozen source and failed attempt. Develop a separate versioned inference-only evidence contract with a race-safe bounded census, deterministic concurrency regression tests, exact imported training/selection/retention/checkpoint pins and unchanged timing/quality rules. Freeze that new contract before measurements; do not retrain or regenerate checkpoints.',holdout_scored=False,retry_allowed=False,formal_final_completed=False)
with (base/'campaign001-inference-failure-review.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
print(json.dumps({k:v for k,v in value.items() if k not in ('copied','protected_checkpoints','process')}))
