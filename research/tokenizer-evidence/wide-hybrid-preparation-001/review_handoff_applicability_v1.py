"""Explicit applicability review; preserve executed-source identity and closed runs."""
import datetime,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.tokenizer_wide_hybrid_plan_v2 import source_fingerprint,runtime_identity
from src.research.protocol import digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'checkpoint-handoff-applicability-v1-guard.json',ROOT,timeout_seconds=180):
 original=json.loads((E/'finite-campaign-v2-frozen/meta/plan.json').read_text())
 old=json.loads((E/'control-census-source-applicability-v2-review.json').read_text())
 current=source_fingerprint();runtime=runtime_identity()
 assert runtime==original['runtime'] and digest(runtime)==old['runtime_sha256']
 changes={n:dict(before=original['source'].get(n),after=current.get(n)) for n in set(current)|set(original['source']) if current.get(n)!=original['source'].get(n)}
 checkpoint='src/research/tokenizer_wide_hybrid_checkpoint_writer.py'
 assert set(changes)==set(old['changed_files'])|{checkpoint}
 for n,v in old['changed_files'].items():assert changes[n]==v
 assert file_digest(E/'checkpoint-writer-before-handoff-v1.py')==original['source'][checkpoint]
 review=json.loads((E/'checkpoint-handoff-v1-review.json').read_text())
 assert review['tests_passed']==134 and review['campaign002_all_files_unchanged']
 assert json.loads((E/'checkpoint-handoff-v1-review-guard.json').read_text())['status']=='finished'
 tests=dict(old['test_source_hashes'],**review['test_source_hashes'])
 assert all(file_digest(ROOT/n)==h for n,h in tests.items())
 logs=dict(old['test_logs'],**{'checkpoint-handoff-v1-tests.log':file_digest(E/'checkpoint-handoff-v1-tests.log')})
 assert all(file_digest(E/n)==h for n,h in logs.items())
 original_reports={}
 for n in ('rank-numerical-v2-reviewed-001/meta/numerical.json','training-input-v2-002/meta/training-input.json'):
  path=E/n;value=json.loads(path.read_text())
  assert value['source_sha256']==digest(original['source']) and value['runtime_sha256']==digest(runtime)
  assert all(file_digest(path.parent.parent/k)==h for k,h in value['artifacts'].items())
  original_reports[n]=file_digest(path)
 failed=json.loads((E/'campaign002-lock-failure-v2-review.json').read_text());run=ROOT/'research_runs/tokenizer-wide-hybrid-002'
 assert all(file_digest(run/n)==v['sha256'] for n,v in failed['files'].items())
 protected={n:v for n,v in failed['files'].items() if '/checkpoints/' in n}
 report=dict(kind='wide_hybrid_source_applicability_review_v3',status='numerical_and_input_evidence_applicable_after_census_and_checkpoint_handoff',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  before_source_sha256=digest(original['source']),after_source_sha256=digest(current),runtime_sha256=digest(runtime),changed_files=changes,
  unchanged_source_file_count=len(current)-len(changes),test_source_hashes=tests,test_logs=logs,
  distinct_passing_readiness_tests=26,distinct_passing_entry_tests=40,checkpoint_metadata_process_tests=134,
  original_report_hashes=original_reports,prior_applicability_sha256=file_digest(E/'control-census-source-applicability-v2-review.json'),
  handoff_review_sha256=file_digest(E/'checkpoint-handoff-v1-review.json'),
  protected_campaign002_weight_bytes=sum(v['size'] for v in protected.values()),protected_campaign002_weights=protected,campaign002_all_files_unchanged=True,
  review=['Two earlier census-only readiness/entry corrections remain byte-identical to their reviewed versions.',
   'The only new production change is checkpoint namespace scheduling:20ms outside the lock after accumulated MiB. Tensor payload construction, tensor/RNG/optimizer/EMA values, hashing, writes, tokenizer/config/corpus and runtime remain unchanged.',
   'The completed CPU numerical procedure uses original checkpoint save/read helpers, not this bounded checkpoint writer; its exact-resume and scalar EMA evidence still applies to unchanged mathematical/state logic.',
   'Input encoding has no checkpoint-writer dependency and its original96 train IDs and tokenizer assets remain unchanged.',
   '134 small tests verify serialization roundtrip, exact byte hashing, new publisher handoffs and existing failure/guard cases. New real MPS pilots must verify actual raw/H200/export/resource behavior under the changed output schedule.',
   'Baseline56MiB fixture did not reproduce original timeout; new fixture passed with shorter observed parent wait. Mitigation is not proof no contention failure can occur.',
   'Do not merge or replace old scientific results. A fresh complete panel keeps the same frozen candidate grid/seeds; seed3501 is reused and is not independent replication. Old five-cell results remain descriptive and excluded from new selection. Fresh3511/3512/3513 quality confirmation has not launched in001/002.'],
  numerical_rerun_required=False,input_reencoding_required=False,original_evidence_modified=False,new_campaign_freeze_required=True,
  closed_campaign001_retry_allowed=False,closed_campaign002_retry_allowed=False,
  qualification='Explicit applicability, not a claim that old numerical/input executions used current source. New transport/pilot/retention evidence required; all closed evidence and failed partial remain protected.')
 with (E/'checkpoint-handoff-source-applicability-v1-review.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({k:report[k] for k in ('status','before_source_sha256','after_source_sha256','unchanged_source_file_count','protected_campaign002_weight_bytes')}))
