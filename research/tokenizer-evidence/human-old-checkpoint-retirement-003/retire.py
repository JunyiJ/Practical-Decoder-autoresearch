"""New exact audit of three superseded H100 exports under direct human permission."""
import json,os,shutil
from pathlib import Path
from research.embedding_input_dropout_quality_v1.retention import record,publish,inventory,consumer_check,syncdir
from research.body_allocation_model_v1.evidence import reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
R=Path.cwd();E=Path(__file__).resolve().parent;BASE=R/'research_runs/tokenizer-ema-horizon-v1-001'
def bound(v):
 p=Path(v['path']);assert reference(p)==dict(path=str(p.resolve()),sha256=v['sha256']);return p
with session_lock(GLOBAL_LOCK.parent):
 idle();assert not(E/'manifest.json').exists()
 with CpuPrerequisiteGuard(E/'guard.json',R,timeout_seconds=600,poll_seconds=.25):
  previous=R/'research/tokenizer-evidence/human-old-checkpoint-retirement-002/completed.json'
  assert json.loads(previous.read_text())['deleted_files']==404
  closure=R/'research/tokenizer-evidence/ema-horizon-preparation-001/quality001-closure.json';c=json.loads(closure.read_text());assert c['status']=='closed_not_adopted';bound(c['stop']);review=json.loads(bound(c['review']).read_text());assert review['status']=='closed_not_adopted' and review['quality_adopted'] is False
  bindings=[];eligibility=[reference(previous),reference(closure),c['stop'],c['review']]
  for slot in range(3):
   p=BASE/f'evidence/quality/{slot:02d}/accepted.json';a=json.loads(p.read_text())['artifact_acceptance'];assert a['accepted'] and a['audit']['completed_steps']==1200
   v=a['files']['h100'];assert Path(v['path'])==BASE/f'evidence/quality/{slot:02d}/checkpoints/h100.pt';bound(v)
   bindings.append(record(Path(v['path'])));eligibility.append(reference(p))
  paths=[v['path'] for v in bindings];assert len(set(paths))==3;consumer_check(paths)
  survivors=[record(p) for p in inventory(E) if str(p) not in set(paths)];shards=[]
  for i in range(0,len(survivors),500):
   p=E/f'survivors-{i//500:03d}.json';publish(p,survivors[i:i+500]);shards.append(reference(p))
  publish(E/'manifest.json',dict(source=reference(__file__),authorization='Latest human: clean up disk space as much as possible - ok to delete old checkpoint directly. Explicit exception for these three superseded historical H100 exports; current incumbent/recent body/final-scale models and all original records remain protected.',eligibility=eligibility,delete=bindings,survivors=shards,dependency_closure='Original EMA panel closed/not adopted, current recipe retains H200; future worker requires fresh initialization and does not load historical H100. Completed exact prior404-retirement preserved. No restoration or closed-command replay.'))
  assert set(map(str,inventory(E)))=={v['path'] for v in survivors}|set(paths)
  with (E/'journal.jsonl').open('x') as j:
   for v in bindings:
    consumer_check([v['path']]);p=Path(v['path']);assert record(p)==v;j.write(json.dumps(dict(event='intent_permanent_old_H100_retirement',file=v))+'\n');j.flush();os.fsync(j.fileno());p.unlink();syncdir(p.parent);j.write(json.dumps(dict(event='removed',path=str(p)))+'\n');j.flush();os.fsync(j.fileno())
  assert set(map(str,inventory(E)))=={v['path'] for v in survivors}
  for v in survivors:assert record(Path(v['path']))==v
  for v in eligibility:bound(v)
  result=dict(status='completed',deleted_files=3,logical_removed=sum(v['size'] for v in bindings),allocated_reclaimed=sum(v['allocated'] for v in bindings),survivors_verified=len(survivors),all_current_recent_models_and_original_records_unchanged=True,manifest=reference(E/'manifest.json'),free_after=shutil.disk_usage(R).free)
  publish(E/'completed.json',result);print(json.dumps(result),flush=True)
