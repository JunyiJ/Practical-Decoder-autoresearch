"""Read-only reconciliation of both lossless archives and original evidence."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent;OLD=E.parent/'wide-hybrid-preparation-001'
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'logical-evidence-review-guard.json',ROOT,timeout_seconds=600):
 archived={};archives={};receipts={};plans=[]
 for group in ('checkpoint-compaction-001','checkpoint-compaction-002'):
  folder=E.parent/group;completed=json.loads((folder/'completed.json').read_text());plan=json.loads((folder/'plan.json').read_text());plans.append(plan)
  assert completed['status']=='completed' and completed['no_checkpoint_bytes_lost']
  assert json.loads((folder/'apply-guard.json').read_text())['status']=='finished'
  members=set()
  for record in completed['archives']:
   p=ROOT/record['archive'];assert not p.is_symlink() and p.stat().st_size==record['archive_size'] and file_digest(p)==record['archive_sha256']
   pinned=json.loads((folder/(record['group']+'-archive.json')).read_text())
   assert pinned['members']==record['members']==plan['groups'][record['group']] and pinned['verified_before_removal']
   assert pinned['archive_sha256']==record['archive_sha256']
   archives[record['archive']]=dict(size=record['archive_size'],sha256=record['archive_sha256'])
   for n,v in record['members'].items():
    assert n not in archived and not (ROOT/n).exists();archived[n]=v;members.add(n)
  journal=[json.loads(line) for line in (folder/'journal.jsonl').read_text().splitlines()]
  removed=[x['path'] for x in journal if x['event']=='loose_copy_removed']
  assert len(removed)==len(set(removed))==len(members) and set(removed)==members
  receipts[group]={n:file_digest(folder/n) for n in ('plan.json','completed.json','journal.jsonl')}
 checked={}
 def logical(p,sha,size=None):
  n=str(p.relative_to(ROOT))
  if n in archived:v=archived[n]
  else:
   if n not in checked:
    assert p.is_file() and not p.is_symlink();checked[n]=dict(size=p.stat().st_size,sha256=file_digest(p))
   v=checked[n]
  assert v['sha256']==sha and (size is None or v['size']==size)
 counts={}
 for name,review in [('tokenizer-wide-hybrid-002','campaign002-lock-failure-v2-review.json'),('tokenizer-wide-hybrid-003','campaign003-lock-failure-v2-review.json'),('tokenizer-wide-hybrid-004','campaign004-fresh16-failure-review-v2.json')]:
  report=json.loads((OLD/review).read_text());root=ROOT/'research_runs'/name
  for n,v in report['files'].items():logical(root/n,v['sha256'],v['size'])
  counts[name]=len(report['files'])
 for name in ('tokenizer-metadata-contention-baseline-001','tokenizer-metadata-contention-handoff-001','tokenizer-metadata-contention-trace-v3-001'):
  root=ROOT/'research_runs'/name;result=json.loads((root/'result.json').read_text())
  for n,v in result['files'].items():logical(root/n,v['sha256'],v['size'])
  counts[name]=len(result['files'])
 numeric=json.loads((OLD/'rank-numerical-v2-reviewed-001/meta/numerical-trace.json').read_text())
 for n,h in numeric['original_files'].items():logical(Path(n),h)
 counts['completed_numerical_prerequisite']=len(numeric['original_files'])
 for plan in plans:
  for n,v in plan['protected'].items():logical(ROOT/n,v['sha256'],v['size'])
 for n,v in plans[-1]['metadata'].items():logical(ROOT/n,v['sha256'],v['size'])
 assert len(archived)==38 and len(archives)==4
 result=dict(status='logical_original_evidence_preserved_in_verified_archives',original_file_counts=counts,archives=archives,archived_members=archived,receipt_hashes=receipts,
  protected_loose_checkpoint_files_verified=len(plans[-1]['protected']),noncheckpoint_files_verified=len(plans[-1]['metadata']),
  original_reports_modified=False,numerical_reexecuted=False,training=False,holdout_consumed=False,
  qualification='All member hashes verified by full decompression before loose removal; this independent review rehashes unchanged archives and reconciles every logical member with original evidence. Use both manifests or restore exact bytes before original-path-only audits. Selected/fresh/failed weights remain loose.')
 with (E/'logical-evidence-review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=result['status'],original_file_counts=counts,archived_members=len(archived),protected=len(plans[-1]['protected']))),flush=True)
