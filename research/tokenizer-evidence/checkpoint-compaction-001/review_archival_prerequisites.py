"""Verify archived logical evidence against original frozen file identities."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent;OLD=ROOT/'research/tokenizer-evidence/wide-hybrid-preparation-001'
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'logical-evidence-review-guard.json',ROOT,timeout_seconds=600):
 completed=json.loads((E/'completed.json').read_text());plan=json.loads((E/'plan.json').read_text())
 assert completed['status']=='completed' and completed['no_checkpoint_bytes_lost']
 assert json.loads((E/'apply-guard.json').read_text())['status']=='finished'
 archived={};archives={}
 for record in completed['archives']:
  path=ROOT/record['archive'];assert not path.is_symlink() and path.stat().st_size==record['archive_size'] and file_digest(path)==record['archive_sha256']
  pinned=json.loads((E/(record['group']+'-archive.json')).read_text())
  assert pinned['members']==record['members'] and pinned['verified_before_removal']
  assert pinned['archive_sha256']==record['archive_sha256']
  assert plan['groups'][record['group']]==record['members']
  archives[record['archive']]=dict(size=record['archive_size'],sha256=record['archive_sha256'])
  for n,v in record['members'].items():assert n not in archived and not (ROOT/n).exists();archived[n]=v
 journal=[json.loads(line) for line in (E/'journal.jsonl').read_text().splitlines()]
 removed=[x['path'] for x in journal if x['event']=='loose_copy_removed'];assert len(removed)==len(set(removed))==35 and set(removed)==set(archived)
 def logical(path,sha,size=None):
  relative=str(path.relative_to(ROOT))
  if relative in archived:
   assert archived[relative]['sha256']==sha and (size is None or archived[relative]['size']==size)
  else:
   assert path.is_file() and not path.is_symlink() and file_digest(path)==sha
   assert size is None or path.stat().st_size==size
 counts={}
 for name,review in [('tokenizer-wide-hybrid-002','campaign002-lock-failure-v2-review.json'),('tokenizer-wide-hybrid-003','campaign003-lock-failure-v2-review.json')]:
  report=json.loads((OLD/review).read_text());root=ROOT/'research_runs'/name
  for n,v in report['files'].items():logical(root/n,v['sha256'],v['size'])
  counts[name]=len(report['files'])
 numeric=json.loads((OLD/'rank-numerical-v2-reviewed-001/meta/numerical-trace.json').read_text())
 for n,h in numeric['original_files'].items():logical(Path(n),h)
 counts['completed_numerical_prerequisite']=len(numeric['original_files'])
 for n,v in plan['protected'].items():logical(ROOT/n,v['sha256'],v['size'])
 assert len(archived)==35
 result=dict(status='logical_original_evidence_preserved_in_verified_archives',original_file_counts=counts,archives=archives,archived_members=archived,
  protected_loose_checkpoint_files_verified=len(plan['protected']),original_reports_modified=False,numerical_reexecuted=False,training=False,
  archive_manifest_sha256=file_digest(E/'plan.json'),completion_receipt_sha256=file_digest(E/'completed.json'),journal_sha256=file_digest(E/'journal.jsonl'),
  qualification='Archived member hashes were verified by full decompression before loose-copy removal; this independent review rehashed unchanged archives and reconciled every logical member with original frozen evidence. Original-path-only validators require exact restoration or this explicit archive resolver; they must not be rerun assuming loose files remain.')
 with (E/'logical-evidence-review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(status=result['status'],original_file_counts=counts,archived_members=len(archived),protected=len(plan['protected']))),flush=True)
