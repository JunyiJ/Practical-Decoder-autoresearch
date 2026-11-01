"""Exact user-authorized retirement of three obsolete successful numerical archives."""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R));B=Path(__file__).resolve().parent
S=R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('primitives',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.E=B
from research.ffn_norm_campaign_v1.pilot import idle
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
OLD=R/'research/tokenizer-evidence/checkpoint-compaction-001'
SYN=R/'research/tokenizer-evidence/mla-score-preparation-001/synthetic-archive001'
TARGETS={'research_archives/checkpoint-compaction-001/numerical.tar.zst','research_archives/mla-score-synthetic-001/score.tar.zst','research_archives/mla-score-synthetic-001/ffn.tar.zst'}
def paths():
 out=set(a.all_paths());out.update(p for p in (R/'research_state').rglob('*.pt') if p.is_file())
 for n in subprocess.check_output(['git','ls-files','research/tokenizer-evidence'],cwd=R,text=True).splitlines():
  p=R/n
  if not p.is_relative_to(B) and p.is_file():out.add(p)
 return sorted(out)
def audit():
 assert not (B/'manifest.json').exists()
 numerical=json.loads((OLD/'numerical-archive.json').read_text());terminal=R/'research/tokenizer-evidence/wide-hybrid-preparation-001/rank-numerical-v2-terminal-review.json'
 review=json.loads(terminal.read_text());assert review['status']=='passed' and len(review['cases'])==9 and all(x['exact_state_hash_match'] for x in review['cases'])
 assert numerical['verified_before_removal'] and len(numerical['members'])==9
 assert set(numerical['members'])=={f'research_runs/tokenizer-wide-hybrid-numerical-v2-001/{i:02d}/partial.pt' for i in range(9)}
 original=json.loads((OLD/'plan.json').read_text());assert original['groups']['numerical']==numerical['members']
 synthetic=json.loads((SYN/'completed.json').read_text());assert synthetic['status']=='completed' and synthetic['archived_files']==12 and synthetic['selected_fresh_failure_weights_unchanged']
 eligibility=[a.ref(terminal),a.ref(OLD/'numerical-archive.json'),a.ref(OLD/'plan.json'),a.ref(SYN/'completed.json'),a.ref(SYN/'plan.json'),a.ref(R/'research/ffn_norm_campaign_v1/pilot.py')]
 sp=json.loads((SYN/'plan.json').read_text())
 for r in sp['eligibility']:
  d=a.bound(r);assert 'pass' in d['status'];eligibility.append(r)
 archives=[dict(archive=dict(path=str(R/numerical['archive']),sha256=numerical['archive_sha256']),archive_size=numerical['archive_size'],members=numerical['members'])]+synthetic['archives']
 delete=[]
 for item in archives:
  path=a.verify(item['archive']);row=a.record(path);assert row['size']==item['archive_size'];assert str(path.relative_to(R)) in TARGETS
  for n,v in item['members'].items():
   assert not (R/n).exists() and n.endswith('.pt') and len(v['sha256'])==64
   assert n.startswith(('research_runs/tokenizer-wide-hybrid-numerical-v2-001/','research_runs/tokenizer-mla-score-checkpoint-prerequisite-001/','research_runs/tokenizer-ffn-structure-checkpoint-prerequisite-001/'))
  delete.append(dict(**row,members=item['members']))
 assert {str(Path(x['path']).relative_to(R)) for x in delete}==TARGETS
 remove={x['path'] for x in delete};survivors=[a.record(p) for p in paths() if str(p) not in remove];shards=[]
 for start in range(0,len(survivors),500):
  p=B/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(B/'manifest.json',dict(status='audited',authorization='Human explicitly authorized deleting obsolete numerical checkpoint archives, preserving selected models and fresh confirmations; failed artifacts remain protected.',source=a.ref(Path(__file__)),primitives=a.ref(S),policy=a.ref(POLICY),eligibility=eligibility,delete_files=delete,inventory_shards=shards,deleted_bytes=sum(r['size'] for r in delete),free_bytes_before=shutil.disk_usage(R).free,
  dependency_review='Current FFN-product-norm pilot imports original source and reads preserved success receipts/configs only. These three closed older numerical weight sets have no remaining live loading dependency. No selected/fresh/failed weights or active work.',limitation='Permanent retirement of21 successful numerical checkpoint payloads; source, original audit/metric/hash evidence retained. Original archive restoration instructions superseded only for exact three paths.'))
 print(json.dumps(dict(status='audited',bytes=sum(r['size'] for r in delete),survivors=len(survivors))),flush=True)
def apply():
 m=json.loads((B/'manifest.json').read_text());assert json.loads((B/'audit-guard.json').read_text())['status']=='finished'
 for r in [m['source'],m['primitives'],m['policy'],*m['eligibility']]:a.verify(r)
 survivors=[r for shard in m['inventory_shards'] for r in a.bound(shard)];remove={r['path'] for r in m['delete_files']}
 assert remove=={str(R/n) for n in TARGETS} and set(map(str,paths()))=={r['path'] for r in survivors}|remove
 for r in survivors:
  s=Path(r['path']).stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(r['device'],r['inode'],r['size'],r['mtime_ns'])
 for row in m['delete_files']:a.check(Path(row['path']),row)
 for row in m['delete_files']:
  a.check(Path(row['path']),row);a.journal(dict(event='intent_permanent_retirement',file=row));Path(row['path']).unlink();a.syncdir(Path(row['path']).parent);a.journal(dict(event='removed',path=row['path']))
 assert set(map(str,paths()))=={r['path'] for r in survivors}
 for r in survivors:assert a.record(Path(r['path']))==r
 result=dict(status='completed',manifest=a.ref(B/'manifest.json'),deleted_bytes=m['deleted_bytes'],deleted_archives=sorted(remove),retained_files_verified=len(survivors),selected_fresh_failed_unchanged=True,original_evidence_unchanged=True,free_bytes_after=shutil.disk_usage(R).free,required_free_bytes=13555990528,training_launched=False)
 a.publish(B/'completed.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('audit','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'):
  idle()
  with a.CpuPrerequisiteGuard(B/(mode+'-guard.json'),R,timeout_seconds=900):audit() if mode=='audit' else apply()
