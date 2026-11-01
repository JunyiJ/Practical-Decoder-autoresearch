"""Exact user-authorized retirement of three obsolete incomplete numerical checkpoint writes."""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R));B=Path(__file__).resolve().parent
S=R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('primitives',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.E=B
from research.warmup_campaign_v1.pilot import idle
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
W=R/'research/tokenizer-evidence/warmup-preparation-001'
TARGETS={'research_runs/tokenizer-metadata-contention-trace-v2-001/campaign/evidence/quality/23/checkpoints/fixture.pt.tmp', 'research_runs/tokenizer-wide-hybrid-002/evidence/quality/05/checkpoints/final_001200_1793315322602681000.pt.tmp', 'research_runs/tokenizer-wide-hybrid-003/evidence/quality/08/checkpoints/final_001200_1793325864452219000.pt.tmp'}
def paths():
 out=set(a.all_paths());out.update(p for p in (R/'research_state').rglob('*.pt') if p.is_file())
 for p in (R/'research/tokenizer-evidence').rglob('*'):
  if not p.is_relative_to(B) and '__pycache__' not in p.parts and p.is_file():out.add(p)
 return sorted(out)
def audit():
 assert not (B/'manifest.json').exists()
 H=R/'research/tokenizer-evidence/wide-hybrid-preparation-001';L=R/'research/tokenizer-evidence/local-mix-preparation-001'
 records={};eligibility=[]
 for root,review,key in [('tokenizer-metadata-contention-trace-v2-001', 'metadata-contention-trace-v2-review.json', 'protected_partials'), ('tokenizer-wide-hybrid-002', 'campaign002-lock-failure-v2-review.json', 'failed_partials'), ('tokenizer-wide-hybrid-003', 'campaign003-lock-failure-v2-review.json', 'failed_partials')]:
  p=H/review;eligibility.append(a.ref(p));d=json.loads(p.read_text())
  assert 'closed' in d['status']
  assert d.get('retry_allowed',d.get('retries_allowed')) is False
  for n,v in d[key].items():
   assert n.endswith('.pt.tmp')
   if key == 'failed_partials':
    assert d['terminal']['status']=='failed_closed' and d['terminal']['slot']==len(d['accepted_cells'])
    assert f"/quality/{d['terminal']['slot']:02d}/" in n
   records['research_runs/'+root+'/'+n]=v
 current=json.loads((L/'trainer-worker001-source-plan.json').read_text());eligibility.append(a.ref(L/'trainer-worker001-source-plan.json'))
 a.verify(current['runner']);eligibility.append(current['runner'])
 for n,h in current['sources'].items():assert a.file_digest(R/n)==h
 assert set(records)==TARGETS
 delete=[]
 for n in sorted(TARGETS):
  p=R/n;a.check(p,records[n]);delete.append(dict(**a.record(p),kind='obsolete_incomplete_failed_checkpoint_write'))
 remove={x['path'] for x in delete};survivors=[a.record(p) for p in paths() if str(p) not in remove];shards=[]
 for start in range(0,len(survivors),500):
  p=B/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(B/'manifest.json',dict(status='audited',authorization='Latest human explicitly authorizes deleting old failed numerical checkpoints no longer needed, keeping selected models and fresh confirmations. Narrow exception for these3 exact partial-write paths only.',source=a.ref(Path(__file__)),primitives=a.ref(S),policy=a.ref(POLICY),eligibility=eligibility,delete_files=delete,inventory_shards=shards,deleted_bytes=sum(r['size'] for r in delete),free_bytes_before=shutil.disk_usage(R).free,
 dependency_review='One actual-scale synthetic serializer diagnostic and two incomplete failed screen checkpoint writes are closed with retries forbidden. These partial bytes are not accepted endpoints, selected models, or fresh confirmations. Current local-mix workers have no dependency on them. Preserve every original failure trace/source/review and all accepted scientific weights.',limitation='Permanent retirement of three exact incomplete checkpoint writes; latest human authorization for obsolete failed numerical checkpoints supersedes their earlier partial-retention wording. No selected/fresh weights removed. No restoration or retry.'))
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
 for r in [m['source'],m['primitives'],m['policy'],*m['eligibility']]:a.verify(r)
 result=dict(status='completed',manifest=a.ref(B/'manifest.json'),deleted_bytes=m['deleted_bytes'],deleted_paths=sorted(remove),retained_files_verified=len(survivors),selected_fresh_and_nontarget_failures_unchanged=True,original_evidence_unchanged=True,free_bytes_after=shutil.disk_usage(R).free,required_free_bytes=10066329600,training_launched=False)
 a.publish(B/'completed.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('audit','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'):
  idle()
  with a.CpuPrerequisiteGuard(B/(mode+'-guard.json'),R,timeout_seconds=900):audit() if mode=='audit' else apply()
