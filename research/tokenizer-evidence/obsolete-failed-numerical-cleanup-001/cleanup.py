"""Exact user-authorized retirement of eleven obsolete closed failed numerical weights."""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R));B=Path(__file__).resolve().parent
S=R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('primitives',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.E=B
from research.warmup_campaign_v1.pilot import idle
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
W=R/'research/tokenizer-evidence/warmup-preparation-001'
TARGETS={'research_runs/tokenizer-warmup-model-prerequisite-001/warmup100/h200.pt', 'research_runs/tokenizer-local-mix-resume-001/causal3/step2.pt', 'research_runs/tokenizer-warmup-model-prerequisite-001/warmup100/step1.pt', 'research_runs/tokenizer-ffn-structure-trainer-prerequisite-002/swiglu/h200.pt', 'research_runs/tokenizer-ffn-structure-trainer-prerequisite-002/swiglu/final_000002_1793504898478490000.pt', 'research_runs/tokenizer-block-norm-v2-pilots-001/evidence/pilots/00/checkpoints/final_000020_1793485270801090000.pt', 'research_runs/tokenizer-warmup-model-prerequisite-001/none/step1.pt', 'research_runs/tokenizer-warmup-model-prerequisite-001/warmup100/final.pt', 'research_runs/tokenizer-warmup-model-prerequisite-001/none/final.pt', 'research_runs/tokenizer-warmup-model-prerequisite-001/none/h200.pt', 'research_runs/tokenizer-block-norm-v2-pilots-001/evidence/pilots/00/checkpoints/h200.pt'}
def paths():
 out=set(a.all_paths());out.update(p for p in (R/'research_state').rglob('*.pt') if p.is_file())
 for p in (R/'research/tokenizer-evidence').rglob('*'):
  if not p.is_relative_to(B) and '__pycache__' not in p.parts and p.is_file():out.add(p)
 return sorted(out)
def audit():
 assert not (B/'manifest.json').exists()
 W=R/'research/tokenizer-evidence/warmup-preparation-001';L=R/'research/tokenizer-evidence/local-mix-preparation-001';F=R/'research/tokenizer-evidence/ffn-structure-preparation-001'
 records={};eligibility=[]
 def bind(p):
  eligibility.append(a.ref(p));return json.loads(p.read_text())
 warm=bind(W/'model-receipt-recovery001.json');assert warm['status']=='recovered_full_model_rates_short_repeat_resume_h200_passed'
 bind(W/'model-resume001-failure.json');assert bind(R/'research_runs/tokenizer-warmup-model-prerequisite-001/STOP')['no_retry']
 for row in warm['arms']:
  for n,v in row['files'].items():
   if n.endswith('.pt'):records['research_runs/tokenizer-warmup-model-prerequisite-001/'+row['arm']+'/'+n]=v
 failed=bind(F/'trainer-worker002-failure-decision.json');assert failed['status']=='closed_failed_validation_after_training'
 records.update(failed['weights'])
 failure=bind(L/'model-resume001-failure.json');assert failure['status']=='failed'
 a.verify(failure['runner']);a.verify(failure['trace']);eligibility.extend([failure['runner'],failure['trace']])
 assert bind(R/'research_runs/tokenizer-local-mix-resume-001/STOP')['retry_allowed'] is False
 for line in (L/'model-resume001-trace.jsonl').read_text().splitlines():
  row=json.loads(line)
  if row['event']=='saved':records[str(Path(row['path']).relative_to(R))]=dict(size=row['bytes'],sha256=row['sha256'])
 P=R/'research_runs/tokenizer-block-norm-v2-pilots-001'
 assert bind(P/'STOP')['status']=='closed_failed_pilot001'
 export=bind(P/'evidence/pilots/00/export.json')
 for p,h in [(export['checkpoint'],export['checkpoint_sha256']),(export['parent_checkpoint']['path'],export['parent_checkpoint']['sha256'])]:records[str(Path(p).relative_to(R))]=dict(size=Path(p).stat().st_size,sha256=h)
 current=bind(L/'trainer-worker001-source-plan.json');a.verify(current['runner']);eligibility.append(current['runner'])
 for n,h in current['sources'].items():assert a.file_digest(R/n)==h
 assert current['scope'].startswith('Fresh final-horizon worker pair; no loading of failed checkpoint')
 # No active consumer: new runner creates a new corpus/config/weights and reads only
 # model001 metadata. Historical closure commands/fixtures must never be rerun.
 assert set(records)==TARGETS
 delete=[]
 for n in sorted(TARGETS):
  p=R/n;a.check(p,records[n]);delete.append(dict(**a.record(p),kind='obsolete_closed_failed_numerical_checkpoint'))
 remove={x['path'] for x in delete};survivors=[a.record(p) for p in paths() if str(p) not in remove];shards=[]
 for start in range(0,len(survivors),500):
  p=B/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(B/'manifest.json',dict(status='audited',authorization='Latest human explicitly authorizes deleting old failed numerical checkpoints no longer needed, keeping selected models and fresh confirmations. Narrow exception for these11 exact paths only.',source=a.ref(Path(__file__)),primitives=a.ref(S),policy=a.ref(POLICY),eligibility=eligibility,delete_files=delete,inventory_shards=shards,deleted_bytes=sum(r['size'] for r in delete),free_bytes_before=shutil.disk_usage(R).free,
 dependency_review='All four failed numerical roots are closed. Warmup and FFN/block-norm campaigns closed. New local-mix final-horizon worker uses new generated corpus/weights and preserved model001 receipts; it never loads these tensors. Preserve all failure source/config/log/trace/guard/receipt evidence and all selected/fresh/other failed artifacts.',limitation='Permanent deletion of the11 specified failed numerical tensors only. Prior retain-failed language superseded by current human request for these proved-obsolete tensors; no closed fixture may be rerun and retired bytes cannot be restored.'))
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
