"""Exact user-authorized retirement of four obsolete successful decay generated-worker weights."""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R));B=Path(__file__).resolve().parent
S=R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('primitives',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.E=B
from research.decay_campaign_v1.pilot import idle
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
W=R/'research/tokenizer-evidence/decay-preparation-001'
TARGETS={'research_runs/tokenizer-decay-trainer-prerequisite-001/evidence/pilots/01/checkpoints/final_000004_1793581205676113000.pt', 'research_runs/tokenizer-decay-trainer-prerequisite-001/evidence/pilots/00/checkpoints/final_000004_1793581167235194000.pt', 'research_runs/tokenizer-decay-trainer-prerequisite-001/evidence/pilots/01/checkpoints/h200.pt', 'research_runs/tokenizer-decay-trainer-prerequisite-001/evidence/pilots/00/checkpoints/h200.pt'}
def paths():
 out=set(a.all_paths());out.update(p for p in (R/'research_state').rglob('*') if p.is_file() and (p.suffix in ('.pt','.tmp') or p.stat().st_size>=1024**2))
 for p in (R/'research/tokenizer-evidence').rglob('*'):
  if not p.is_relative_to(B) and '__pycache__' not in p.parts and p.is_file():out.add(p)
 return sorted(out)
def audit():
 assert not (B/'manifest.json').exists()
 closure=json.loads((W/'worker-dependencies-closed001.json').read_text());assert closure['status']=='generated_worker_tensor_dependencies_closed'
 integration=json.loads((W/'pilot-integration001.json').read_text());assert integration['status']=='finite_pilot_config_and_parent_bridge_passed'
 eligibility=[a.ref(W/n) for n in ['worker-dependencies-closed001.json','pilot-integration001.json','trainer-worker001.json','trained-inference001.json','retention-boundary001.json']]
 for n,h in integration['sources'].items():assert a.file_digest(R/n)==h
 delete=[]
 worker=json.loads((W/'trainer-worker001.json').read_text());inf=json.loads((W/'trained-inference001.json').read_text())
 assert worker['status']=='real_bounded_worker_parent_acceptance_passed' and inf['status']=='generated_trained_decay_pair_inference_passed'
 assert len(worker['arms'])==2 and len(integration['parent_reaudits'])==2
 assert {r['path'] for r in closure['eligible_successful_generated_weights']}=={v['binding'][k]['path'] for v in worker['arms'] for k in ('parent','h200')}
 for row in worker['arms']:
  accepted=a.bound(row['accepted']);assert accepted['acceptance']['accepted'] and row['acceptance']['audit']==dict(completed_steps=4,tokens_processed=4096,ema_parameters=88,ema_snapshots=3)
  assert row['independent_parent_validation_exact'] and any(x['arm']==row['arm'] and x['validation_exact'] for x in integration['parent_reaudits'])
  eligibility.extend([row['accepted'],row['worker'],row['process'],row['parent_guard']])
  for kind in ('parent','h200'):
   path=a.verify(row['binding'][kind]);record=a.record(path)
   assert path.is_relative_to(R/'research_runs/tokenizer-decay-trainer-prerequisite-001/evidence/pilots')
   assert path.suffix=='.pt' and str(path.relative_to(R)) in TARGETS
   delete.append(dict(**record,kind='closed_successful_generated_worker',original_acceptance=row['accepted']))
 assert len(delete)==4 and sum(x['size'] for x in delete)==1118318420
 assert {str(Path(x['path']).relative_to(R)) for x in delete}==TARGETS
 remove={x['path'] for x in delete};survivors=[a.record(p) for p in paths() if str(p) not in remove];shards=[]
 for start in range(0,len(survivors),500):
  p=B/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(B/'manifest.json',dict(status='audited',authorization='Human authorized obsolete checkpoint retirement while keeping selected models and fresh confirmations; these four successful generated prerequisite weights are obsolete after tested dependency closure.',source=a.ref(Path(__file__)),primitives=a.ref(S),policy=a.ref(POLICY),eligibility=eligibility,delete_files=delete,inventory_shards=shards,deleted_bytes=sum(r['size'] for r in delete),free_bytes_before=shutil.disk_usage(R).free,
  dependency_review='Both generated arms passed worker/parent validation, final H200, trained inference and additive pilot payload/cache integration. Dependencies closed; pilot config reads retained metadata only. Selected/fresh/failed and all historical evidence remain untouched.',limitation='Permanent retirement of only four successful generated prerequisite payloads. Original receipts remain immutable; no restoration or prerequisite rerun.'))
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
 result=dict(status='completed',manifest=a.ref(B/'manifest.json'),deleted_bytes=m['deleted_bytes'],deleted_paths=sorted(remove),retained_files_verified=len(survivors),selected_fresh_failed_unchanged=True,original_evidence_unchanged=True,free_bytes_after=shutil.disk_usage(R).free,required_free_bytes=12381585408,training_launched=False)
 a.publish(B/'completed.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('audit','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'):
  idle()
  with a.CpuPrerequisiteGuard(B/(mode+'-guard.json'),R,timeout_seconds=900):audit() if mode=='audit' else apply()
