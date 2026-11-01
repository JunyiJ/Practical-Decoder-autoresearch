"""Exact user-authorized retirement of eight obsolete numerical archives and four completed generated-worker weights."""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R));B=Path(__file__).resolve().parent
S=R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('primitives',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.E=B
from research.warmup_campaign_v1.pilot import idle
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
W=R/'research/tokenizer-evidence/warmup-preparation-001'
TARGETS={'research_archives/ffn-norm-historical-numerical-001/successful-numerical.tar.zst', 'research_runs/tokenizer-warmup-trainer-prerequisite-001/evidence/pilots/00/checkpoints/h200.pt', 'research_archives/ffn-norm-prerequisites-001/resume.tar.zst', 'research_runs/tokenizer-warmup-trainer-prerequisite-001/evidence/pilots/01/checkpoints/final_000004_1793560303643000000.pt', 'research_archives/ffn-norm-prerequisites-001/h200.tar.zst', 'research_runs/tokenizer-warmup-trainer-prerequisite-001/evidence/pilots/01/checkpoints/h200.pt', 'research_archives/ffn-norm-prerequisites-001/trainer.tar.zst', 'research_archives/mla-score-worker-001/successful-workers.tar.zst', 'research_runs/tokenizer-warmup-trainer-prerequisite-001/evidence/pilots/00/checkpoints/final_000004_1793560264951415000.pt', 'research_archives/ffn-norm-worker-001/rms.tar.zst', 'research_archives/ffn-norm-worker-001/none.tar.zst', 'research_archives/ffn-norm-prerequisites-001/transport.tar.zst'}
def paths():
 out=set(a.all_paths());out.update(p for p in (R/'research_state').rglob('*.pt') if p.is_file())
 for p in (R/'research/tokenizer-evidence').rglob('*'):
  if not p.is_relative_to(B) and '__pycache__' not in p.parts and p.is_file():out.add(p)
 return sorted(out)
def audit():
 assert not (B/'manifest.json').exists()
 closure=json.loads((W/'worker-dependencies-closed001.json').read_text());assert closure['status']=='generated_worker_tensor_dependencies_closed'
 integration=json.loads((W/'pilot-integration001.json').read_text());assert integration['status']=='finite_pilot_config_and_parent_bridge_passed'
 eligibility=[a.ref(W/n) for n in ['worker-dependencies-closed001.json','pilot-integration001.json','trainer-worker001.json','trained-inference001.json','retention-boundary001.json']]
 for n,h in integration['sources'].items():assert a.file_digest(R/n)==h
 roots=[('ffn-norm-preparation-001/historical-numerical-archive001',220),('ffn-norm-preparation-001/archive001',14),('ffn-norm-preparation-001/worker-archive001',4),('mla-score-preparation-001/worker-archive001',6)]
 delete=[]
 for folder,count in roots:
  directory=R/'research/tokenizer-evidence'/folder
  completed=json.loads((directory/'completed.json').read_text());original=a.bound(completed['plan'])
  assert completed['status']=='completed' and completed['archived_files']==count and completed['selected_fresh_failed_unchanged']
  eligibility.extend([a.ref(directory/'completed.json'),completed['plan']])
  for r in original['eligibility']:a.verify(r);eligibility.append(r)
  expected=original.get('files') or {n:v for panel in original['panels'] for n,v in panel['files'].items()}
  archives=completed.get('archives') or [completed['archive']]
  assert {n:v for item in archives for n,v in item['members'].items()}==expected and len(expected)==count
  for item in archives:
   assert item['verified_before_removal']
   path=a.verify(item['archive']);row=a.record(path);assert row['size']==item['archive_size'] and str(path.relative_to(R)) in TARGETS
   for n,v in item['members'].items():assert not (R/n).exists() and n.endswith('.pt') and len(v['sha256'])==64
   delete.append(dict(**row,members=item['members'],kind='obsolete_successful_numerical_archive'))
 worker=json.loads((W/'trainer-worker001.json').read_text());inf=json.loads((W/'trained-inference001.json').read_text())
 assert worker['status']=='real_bounded_worker_parent_acceptance_passed' and inf['status']=='generated_trained_warmup_pair_inference_passed'
 assert len(worker['arms'])==2 and len(integration['parent_reaudits'])==2
 assert {r['path'] for r in closure['eligible_successful_generated_weights']}=={v['binding'][k]['path'] for v in worker['arms'] for k in ('parent','h200')}
 for row in worker['arms']:
  accepted=a.bound(row['accepted']);assert accepted['acceptance']['accepted'] and row['acceptance']['audit']==dict(completed_steps=4,tokens_processed=4096,ema_parameters=88,ema_snapshots=3)
  assert row['independent_parent_validation_exact'] and any(x['arm']==row['arm'] and x['validation_exact'] for x in integration['parent_reaudits'])
  eligibility.extend([row['accepted'],row['worker'],row['process'],row['parent_guard']])
  for kind in ('parent','h200'):
   path=a.verify(row['binding'][kind]);record=a.record(path)
   assert path.is_relative_to(R/'research_runs/tokenizer-warmup-trainer-prerequisite-001/evidence/pilots')
   assert path.suffix=='.pt' and str(path.relative_to(R)) in TARGETS
   delete.append(dict(**record,kind='closed_successful_generated_worker',original_acceptance=row['accepted']))
 assert len(delete)==12 and sum(x['size'] for x in delete)==4691175838
 assert {str(Path(x['path']).relative_to(R)) for x in delete}==TARGETS
 remove={x['path'] for x in delete};survivors=[a.record(p) for p in paths() if str(p) not in remove];shards=[]
 for start in range(0,len(survivors),500):
  p=B/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(B/'manifest.json',dict(status='audited',authorization='Human explicitly authorized deleting obsolete numerical checkpoint archives, preserving selected models and fresh confirmations; failed artifacts remain protected.',source=a.ref(Path(__file__)),primitives=a.ref(S),policy=a.ref(POLICY),eligibility=eligibility,delete_files=delete,inventory_shards=shards,deleted_bytes=sum(r['size'] for r in delete),free_bytes_before=shutil.disk_usage(R).free,
  dependency_review='All historical numerical families and scientific campaigns concerned are closed; their original successful archive/member manifests and reviews are hash-verified. Current warmup generated workers have passed full payload/inference and finite-pilot source integration; exact closure binds the end of their tensor-loading dependencies. No selected/fresh/failed or active weights. Future pilot prepare reads retained receipts/configs, not these weights.',limitation='Permanent retirement of244 archived successful numerical payloads and four closed successful generated-worker weights. All original source/review/metric/config/hash evidence remains. Prior archive restoration instructions superseded only for the eight exact archives. New warmup publication-failure source and every checkpoint are excluded and preserved.'))
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
 result=dict(status='completed',manifest=a.ref(B/'manifest.json'),deleted_bytes=m['deleted_bytes'],deleted_paths=sorted(remove),retained_files_verified=len(survivors),selected_fresh_failed_unchanged=True,original_evidence_unchanged=True,free_bytes_after=shutil.disk_usage(R).free,required_free_bytes=13555990528,training_launched=False)
 a.publish(B/'completed.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('audit','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'):
  idle()
  with a.CpuPrerequisiteGuard(B/(mode+'-guard.json'),R,timeout_seconds=900):audit() if mode=='audit' else apply()
