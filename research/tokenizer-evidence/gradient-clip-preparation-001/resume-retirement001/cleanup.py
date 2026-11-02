"""Exact user-authorized retirement of four obsolete successful gradient-clip generated-resume weights."""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R));B=Path(__file__).resolve().parent
S=R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('primitives',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.E=B
from research.ffn_norm_quality_v1.support import idle
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
W=R/'research/tokenizer-evidence/gradient-clip-preparation-001'
TARGETS={f'research_runs/tokenizer-gradient-clip-resume-001/{arm}/{name}' for arm in ('clip1','cliphalf') for name in ('final.pt','h200.pt')}
def paths():
 out=set(a.all_paths());out.update(p for p in (R/'research_state').rglob('*') if p.is_file() and (p.suffix in ('.pt','.tmp') or p.stat().st_size>=1024**2))
 for p in (R/'research/tokenizer-evidence').rglob('*'):
  if not p.is_relative_to(B) and '__pycache__' not in p.parts and p.is_file():out.add(p)
 return sorted(out)
def audit():
 assert not (B/'manifest.json').exists()
 closure=json.loads((W/'resume-dependencies-closed001.json').read_text());assert closure['status']=='successful_resume_tensor_dependencies_closed'
 integration=json.loads((W/'worker-integration001.json').read_text());assert integration['status']=='gradient_clip_actual_resume_payload_adapter_integration_passed'
 eligibility=[a.ref(W/n) for n in ['resume-dependencies-closed001.json','worker-integration001.json','worker-integration001-plan.json','worker-integration001-inference.json','worker-integration001-guard.json','model-resume001.json','model-resume001-plan.json','model-resume001-guard.json','trainer-worker001.py','trainer-worker001-source-plan.json','trained-inference001.py']]
 from research.gradient_clip_worker_v1 import evidence_contract as c
 c.verify_sources(integration['sources'])
 assert a.bound(integration['guard'])['status']=='finished'
 resume=json.loads((W/'model-resume001.json').read_text());assert resume['status']=='gradient_clip_both_arms_serialized_resume_final_h200_passed'
 assert len(resume['arms'])==len(integration['rows'])==2
 assert {r['path'] for r in closure['eligible_successful_numerical_weights']}=={str(R/n) for n in TARGETS}
 delete=[]
 for row in resume['arms']:
  assert row['repeat_max_abs']==row['resume_max_abs']==0 and row['restored_state_and_rng_exact']
  checked=next(x for x in integration['rows'] if x['arm']==row['arm'])
  assert checked['ema_parameters']==88 and checked['ema_snapshots']==3 and checked['cpu_mps_ce_abs']<1e-5
  for name in ('final.pt','h200.pt'):
   path=Path(resume['root'])/row['arm']/name;record=a.record(path)
   assert str(path.relative_to(R)) in TARGETS and record['sha256']==row['files'][name]['sha256'] and record['size']==row['files'][name]['size']
   assert any(x['path']==str(path) and x['size']==record['size'] and x['sha256']==record['sha256'] for x in closure['eligible_successful_numerical_weights'])
   delete.append(dict(**record,kind='closed_successful_generated_resume',original_acceptance=a.ref(W/'model-resume001.json')))
 assert len(delete)==4 and sum(x['size'] for x in delete)==1118316500
 assert {str(Path(x['path']).relative_to(R)) for x in delete}==TARGETS
 remove={x['path'] for x in delete};survivors=[a.record(p) for p in paths() if str(p) not in remove];shards=[]
 for start in range(0,len(survivors),500):
  p=B/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(B/'manifest.json',dict(status='audited',authorization='Human authorized obsolete checkpoint retirement while keeping selected models and fresh confirmations; these four successful generated prerequisite weights are obsolete after tested dependency closure.',source=a.ref(Path(__file__)),primitives=a.ref(S),policy=a.ref(POLICY),eligibility=eligibility,delete_files=delete,inventory_shards=shards,deleted_bytes=sum(r['size'] for r in delete),free_bytes_before=shutil.disk_usage(R).free,
  dependency_review='Both generated resume arms passed exact repeat/resume/final H200 and new real payload/CPU/MPS full-validation/cache adapter audit. Dependencies closed; prepared fresh worker reads retained metadata only. Selected/fresh/failed and all historical evidence remain untouched.',limitation='Permanent retirement of only four successful generated prerequisite payloads. Original receipts remain immutable; no restoration or prerequisite rerun.'))
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
 result=dict(status='completed',manifest=a.ref(B/'manifest.json'),deleted_bytes=m['deleted_bytes'],deleted_paths=sorted(remove),retained_files_verified=len(survivors),selected_fresh_failed_unchanged=True,original_evidence_unchanged=True,free_bytes_after=shutil.disk_usage(R).free,required_free_bytes=9814671360,training_launched=False)
 a.publish(B/'completed.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('audit','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'):
  idle()
  with a.CpuPrerequisiteGuard(B/(mode+'-guard.json'),R,timeout_seconds=900):audit() if mode=='audit' else apply()
