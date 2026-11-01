"""Exact user-authorized retirement of two obsolete successful FFN numerical archives."""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R));B=Path(__file__).resolve().parent
S=R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('primitives',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.E=B
from research.ffn_norm_quality_v1.support import idle
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
OLD=R/'research/tokenizer-evidence/ffn-norm-preparation-001'
ARCHIVE=OLD/'resume-archive001'
TARGETS={'research_archives/ffn-norm-resume-001/none.tar.zst','research_archives/ffn-norm-resume-001/rms.tar.zst'}
def paths():
 out=set(a.all_paths());out.update(p for p in (R/'research_state').rglob('*.pt') if p.is_file())
 for p in (R/'research/tokenizer-evidence').rglob('*'):
  if not p.is_relative_to(B) and '__pycache__' not in p.parts and p.is_file():out.add(p)
 return sorted(out)
def audit():
 assert not (B/'manifest.json').exists()
 completed=json.loads((ARCHIVE/'completed.json').read_text());assert completed['status']=='completed' and completed['archived_files']==6 and completed['selected_fresh_failed_unchanged']
 original=a.bound(completed['plan']);review=json.loads((OLD/'checkpoint-resume001.json').read_text())
 assert review['status']=='serialized_product_norm_mps_repeat_resume_h200_passed'
 assert json.loads((OLD/'checkpoint-resume001-guard.json').read_text())['status']=='finished'
 closure=json.loads((OLD/'quality001-closure.json').read_text());assert (R/'research_runs/tokenizer-ffn-norm-quality-v1-001/STOP').exists()
 warmup=R/'research/tokenizer-evidence/warmup-preparation-001/model-receipt-recovery001.json';recovered=json.loads(warmup.read_text())
 assert recovered['status']=='recovered_full_model_rates_short_repeat_resume_h200_passed'
 assert recovered['root']==str(R/'research_runs/tokenizer-warmup-model-prerequisite-001')
 eligibility=[a.ref(ARCHIVE/'completed.json'),completed['plan'],a.ref(OLD/'checkpoint-resume001.json'),a.ref(OLD/'checkpoint-resume001-guard.json'),a.ref(OLD/'quality001-closure.json'),a.ref(R/'research_runs/tokenizer-ffn-norm-quality-v1-001/STOP'),a.ref(warmup)]
 for r in original['eligibility']:a.verify(r);eligibility.append(r)
 archives=completed['archives'];assert len(archives)==2
 originals={panel['label']:panel['files'] for panel in original['panels']}
 for item,row in zip(archives,review['arms']):
  assert item['verified_before_removal'] and row['repeat_max_abs']==row['resume_max_abs']==0
  assert item['members']==originals[row['arm']]
  for n,v in item['members'].items():assert row['files'][Path(n).name]==v
 delete=[]
 for item in archives:
  path=a.verify(item['archive']);row=a.record(path);assert row['size']==item['archive_size'];assert str(path.relative_to(R)) in TARGETS
  for n,v in item['members'].items():
   assert not (R/n).exists() and n.endswith('.pt') and len(v['sha256'])==64
   assert n.startswith('research_runs/tokenizer-ffn-norm-checkpoint-prerequisite-001/')
  delete.append(dict(**row,members=item['members']))
 assert {str(Path(x['path']).relative_to(R)) for x in delete}==TARGETS
 remove={x['path'] for x in delete};survivors=[a.record(p) for p in paths() if str(p) not in remove];shards=[]
 for start in range(0,len(survivors),500):
  p=B/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(B/'manifest.json',dict(status='audited',authorization='Human explicitly authorized deleting obsolete numerical checkpoint archives, preserving selected models and fresh confirmations; failed artifacts remain protected.',source=a.ref(Path(__file__)),primitives=a.ref(S),policy=a.ref(POLICY),eligibility=eligibility,delete_files=delete,inventory_shards=shards,deleted_bytes=sum(r['size'] for r in delete),free_bytes_before=shutil.disk_usage(R).free,
  dependency_review='FFN-product normalization campaign is terminally closed. New warmup prerequisites have their own generated corpus and preserved checkpoints. Future warmup worker uses the new recovered warmup configs/corpus, never these six old FFN numerical weights. All original code/config/receipts remain available. No selected/fresh/failed or active weights.',limitation='Permanent retirement of6 successful numerical checkpoint payloads; original sources, reviews, configs, metrics and hashes retained. Prior restore instructions superseded only for the two exact obsolete FFN resume archives. New warmup publication-failure source and all its outputs are excluded and preserved.'))
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
 result=dict(status='completed',manifest=a.ref(B/'manifest.json'),deleted_bytes=m['deleted_bytes'],deleted_archives=sorted(remove),retained_files_verified=len(survivors),selected_fresh_failed_unchanged=True,original_evidence_unchanged=True,free_bytes_after=shutil.disk_usage(R).free,required_free_bytes=10066329600,training_launched=False)
 a.publish(B/'completed.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('audit','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'):
  idle()
  with a.CpuPrerequisiteGuard(B/(mode+'-guard.json'),R,timeout_seconds=900):audit() if mode=='audit' else apply()
