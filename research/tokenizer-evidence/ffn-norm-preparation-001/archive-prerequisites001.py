"""Exact lossless recovery of successful non-selected numerical fixtures only."""
import importlib.util,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('archive_primitives',SOURCE);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
B=Path(__file__).resolve().parent;E=B/'archive001';COLD=ROOT/'research_archives/ffn-norm-prerequisites-001'
a.E=E;a.COLD=COLD
OLD=ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
CASES=(('h200','block-norm-v2-h200-fixture-001','grouped-embedding-h200-fixture-manifest.json','grouped-embedding-h200-001-review.json'),
 ('trainer','block-norm-trainer-fixture-002','trainer-binding-fixture-manifest.json','trainer-binding-prerequisite-002-review.json'),
 ('resume','block-norm-resume-fixture-001','resume-replay-fixture-manifest.json','resume-replay-prerequisite-001-review.json'),
 ('transport','block-norm-checkpoint-fixture-001','checkpoint-transport-fixture-manifest.json','checkpoint-transport-prerequisite-001-review.json'))
def paths():
 values=set(a.all_paths())
 values.update(p for p in (ROOT/'research_state').rglob('*.pt') if p.is_file())
 for _,name,_,_ in CASES:values.update(p for p in (ROOT/'research_state'/name).rglob('*') if p.is_file())
 return sorted(values)
def plan():
 E.mkdir();panels=[];eligibility=[];files={}
 for label,root,mn,rn in CASES:
  manifest=json.loads((OLD/mn).read_text());review=json.loads((OLD/rn).read_text());assert review['status']=='passed' and review['error'] is None
  entries=manifest['attempt002']['files'] if label=='trainer' else manifest['files']
  weights={n:dict(size=v['bytes'],sha256=v['sha256']) for n,v in entries.items() if n.endswith('.pt')}
  assert len(weights)==(2 if label=='resume' else 4)
  assert set(weights)=={str(p.relative_to(ROOT)) for p in (ROOT/'research_state'/root).rglob('*.pt')}
  for n,v in weights.items():a.check(ROOT/n,v)
  for n,v in entries.items():
   assert n.startswith('research_state/'+root+'/');a.check(ROOT/n,dict(size=v['bytes'],sha256=v['sha256']))
  panels.append(dict(label=label,files=weights));files.update(weights);eligibility.extend([a.ref(OLD/mn),a.ref(OLD/rn)])
 survivors=[a.record(p) for p in paths() if str(p.relative_to(ROOT)) not in files];shards=[]
 for start in range(0,len(survivors),500):
  p=E/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(E/'plan.json',dict(status='planned_successful_prerequisite_lossless_archive',source=a.ref(Path(__file__)),primitives=a.ref(SOURCE),policy=a.ref(ROOT/'research/TOKENIZER_CHECKPOINT_RETENTION.md'),
  eligibility=eligibility,panels=panels,inventory_shards=shards,free_bytes=shutil.disk_usage(ROOT).free,
  reason='Four closed successful generated-text/numerical prerequisites; no selected, fresh, failed, active or unknown weights. Exact original manifest identities required.',
  restore='Resolve these later archive manifests or restore absent original paths through verified staging with reserve; never overwrite or rerun closed fixtures.',
  limits=dict(reserve_bytes=a.RESERVE,archive_cap_bytes=2300*1024**2,threads=1,rss_bytes=4*1024**3,subprocess_seconds=600)))
 print(json.dumps(dict(status='planned',files=len(files),logical_bytes=sum(v['size'] for v in files.values()),survivors=len(survivors))),flush=True)
def apply():
 p=json.loads((E/'plan.json').read_text())
 for r in [p['source'],p['primitives'],p['policy'],*p['eligibility']]:a.verify(r)
 survivors=[r for shard in p['inventory_shards'] for r in a.bound(shard)]
 files={n:v for panel in p['panels'] for n,v in panel['files'].items()}
 for r in survivors:
  path=Path(r['path']);s=path.stat();assert not path.is_symlink()
  assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(r['device'],r['inode'],r['size'],r['mtime_ns'])
 assert set(map(str,paths()))=={r['path'] for r in survivors}|{str(ROOT/n) for n in files}
 COLD.mkdir();receipts=[]
 for panel in p['panels']:
  archive=COLD/(panel['label']+'.tar.zst');a.create_archive(archive,panel['files']);members=a.verify_archive(archive,panel['files'])
  receipt=dict(archive=a.ref(archive),archive_size=archive.stat().st_size,members=members,verified_before_removal=True)
  a.publish(E/(panel['label']+'-verified.json'),receipt)
  for n,v in panel['files'].items():a.check(ROOT/n,v)
  for n,v in panel['files'].items():
   a.check(ROOT/n,v);a.journal(dict(event='intent_remove_verified_loose_copy',path=n,sha256=v['sha256'],archive=receipt['archive']))
   (ROOT/n).unlink();a.syncdir((ROOT/n).parent);a.journal(dict(event='removed_verified_loose_copy',path=n))
  a.verify(receipt['archive']);receipts.append(receipt)
 for r in survivors:assert a.record(Path(r['path']))==r
 assert set(map(str,paths()))=={r['path'] for r in survivors}|{r['archive']['path'] for r in receipts}
 result=dict(status='completed',plan=a.ref(E/'plan.json'),archives=receipts,archived_files=len(files),logical_bytes=sum(v['size'] for v in files.values()),
  net_logical_reclaimed_bytes=sum(v['size'] for v in files.values())-sum(r['archive_size'] for r in receipts),free_bytes=shutil.disk_usage(ROOT).free,
  retained_files_verified=len(survivors),no_checkpoint_bytes_lost=True,selected_fresh_failed_unchanged=True,training=False,holdout_consumed=False)
 a.publish(E/'completed.json',result);print(json.dumps({k:v for k,v in result.items() if k!='archives'}),flush=True)
if __name__=='__main__':
 from research.mla_score_quality_v1.support import idle
 mode=sys.argv[1];assert mode in ('plan','apply')
 with a.session_lock(ROOT/'research_state/tokenizer-worker-lock'):
  idle()
  with a.CpuPrerequisiteGuard(B/'archive001-plan-guard.json' if mode=='plan' else E/'apply-guard.json',ROOT,timeout_seconds=600 if mode=='plan' else 1800):
   plan() if mode=='plan' else apply()
