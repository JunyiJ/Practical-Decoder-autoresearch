"""Exact lossless recovery of successful non-selected numerical fixtures only."""
import importlib.util,json,os,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
SOURCE=ROOT/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('archive_primitives',SOURCE);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
B=Path(__file__).resolve().parent;E=B/'historical-numerical-archive001';COLD=ROOT/'research_archives/ffn-norm-historical-numerical-001'
a.E=E;a.COLD=COLD
OLD=ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
NAMES={'cache-check.pt','table-math-check.pt','branch-math-check.pt','attention-math-check.pt','linked-graph-check.pt','bias-math-check.pt','norm-math.pt','gate-math.pt','k-norm-math.pt','q-norm-math.pt','w256-midpoint.pt','w64-midpoint.pt'}
def review_paths():
 return sorted({p.parent/'review.json' for p in (ROOT/'research_state').rglob('*.pt') if (p.parent/'review.json').is_file()})
def paths():
 values=set(a.all_paths())
 values.update(p for p in (ROOT/'research_state').rglob('*.pt') if p.is_file())
 for review in review_paths():values.update(p for p in review.parent.rglob('*') if p.is_file() and '__pycache__' not in p.parts)
 return sorted(values)
def hash_fields(value,prefix=''):
 out={}
 if isinstance(value,dict):
  for k,v in value.items():
   key=prefix+'.'+k
   if k.endswith('sha256') and isinstance(v,str):out[key]=v
   else:out.update(hash_fields(v,key))
 elif isinstance(value,list):
  for i,v in enumerate(value):out.update(hash_fields(v,prefix+'.'+str(i)))
 return out
def plan():
 E.mkdir();panels=[];eligibility=[];files={}
 reviews=review_paths();review_data={r:json.loads(r.read_text()) for r in reviews};excluded=[]
 for path in sorted((ROOT/'research_state').rglob('*.pt')):
  review=path.parent/'review.json';d=review_data.get(review,{})
  if path.name not in NAMES or d.get('status')!='verified':
   excluded.append(dict(path=str(path),size=path.stat().st_size,reason='No verified original numerical review; keep loose and protected'));continue
  r=a.record(path);fields=[k for k,h in hash_fields(d).items() if h==r['sha256']]
  assert fields,'Original success receipt must bind exact numerical bytes'
  n=str(path.relative_to(ROOT));files[n]=dict(size=r['size'],sha256=r['sha256']);a.check(path,files[n])
  eligibility.append(dict(checkpoint=n,review=a.ref(review),matched_sha256_fields=fields,scope='Successful non-selected numerical/math/cache fixture; no scientific endpoint or failed fixture'))
 assert len(files)==220 and sum(v['size'] for v in files.values())==703598400 and len(excluded)==4
 a.publish(E/'eligibility.json',dict(eligible=eligibility,excluded=excluded))
 panels=[dict(label='successful-numerical',files=files)]
 eligibility=[a.ref(E/'eligibility.json'),*[a.ref(r) for r in reviews]]
 survivors=[a.record(p) for p in paths() if str(p.relative_to(ROOT)) not in files];shards=[]
 for start in range(0,len(survivors),500):
  p=E/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(E/'plan.json',dict(status='planned_successful_prerequisite_lossless_archive',source=a.ref(Path(__file__)),primitives=a.ref(SOURCE),policy=a.ref(ROOT/'research/TOKENIZER_CHECKPOINT_RETENTION.md'),
  eligibility=eligibility,panels=panels,inventory_shards=shards,free_bytes=shutil.disk_usage(ROOT).free,
  reason='Exact original verified reviews bind these historical character numerical/math/cache and coupled-resume fixtures. All current tokenizer runtimes and campaigns use separate sources/receipts and never load these numerical payloads. Keep four unreviewed/failed fixtures loose. No selected, fresh, active, or scientific endpoint weights. Lossless archival only; original sources and reviews remain immutable.',
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
  with a.CpuPrerequisiteGuard(B/'historical-numerical-archive001-plan-guard.json' if mode=='plan' else E/'apply-guard.json',ROOT,timeout_seconds=600 if mode=='plan' else 1800):
   plan() if mode=='plan' else apply()
