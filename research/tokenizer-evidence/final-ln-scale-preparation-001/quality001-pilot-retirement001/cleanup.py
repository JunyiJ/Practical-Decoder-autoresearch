"""One new exact retirement of dependency-closed cross-root successful pilots.

Imports hash/journal/guard primitives only; never invokes a closed cleanup or
pilot command. The original stopped root and every non-tensor byte are retained.
"""
import importlib.util,json,os,shutil,subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[4];sys.path.insert(0,str(R));B=Path(__file__).resolve().parent
S=R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py'
spec=importlib.util.spec_from_file_location('journal_primitives',S);a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a);a.E=B
from research.final_ln_scale_quality_v1 import campaign_contract as c
from research.final_ln_scale_quality_v1.pilot_retirement import targets,verify_retirement,EXPECTED_BYTES
from research.final_ln_scale_quality_v1.support import idle
from research.final_ln_scale_campaign_v2 import pilot
POLICY=R/'research/TOKENIZER_CHECKPOINT_RETENTION.md';W=pilot.B
CLOSURE=W/'quality001-pilot-dependencies-closed.json'

def paths():
 out=set(a.all_paths())
 out.update(p for p in (R/'research_state').rglob('*') if p.is_file() and (p.suffix in ('.pt','.tmp') or p.stat().st_size>=1024**2))
 for p in (R/'research/tokenizer-evidence').rglob('*'):
  if not p.is_relative_to(B) and '__pycache__' not in p.parts and p.is_file():out.add(p)
 return sorted(out)

def consumers():
 result=subprocess.run(['/usr/sbin/lsof','-nP','-Fpn','--',*[r['path'] for r in targets()]],capture_output=True,text=True,timeout=30)
 assert result.returncode==1 and not result.stdout.strip() and not result.stderr.strip(),('Pilot consumer check blocked',result.returncode,result.stdout,result.stderr)

def check_target(row):
 p=Path(row['path']);a.check(p,row);st=p.stat()
 assert (st.st_dev,st.st_ino,st.st_uid,st.st_gid,st.st_nlink)==(row['device'],row['inode'],row['owner_uid'],row['owner_gid'],row['hardlinks'])
 for parent in (p,*p.parents):
  assert not parent.is_symlink()
  if parent==R:break

def audit():
 assert not (B/'manifest.json').exists();idle();consumers()
 closure=a.bound(a.ref(CLOSURE));assert closure['status']=='accepted_resource_pilot_tensor_dependencies_closed'
 reviewed=a.bound(closure['integration']);plan=a.bound(closure['plan']);c.verify_sources(plan['sources'])
 assert reviewed['status']=='passed' and reviewed['protected_pilot_roots_unchanged'] and len(reviewed['actual_pilot_audits'])==2 and all(r['outcome']=='passed' for r in reviewed['tests'])
 assert (pilot.OLDROOT/'STOP').exists() and not (pilot.OLDROOT/'evidence/pilots/01').exists()
 eligibility=[closure['integration'],closure['plan'],a.ref(W/'pilot002-terminal-review.json'),a.ref(W/'pilot002-review.json'),a.ref(W/'pilot002-final003-review.json'),plan['original_stop'],plan['original_failed_partial_receipt']]
 delete=[]
 for x in targets():
  p=Path(x['path']);r=a.record(p);st=p.stat();assert r['sha256']==x['sha256'] and r['size']==x['size'] and st.st_nlink==1 and st.st_uid==os.getuid()
  item=dict(**r,allocated_bytes=st.st_blocks*512,owner_uid=st.st_uid,owner_gid=st.st_gid,hardlinks=st.st_nlink,reason='Accepted successful resource-pilot tensor; finite quality/actual optimizer/EMA/RNG/history/validation/cache dependencies closed',acceptance=x['acceptance'])
  check_target(item);delete.append(item);eligibility.append(x['acceptance'])
 assert len(delete)==4 and sum(x['size'] for x in delete)==EXPECTED_BYTES and {r['path'] for r in delete}=={r['path'] for r in closure['eligible_successful_pilot_weights']}
 remove={r['path'] for r in delete};survivors=[a.record(p) for p in paths() if str(p) not in remove];shards=[]
 for start in range(0,len(survivors),500):
  p=B/f'survivors-{start//500:03d}.json';a.publish(p,survivors[start:start+500]);shards.append(a.ref(p))
 a.publish(B/'manifest.json',dict(status='audited',source=a.ref(Path(__file__)),primitives=a.ref(S),policy=a.ref(POLICY),closure=a.ref(CLOSURE),eligibility=eligibility,delete_files=delete,inventory_shards=shards,deleted_bytes=EXPECTED_BYTES,free_before=shutil.disk_usage(R).free,source_sha256=c.fingerprint(plan['sources']),authorization='Standing human authorization to retire obsolete successful pilot tensors after actual downstream closure. Exact four accepted files only; never selected/fresh or original failure evidence.',original_failed_root_remains_closed=True))
 print(json.dumps(dict(status='audited',delete_files=4,bytes=EXPECTED_BYTES,survivors=len(survivors))),flush=True)

def apply():
 assert not (B/'completed.json').exists();m=a.bound(a.ref(B/'manifest.json'));assert a.bound(a.ref(B/'audit-guard.json'))['status']=='finished'
 for r in [m['source'],m['primitives'],m['policy'],m['closure'],*m['eligibility']]:a.verify(r)
 closure=a.bound(m['closure']);c.verify_sources(a.bound(closure['plan'])['sources']);idle();consumers()
 survivors=[r for shard in m['inventory_shards'] for r in a.bound(shard)];remove={r['path'] for r in m['delete_files']}
 assert remove=={r['path'] for r in targets()} and set(map(str,paths()))=={r['path'] for r in survivors}|remove
 for r in survivors:
  st=Path(r['path']).stat();assert (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(r['device'],r['inode'],r['size'],r['mtime_ns'])
 for r in m['delete_files']:check_target(r)
 for r in m['delete_files']:
  idle();check_target(r);a.journal(dict(event='intent_permanent_retirement',file=r));Path(r['path']).unlink();a.syncdir(Path(r['path']).parent);a.journal(dict(event='removed',path=r['path']))
 assert set(map(str,paths()))=={r['path'] for r in survivors}
 for r in survivors:assert a.record(Path(r['path']))==r
 for r in [m['source'],m['primitives'],m['policy'],m['closure'],*m['eligibility']]:a.verify(r)
 result=dict(status='completed',manifest=a.ref(B/'manifest.json'),deleted_bytes=EXPECTED_BYTES,allocated_bytes_removed=sum(r['allocated_bytes'] for r in m['delete_files']),deleted_paths=sorted(remove),retained_files_verified=len(survivors),selected_fresh_failed_unchanged=True,original_evidence_unchanged=True,free_bytes_after=shutil.disk_usage(R).free,required_free_bytes=12381585408,training_launched=False)
 a.publish(B/'completed.json',result);verify_retirement(a.ref(B/'completed.json'));print(json.dumps(result),flush=True)

if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('audit','apply')
 with a.session_lock(R/'research_state/tokenizer-worker-lock'),a.session_lock(pilot.OLDROOT),a.session_lock(pilot.ROOT):
  idle()
  with a.CpuPrerequisiteGuard(B/(mode+'-guard.json'),R,timeout_seconds=900):audit() if mode=='audit' else apply()
