"""New closed-metadata recovery: exact bytes preserved, no checkpoint mutation."""
import fcntl,hashlib,importlib.util,json,os,shutil,stat,subprocess
from pathlib import Path
R=Path.cwd();E=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('primitives',R/'research/tokenizer-evidence/filesystem-compression-001/compress.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.adamw_epsilon_quality_v1.support import idle
B=R/'research/tokenizer-evidence/final-ln-scale-preparation-001'
HIST=R/'research/tokenizer-evidence'
ROOTS=[HIST]

def info(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==os.getuid() and not p.is_symlink()
 return dict(a.info(p),device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,reason='Closed previously untried sub4KiB metadata;exact readable bytes retained; lossless transparent compression only')
def check(p,v,physical=False):
 a.check(p,v,physical=False)
 if physical:assert info(p)==v
fd=os.open(R/'research_state/tokenizer-worker-lock/.lock',os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 idle()
 with CpuPrerequisiteGuard(E/'guard.json',R,timeout_seconds=600):
  attempted=set()
  for package in HIST.iterdir():
   if not package.is_dir() or package==E or not(package.name.startswith('storage-recovery-') or package.name.startswith('filesystem-compression-')):continue
   for plan_path in package.rglob('*plan*.json'):
    if plan_path.stat().st_size>64*1024**2:continue
    previous=json.loads(plan_path.read_text());old_files=previous.get('files',{}) if isinstance(previous,dict) else {}
    if isinstance(old_files,dict):attempted.update(str((R/n).resolve()) if not Path(n).is_absolute() else n for n in old_files if isinstance(n,str))
    elif isinstance(old_files,list):attempted.update(str((R/v['path']).resolve()) if not Path(v['path']).is_absolute() else v['path'] for v in old_files if isinstance(v,dict) and isinstance(v.get('path'),str))
  # New untried sub4KiB class. Earlier032/034/035 used>=4096B.
  # No live numerical worker;all paths/readable bytes stay immutable.
  files={str(p):info(p) for p in HIST.rglob('*') if E not in p.parents and p.suffix in ('.json','.jsonl','.log','.txt') and p.is_file() and not p.is_symlink() and 0<p.stat().st_size<4096 and p.stat().st_blocks>0 and not(p.stat().st_flags&32) and str(p) not in attempted}
  assert len(files)<=12000 and sum(v['allocated_bytes'] for v in files.values())<=64*1024**2
  assert files
  assert json.loads((B/'quality001-pilot-retirement001/verification002.json').read_text())['status']=='passed'
  assert (B/'quality001-accepted-02.json').exists() and json.loads((B/'quality001-slot03-entry-failure001.json').read_text())['optimizer_updates']==0
  assert (R/'research_runs/tokenizer-final-ln-scale-quality-v1-001/STOP').exists()
  done=json.loads((R/'research/tokenizer-evidence/storage-recovery-028/raw-completed.json').read_text());assert done['status']=='completed_lossless_raw_storage'
  protected={}
  for row in done['files']:
   for key in ('archive','h200'):
    p=Path(row[key]['path']);v=info(p);assert v['sha256']==row[key]['sha256'];protected[str(p)]=v
  for slot in ('00','01','02'):
   row=json.loads((R/f'research_runs/tokenizer-final-ln-scale-quality-v1-001/evidence/quality/{slot}/resource-events.jsonl').read_text());worker=json.loads(Path(row['worker']['path']).read_text())
   assert row['artifact_acceptance']['accepted']
   for kind in ('parent','h200'):
    bound=worker['acceptance_request']['files'][kind];p=Path(bound['path']);v=info(p);assert v['sha256']==bound['sha256'];protected[str(p)]=v
  a.publish(E/'plan.json',dict(operation='lossless_APFS_closed_metadata_compression',source=info(Path(__file__)),files=files,protected_affected_models=protected,authorization='Standing human storage recovery; original evidence paths/readable bytes and every selected/fresh model preserved',free_before=shutil.disk_usage(R).free,required=10705268590,target_with_extra_headroom=12852752238,nonblocking_research_lock=True))
  if files:
   names=list(files)
   for start in range(0,len(names),150):
    consumer=subprocess.run(['/usr/sbin/lsof','-nP','-Fpn','--',*names[start:start+150]],capture_output=True,text=True,timeout=30)
    assert consumer.returncode==1 and not consumer.stdout.strip() and not consumer.stderr.strip(),'Closed metadata has a consumer'
  saved=0
  with (E/'journal.jsonl').open('x') as log:
   def event(d):log.write(json.dumps(d)+'\n');log.flush();os.fsync(log.fileno())
   for n,v in files.items():
    p=Path(n);check(p,v,True);tmp=p.with_name(p.name+'.storage036.tmp');assert not tmp.exists() and shutil.disk_usage(R).free>=8*1024**3+v['size']
    subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(p),str(tmp)],check=True,timeout=30);w=a.check(tmp,v)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    check(p,v,True)
    if w['allocated_bytes']>=v['allocated_bytes']:event(dict(event='verified_no_savings',path=n));tmp.unlink();a.syncdir(p.parent);continue
    assert w['flags']&32;event(dict(event='verified_staged_copy',path=n,sha256=v['sha256'],before=v['allocated_bytes'],after=w['allocated_bytes']));os.replace(tmp,p);a.syncdir(p.parent);check(p,v);saved+=v['allocated_bytes']-w['allocated_bytes'];event(dict(event='replacement_verified',path=n,sha256=v['sha256']))
  for n,v in files.items():check(Path(n),v)
  for n,v in protected.items():check(Path(n),v,True)
  a.publish(E/'completed.json',dict(status='completed_lossless_metadata_compression',files=len(files),allocated_bytes_reclaimed=saved,protected_affected_models_verified=len(protected),all_paths_and_readable_bytes_unchanged=True,no_weights_modified=True,free_after=shutil.disk_usage(R).free))
  print((E/'completed.json').read_text())
finally:os.close(fd)
