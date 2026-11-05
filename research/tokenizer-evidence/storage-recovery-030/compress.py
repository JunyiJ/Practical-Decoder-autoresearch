"""New closed-metadata recovery: exact bytes preserved, no checkpoint mutation."""
import fcntl,hashlib,importlib.util,json,os,shutil,stat,subprocess
from pathlib import Path
R=Path.cwd();E=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('primitives',R/'research/tokenizer-evidence/filesystem-compression-001/compress.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.adamw_epsilon_quality_v1.support import idle
B=R/'research/tokenizer-evidence/final-ln-scale-preparation-001'
ROOTS=[B/'resume-retirement001',B/'worker-retirement001',B/'trainer-worker001-evidence']
def info(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==os.getuid() and not p.is_symlink()
 return dict(a.info(p),device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,reason='Closed immutable audit JSON; lossless transparent compression only')
def check(p,v,physical=False):
 a.check(p,v,physical=False)
 if physical:assert info(p)==v
fd=os.open(R/'research_state/tokenizer-worker-lock/.lock',os.O_RDWR);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 idle()
 with CpuPrerequisiteGuard(E/'guard.json',R,timeout_seconds=300):
  files={str(p):info(p) for root in ROOTS for p in root.rglob('*.json') if p.is_file() and not p.is_symlink() and p.stat().st_size>=128*1024 and not(p.stat().st_flags&32)}
  assert files
  assert json.loads((B/'worker-retirement001/completed.json').read_text())['status']=='completed'
  done=json.loads((R/'research/tokenizer-evidence/storage-recovery-028/raw-completed.json').read_text());assert done['status']=='completed_lossless_raw_storage'
  protected={}
  for row in done['files']:
   for key in ('archive','h200'):
    p=Path(row[key]['path']);v=info(p);assert v['sha256']==row[key]['sha256'];protected[str(p)]=v
  a.publish(E/'plan.json',dict(operation='lossless_APFS_closed_metadata_compression',source=info(Path(__file__)),files=files,protected_affected_models=protected,authorization='Standing human storage recovery; original evidence paths/readable bytes and every selected/fresh model preserved',free_before=shutil.disk_usage(R).free,required=12381585408,target_with_extra_headroom=14529069056,nonblocking_research_lock=True))
  saved=0
  with (E/'journal.jsonl').open('x') as log:
   def event(d):log.write(json.dumps(d)+'\n');log.flush();os.fsync(log.fileno())
   for n,v in files.items():
    p=Path(n);check(p,v,True);tmp=p.with_name(p.name+'.storage030.tmp');assert not tmp.exists() and shutil.disk_usage(R).free>=8*1024**3+v['size']
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
