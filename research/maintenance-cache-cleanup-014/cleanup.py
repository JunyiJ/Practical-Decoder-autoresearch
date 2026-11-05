"""One-shot removal of a newly completed Optimization Hints 774 download."""
import fcntl,hashlib,json,os,stat,subprocess,zipfile,datetime
from pathlib import Path
ROOT=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');OUT=ROOT/'research/maintenance-cache-cleanup-014'
CACHE=Path('/Users/junyi/Library/Application Support/Google/Chrome/component_crx_cache/8b44dcd98d87e5477b57ad85c1570837d0fff7c48f3159048300c6661c7bde7f')
INSTALLED=Path('/Users/junyi/Library/Application Support/Google/Chrome/OptimizationHints/774')
def info(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and p.resolve()==p
 return dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=s.st_mode,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def free():
 s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize
def save(n,x):
 with (OUT/n).open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def clear():
 r=subprocess.run(['/usr/sbin/lsof','-nP',str(CACHE)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout and not r.stderr
with (ROOT/'research_state/external-cache-maintenance.lock').open('r') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'manifest.json').exists()
 row=info(CACHE);assert row['uid']==os.getuid() and row['nlink']==1 and row['sha256']==CACHE.name
 keep=[]
 with zipfile.ZipFile(CACHE) as z:
  for i in z.infolist():
   if i.is_dir():continue
   assert not Path(i.filename).is_absolute() and '..' not in Path(i.filename).parts
   p=INSTALLED/i.filename;assert p.read_bytes()==z.read(i);keep.append(info(p))
 assert len(keep)==3;clear();before=free()
 save('manifest.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidate=row,retained=keep,free_before=before,reason='Completed Chrome Optimization Hints 774 component download; every archive member matches retained installed file, no open consumer.'))
 assert info(CACHE)==row and all(info(Path(r['path']))==r for r in keep);clear()
 with (OUT/'deletion-journal.jsonl').open('x') as j:
  def record(v):j.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**v})+'\n');j.flush();os.fsync(j.fileno())
  record(dict(event='intent',candidate=row));CACHE.unlink();fd=os.open(CACHE.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);record(dict(event='removed',path=str(CACHE)))
  assert all(info(Path(r['path']))==r for r in keep)
  after=free();result=dict(status='complete',allocated_bytes_removed=row['allocated'],free_before=before,free_after=after,free_delta=after-before,retained_files_verified=3,remaining_panel_requirement_bytes=11263066403,cleanup_target_bytes=13410550051,cleanup_target_deficit=max(0,13410550051-after),target_met=after>=13410550051,note='APFS/concurrent writes may affect observed free delta. Installed component and browser metadata/profile/session preserved. Other previously exhausted or blocked classes unchanged.')
  record(dict(event='complete',**result));save('completed.json',result);print(json.dumps(result))
