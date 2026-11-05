import os,sys,json,stat,hashlib,fcntl,subprocess,datetime
from pathlib import Path
ROOT=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI')
OUT=ROOT/'research/maintenance-cache-cleanup-010'
BASE=Path('/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Versions')
TARGET=BASE/'156.0.8078.79'
KEEP=[BASE/'156.0.8078.49',BASE/'156.0.8078.80']
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def free():
 s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize
def digest(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def entry(p):
 s=p.lstat();r=dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=s.st_mode,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 if stat.S_ISLNK(s.st_mode):r['link']=os.readlink(p)
 elif stat.S_ISREG(s.st_mode):r['sha256']=digest(p)
 elif not stat.S_ISDIR(s.st_mode):raise RuntimeError('special file')
 return r
def scan(p):
 rows=[entry(p)]
 for d,ds,fs in os.walk(p,followlinks=False):
  for n in ds+fs: rows.append(entry(Path(d)/n))
 return sorted(rows,key=lambda r:r['path'])
def write(name,obj):
 with open(OUT/name,'x') as f:json.dump(obj,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def consumers():
 r=subprocess.run(['/usr/sbin/lsof','-nP','+D',str(TARGET)],capture_output=True,text=True)
 if r.returncode!=1 or r.stdout or r.stderr:raise RuntimeError('lsof uncertain or consumer: '+r.stdout+r.stderr)
 ps=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True)
 if str(TARGET) in ps or any(x in ps for x in ['GoogleUpdater','GoogleSoftwareUpdate']):raise RuntimeError('consumer/updater')
 return ps
lock=open(ROOT/'research_state/external-cache-maintenance.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert os.readlink(BASE/'Current')=='156.0.8078.80'
assert TARGET.resolve()==TARGET
if sys.argv[1]=='plan':
 ps=consumers();rows=scan(TARGET)
 for r in rows:
  if stat.S_ISREG(r['mode']):assert r['nlink']==1
 keep={str(p):scan(p) for p in KEEP}
 keep['/Applications/Google Chrome.app/Contents/Info.plist']=[entry(Path('/Applications/Google Chrome.app/Contents/Info.plist'))]
 write('manifest.json',dict(at=now(),reason='Superseded intermediate Chrome framework .79; Current and top-level installed Info.plist are .80, live processes use .49; no .79 consumer or updater. Preserve .49 and .80 plus all profiles.',target=str(TARGET),entries=rows,retained=keep,processes=ps,free_before_plan=free()))
 print(json.dumps({'entries':len(rows),'allocated':sum(r['allocated'] for r in rows),'free':free()}))
elif sys.argv[1]=='apply':
 assert not (OUT/'deletion-journal.jsonl').exists()
 m=json.load(open(OUT/'manifest.json'));assert scan(TARGET)==m['entries']
 for p,rows in m['retained'].items():assert (scan(Path(p)) if Path(p).is_dir() else [entry(Path(p))])==rows
 ps=consumers();before=free();removed=0
 with open(OUT/'deletion-journal.jsonl','x') as j:
  def journal(x):j.write(json.dumps(dict(at=now(),**x))+'\n');j.flush();os.fsync(j.fileno())
  journal(dict(event='begin',free=before,reason=m['reason'],processes=ps))
  for r in sorted(m['entries'],key=lambda x:(x['path'].count('/'),x['path']),reverse=True):
   p=Path(r['path'])
   if stat.S_ISDIR(r['mode']):
    s=p.lstat();assert (s.st_dev,s.st_ino,s.st_uid,s.st_gid,s.st_mode)==(r['dev'],r['ino'],r['uid'],r['gid'],r['mode'])
    journal(dict(event='intent_rmdir',path=str(p),dev=r['dev'],ino=r['ino']));p.rmdir()
   else:
    assert entry(p)==r
    journal(dict(event='intent_unlink',path=str(p),dev=r['dev'],ino=r['ino'],sha256=r.get('sha256'),allocated=r['allocated']));p.unlink();removed+=r['allocated']
   journal(dict(event='removed',path=str(p)))
  fd=os.open(BASE,os.O_RDONLY);os.fsync(fd);os.close(fd)
  for p,rows in m['retained'].items():assert (scan(Path(p)) if Path(p).is_dir() else [entry(Path(p))])==rows
  assert os.readlink(BASE/'Current')=='156.0.8078.80'
  after=free();result=dict(at=now(),status='complete',allocated_regular_and_symlink_bytes_removed=removed,free_before=before,free_after=after,free_delta=after-before,cleanup_target=14529069056,remaining_deficit=max(0,14529069056-after),retained_verified=True,note='Filesystem free delta can differ due to APFS shared extents and concurrent writes.')
  journal(dict(event='complete',**result));write('completed.json',result);print(json.dumps(result))
else:raise RuntimeError('mode')
