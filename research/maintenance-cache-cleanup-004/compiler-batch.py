"""One-shot unused Xcode SDK-stat and CoreSimulator Metal build cache cleanup."""
import fcntl,json,os,shutil,subprocess,sys,time
from pathlib import Path
"""One-shot removal of unused component downloads verified against installed bytes."""
import fcntl,hashlib,json,os,re,shutil,stat,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
BASE=Path('/Users/junyi/Library/Application Support')
APPS=[BASE/'Google/Chrome',BASE/'Codex']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def inspect(p):
 for q in (p,*p.parents):assert not q.is_symlink(),str(q)
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1,str(p)
 h=sha(p);t=p.lstat()
 keys=('st_dev','st_ino','st_mode','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns','st_blocks')
 assert all(getattr(s,k)==getattr(t,k) for k in keys)
 return dict(path=str(p),sha256=h,allocated_bytes=s.st_blocks*512,**{k:getattr(s,k) for k in keys})

def save(n,v):
 with (HERE/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def opened():
 r=subprocess.run(['/usr/sbin/lsof','-nP','-Fn'],capture_output=True,text=True,timeout=30)
 assert r.returncode==0 and not r.stderr.strip(),r.stderr
 return {x[1:] for x in r.stdout.splitlines() if x.startswith('n')}


ROOTS=[Path('/Users/junyi/Library/Developer/Xcode/DerivedData/SDKStatCaches.noindex'),Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/com.apple.CoreSimulator.SimRenderingServices.SimMetalHost/com.apple.metal')]

def idle():
 ps=subprocess.check_output(['ps','-axo','pid,comm'],text=True)
 for line in ps.splitlines()[1:]:
  bits=line.strip().split(maxsplit=1)
  if len(bits)<2:continue
  name=Path(bits[1]).name.lower()
  assert not any(s in name for s in ('simulator','simmetal','xcode','clang','swift','metal','ibtoold')),line

if sys.argv[1:]==['plan']:
 idle();handles=opened();rows=[]
 for root in ROOTS:
  for p in root.rglob('*'):
   if p.is_dir():continue
   r=inspect(p);assert str(p) not in handles and time.time()-p.stat().st_mtime>3600;rows.append(r)
 rows.sort(key=lambda r:r['allocated_bytes'],reverse=True)
 save('compiler-manifest.json',dict(files=rows,reason='Regenerable SDK filesystem-stat and CoreSimulator Metal build caches; no Xcode/compiler/simulator/Metal compiler process or open file',source_sha256=sha(Path(__file__)),helper_sha256=sha(HERE/'cleanup.py')))
 print(json.dumps(dict(files=len(rows),allocated_bytes=sum(r['allocated_bytes'] for r in rows))))
elif sys.argv[1:]==['apply']:
 m=json.loads((HERE/'compiler-manifest.json').read_text());assert m['source_sha256']==sha(Path(__file__)) and m['helper_sha256']==sha(HERE/'cleanup.py')
 lock=ROOT/'research_state/external-cache-maintenance.lock';assert not lock.is_symlink()
 with lock.open('a+') as held:
  fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB);idle();handles=opened();before=shutil.disk_usage(ROOT).free;removed=[]
  for r in m['files']:assert r['path'] not in handles and inspect(Path(r['path']))==r
  with (HERE/'compiler-deletion-journal.jsonl').open('x') as j:
   def event(v):j.write(json.dumps(dict(time=time.time(),**v))+'\n');j.flush();os.fsync(j.fileno())
   for r in m['files']:
    p=Path(r['path']);assert any(p.is_relative_to(root) for root in ROOTS)
    idle();assert str(p) not in opened() and inspect(p)==r
    event(dict(event='intent',file=r));p.unlink();assert not p.exists();removed.append(r);event(dict(event='removed',path=str(p)))
  after=shutil.disk_usage(ROOT).free
  save('compiler-receipt.json',dict(status='completed',files_removed=len(removed),allocated_bytes_removed=sum(r['allocated_bytes'] for r in removed),logical_bytes_removed=sum(r['st_size'] for r in removed),free_before=before,free_after=after,observed_free_delta=after-before,research_payloads_modified=False,completed_unix=time.time()))
  print((HERE/'compiler-receipt.json').read_text())
else:raise SystemExit('plan or apply')
