import os,json,stat,hashlib,fcntl,subprocess,datetime
from pathlib import Path
ROOT=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');OUT=ROOT/'research/maintenance-cache-cleanup-012'
CACHE=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/org.python.python/com.apple.metal')
paths=sorted(p for p in CACHE.rglob('*') if p.is_file())+[Path('/Users/junyi/Library/Caches/go-build/README'),Path('/Users/junyi/Library/Caches/go-build/trim.txt')]
def info(p):
 s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
 return dict(path=str(p),size=s.st_size,allocated=s.st_blocks*512,dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=s.st_mode,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def free():
 s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize
def save(n,o):
 with open(OUT/n,'x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def consumers():
 ps=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True)
 for l in ps.splitlines():
  pid,cmd=l.strip().split(maxsplit=1)
  assert int(pid)==os.getpid() or not ('python' in cmd.lower() or cmd.endswith('/go')),l
 r=subprocess.run(['/usr/sbin/lsof','-nP',*map(str,paths)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout and not r.stderr
 return ps
with open(ROOT/'research_state/external-cache-maintenance.lock','a+') as external,open(ROOT/'research_state/tokenizer-worker-lock/.lock','r') as worker:
 fcntl.flock(external,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert len(paths)==10 and not (OUT/'manifest.json').exists()
 rows=[info(p) for p in paths];ps=consumers();before=free()
 save('manifest.json',dict(entries=rows,processes=ps,free_before=before,reason='Unused Python Metal compiled function/library cache, no Python worker or open consumer; regenerate on demand. Go build-cache README and trim metadata have no active consumer. No research output, source, installed environment, or app data is in this allowlist.'))
 assert rows==[info(p) for p in paths];consumers()
 with open(OUT/'deletion-journal.jsonl','x') as j:
  def journal(x):j.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**x})+'\n');j.flush();os.fsync(j.fileno())
  for p,r in zip(paths,rows):
   assert info(p)==r;journal(dict(event='intent',entry=r));p.unlink();journal(dict(event='removed',path=str(p)))
  for p in {p.parent for p in paths}:
   fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
  after=free();result=dict(status='complete',allocated_bytes_removed=sum(r['allocated'] for r in rows),free_before=before,free_after=after,free_delta=after-before,cleanup_target_bytes=14529069056,remaining_deficit_bytes=max(0,14529069056-after),note='No research payload changed. APFS sharing/concurrent activity may affect free-space delta.')
  journal(dict(event='complete',**result));save('completed.json',result);print(json.dumps(result))
