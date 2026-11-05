import os,stat,json,hashlib,fcntl,subprocess,zipfile,datetime
from pathlib import Path
ROOT=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');OUT=ROOT/'research/maintenance-cache-cleanup-011';CHROME=Path('/Users/junyi/Library/Application Support/Google/Chrome')
PAIRS=[('37d183b933697a62791bf4d91eef0f197685818a1d70405561412344dad3e0a9','Subresource Filter/Unindexed Rules/9.73.1'),('152a543fab548ee3391ce32577f9ede7dc4d6deb637bde157ee42b704a0a4885','OptimizationHints/773')]
CACHE=Path('/Users/junyi/Library/Caches/typescript/5.9')
def ident(p):
 s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_nlink==1
 return dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,size=s.st_size,mode=s.st_mode,nlink=s.st_nlink,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def free():
 s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize
def save(n,o):
 with open(OUT/n,'x') as f:json.dump(o,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def clear(paths):
 r=subprocess.run(['/usr/sbin/lsof','-nP',*map(str,paths)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout and not r.stderr,(r.stdout,r.stderr)
 ps=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True)
 assert not any(x in ps for x in ['tsserver','typingsInstaller','/npm','GoogleUpdater','GoogleSoftwareUpdate'])
 return ps
with open(ROOT/'research_state/external-cache-maintenance.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'manifest.json').exists()
 candidates=[];keep=[];reasons={}
 for name,installed in PAIRS:
  p=CHROME/'component_crx_cache'/name;row=ident(p);assert row['sha256']==name
  with zipfile.ZipFile(p) as z:
   for info in z.infolist():
    if info.is_dir():continue
    assert not Path(info.filename).is_absolute() and '..' not in Path(info.filename).parts
    q=CHROME/installed/info.filename;assert q.read_bytes()==z.read(info);keep.append(ident(q))
  candidates.append(row);reasons[str(p)]='Completed component download; archive payload matches retained installed version byte for byte.'
 # Generated automatic type-acquisition package cache; no runtime, project or user data.
 for p in CACHE.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():candidates.append(ident(p));reasons[str(p)]='Unused TypeScript 5.9 automatic type-acquisition download cache; package-lock identifies public npm type declarations/registry, no tsserver or typingsInstaller consumer.'
 assert all(r['uid']==os.getuid() for r in candidates)
 paths=[Path(r['path']) for r in candidates];ps=clear(paths);before=free()
 save('manifest.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=candidates,retained=keep,reasons=reasons,processes=ps,free_before=before))
 assert all(ident(Path(r['path']))==r for r in keep+candidates);clear(paths)
 removed=0
 with open(OUT/'deletion-journal.jsonl','x') as j:
  def journal(v):j.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**v})+'\n');j.flush();os.fsync(j.fileno())
  for r in candidates:
   p=Path(r['path']);assert ident(p)==r;journal(dict(event='intent',entry=r,reason=reasons[str(p)]));p.unlink();removed+=r['allocated'];journal(dict(event='removed',path=str(p)))
  assert all(ident(Path(r['path']))==r for r in keep)
  for p in {Path(r['path']).parent for r in candidates}:
   fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
  after=free();result=dict(status='complete',allocated_bytes_removed=removed,free_before=before,free_after=after,free_delta=after-before,retained_files_verified=len(keep),cleanup_target_bytes=14529069056,remaining_deficit_bytes=max(0,14529069056-after),target_met=after>=14529069056,note='Empty directories retained; profiles/settings and installed Chrome components unchanged. APFS/concurrent writes can change free delta.')
  journal(dict(event='complete',**result));save('completed.json',result);print(json.dumps(result))
