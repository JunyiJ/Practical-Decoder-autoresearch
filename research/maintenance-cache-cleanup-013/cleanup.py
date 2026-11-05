"""One-shot recovery of newly regenerated Metal caches after fresh slot01."""
import os,json,stat,hashlib,fcntl,subprocess,datetime
from pathlib import Path
ROOT=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');OUT=ROOT/'research/maintenance-cache-cleanup-013'
CACHE=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/org.python.python/com.apple.metal')
PATHS=[CACHE/'16777235_403/functions.data',CACHE/'16777235_403/functions.list',CACHE/'32024/libraries.data',CACHE/'32024/libraries.list']
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def entry(p):
 s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode)
 return dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,size=s.st_size,mode=s.st_mode,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=sha(p))
def free():
 s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize
def save(n,v):
 with open(OUT/n,'x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def consumers():
 ps=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True)
 for l in ps.splitlines():
  pid,comm=l.strip().split(maxsplit=1);assert int(pid)==os.getpid() or 'python' not in comm.lower(),l
 r=subprocess.run(['/usr/sbin/lsof','-nP',*map(str,PATHS)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout and not r.stderr
 return ps
with open(ROOT/'research_state/external-cache-maintenance.lock','a+') as ext,open(ROOT/'research_state/tokenizer-worker-lock/.lock','r') as worker:
 fcntl.flock(ext,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(worker,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'manifest.json').exists()
 candidates=[entry(p) for p in PATHS];assert all(r['uid']==os.getuid() and r['nlink']==1 for r in candidates)
 retained=[];seen=set();weight=0
 for slot in [0,1]:
  receipt=ROOT/f'research/tokenizer-evidence/final-ln-scale-preparation-001/quality001-accepted-{slot:02}.json';r=json.loads(receipt.read_text());retained.append(entry(receipt));assert r['slot']==slot
  for row in r['files']+[r['parent_guard']]:
   e=entry(Path(row['path']));assert e['sha256']==row['sha256'];retained.append(e)
   if Path(e['path']).suffix=='.pt' and (e['dev'],e['ino']) not in seen:seen.add((e['dev'],e['ino']));weight+=e['size']
  assert json.loads(Path(r['parent_guard']['path']).read_text())['status']=='finished'
 ps=consumers();before=free();required=max(8*1024**3,12381585408-weight)
 save('manifest.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),candidates=candidates,retained=retained,processes=ps,free_before=before,required_remaining_bytes=required,reason='Four newly regenerated Python Metal function/library cache files after slot01 completed and parent accepted; no Python consumer or open handles. Fresh RAW/H200 endpoints and original evidence retained.'))
 assert [entry(p) for p in PATHS]==candidates;consumers()
 with open(OUT/'deletion-journal.jsonl','x') as j:
  def journal(v):j.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**v})+'\n');j.flush();os.fsync(j.fileno())
  for p,e in zip(PATHS,candidates):
   assert entry(p)==e;journal(dict(event='intent',entry=e));p.unlink();journal(dict(event='removed',path=str(p)))
  for p in {p.parent for p in PATHS}:
   fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
  assert all(entry(Path(e['path']))==e for e in retained)
  after=free();result=dict(status='completed_new_regenerated_cache_recovery',allocated_bytes_removed=sum(e['allocated'] for e in candidates),free_before=before,free_after=after,observed_free_delta=after-before,accepted_slots=[0,1],retained_weight_unique_inodes=len(seen),retained_logical_weight_bytes=weight,retained_entries_verified=len(retained),required_remaining_bytes=required,base_deficit=max(0,required-after),cleanup_target_bytes=required+2*1024**3,cleanup_target_deficit=max(0,required+2*1024**3-after),research_payloads_modified=False,note='APFS accounting and concurrent activity can make observed free delta differ from allocated bytes removed.')
  journal(dict(event='complete',**result));save('completed.json',result);print(json.dumps(result))
