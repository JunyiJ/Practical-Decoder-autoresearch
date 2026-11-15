"""One-shot eviction of verified, unopened Chromium SimpleCache payload entries."""
import struct,os,stat,json,hashlib,fcntl,subprocess,re,datetime,time
from pathlib import Path
ROOT=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');OUT=Path(__file__).resolve().parent
BASES=[Path('/Users/junyi/Library/Caches/Google/Chrome/Default/Cache/Cache_Data')]
MAGIC=bytes.fromhex('305c72a71b6dfbfc')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def row(p):
 s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
 return dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=sha(p))
def opens():
 r=subprocess.run(['/usr/sbin/lsof','-nP','-Fn'],capture_output=True,text=True);assert r.returncode==0 and not r.stderr,r.stderr
 return {s[1:] for s in r.stdout.splitlines() if s.startswith('n')}
def save(n,x):
 with open(OUT/n,'x') as f:json.dump(x,f,indent=2);f.flush();os.fsync(f.fileno())
def free():
 s=os.statvfs(ROOT);return s.f_bavail*s.f_frsize
with open(ROOT/'research_state/external-cache-maintenance.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'manifest.json').exists()
 open_paths=opens();rows=[];preserved=[]
 for base in BASES:
  for p in base.rglob('*'):
   if p.is_symlink() or not p.is_file() or not re.fullmatch('[0-9a-f]{16}_s',p.name) or str(p) in open_paths:continue
   with p.open('rb') as f:header=f.read(8)
   if header!=MAGIC or time.time()-p.stat().st_mtime<3600:continue
   if any(p.with_name(p.name[:-1]+suffix).exists() for suffix in ('0','1')):continue
   with p.open('rb') as f:
    raw=f.read(24)
    magic,version,keylen,keyhash,padding=struct.unpack('<QIIII',raw)
    if version!=9 or padding!=0 or keylen>65536:continue
    f.seek(24+keylen);valid=True;nranges=0;size=p.stat().st_size
    while f.tell()<size:
     rh=f.read(32)
     if len(rh)!=32:valid=False;break
     rm,offset,length,crc,pad=struct.unpack('<QQQII',rh)
     if rm!=0xeb97bf016553676b or pad!=0 or f.tell()+length>size:valid=False;break
     f.seek(length,1);nranges+=1
    if valid and nranges and f.tell()==size:rows.append(row(p))
 rows=sorted(rows,key=lambda e:e['mtime_ns']); selected=[]; allocated=0
 for e in rows:
  selected.append(e);allocated+=e['allocated']
  if allocated>=256*1024**2:break
 rows=selected
 before=free();save('manifest.json',{'reason':'Orphaned unopened Chromium sparse HTTP cache payloads older than one hour. No dense siblings. Validate header/version9/key length and every sparse range framing using the established audit. Exact hashes and consumer rechecks. Preserve all user/application state.','entries':rows,'free_before':before})
 open_paths=opens();removed=[];skipped=[]
 with open(OUT/'deletion-journal.jsonl','x') as j:
  def log(x):j.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**x})+'\n');j.flush();os.fsync(j.fileno())
  for i,e in enumerate(rows):
   if i%100==0:open_paths=opens()
   p=Path(e['path'])
   if str(p) in open_paths or not p.exists() or row(p)!=e:skipped.append(str(p));continue
   log({'event':'intent','entry':e});p.unlink();removed.append(e);log({'event':'removed','path':str(p)})
 after=free();result={'status':'completed','removed_files':len(removed),'skipped_changed_or_open':skipped,'allocated_bytes_removed':sum(e['allocated'] for e in removed),'free_before':before,'free_after':after,'observed_free_delta':after-before,'required_remaining_bytes':19079458222,'cleanup_target_bytes':19079458222,'cleanup_target_deficit':max(0,19079458222-after),'note':'APFS accounting and concurrent application/research writes affect free-space delta; cache misses regenerate from originals.'}
 save('completed.json',result);print(json.dumps(result))
