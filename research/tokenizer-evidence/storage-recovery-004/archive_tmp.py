"""Lossless archive of exact inactive pytest basetemp trees; never research weights."""
import gzip,hashlib,json,os,shutil,stat,subprocess,sys,tarfile,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R));E=Path(__file__).resolve().parent
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.ffn_norm_quality_v1.support import idle
BASE=Path('/private/tmp');ARCHIVE=R/'research_archives/research-pytest-temp-004.tar.zst'
LOGS=R/'research/tokenizer-evidence/wide-hybrid-preparation-001'
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,v):
 with p.open('x') as f:json.dump(v,f);f.write('\n');f.flush();os.fsync(f.fileno())
def sync(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def open_check(roots):
 out=subprocess.run(['/usr/sbin/lsof','-nP','-Fpn'],capture_output=True,text=True,timeout=30)
 assert out.returncode in (0,1)
 names=[line[1:] for line in out.stdout.splitlines() if line.startswith('n')]
 assert not any(n==str(BASE/r) or n.startswith(str(BASE/r)+'/') for r in roots for n in names),'Fixture is open in a live process'
 procs=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True)
 assert not any('pytest' in line and int(line.strip().split()[0])!=os.getpid() and 'archive_tmp.py' not in line for line in procs.splitlines()),'Pytest active'
def record(p):
 s=p.lstat();v=dict(path=str(p.relative_to(BASE)),mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,size=s.st_size,allocated=s.st_blocks*512,uid=s.st_uid,gid=s.st_gid)
 if stat.S_ISLNK(s.st_mode):v.update(kind='symlink',target=os.readlink(p))
 elif stat.S_ISDIR(s.st_mode):v.update(kind='directory')
 else:
  assert stat.S_ISREG(s.st_mode);v.update(kind='file',sha256=sha(p));assert p.stat().st_mtime_ns==s.st_mtime_ns
 return v
def inventory(roots):
 result=[]
 for name in roots:
  root=BASE/name;result.append(record(root))
  for parent,dirs,files in os.walk(root,followlinks=False):
   for name in sorted(dirs+files):result.append(record(Path(parent)/name))
 return result
def plan():
 roots=[];evidence=[]
 for p in sorted(BASE.iterdir()):
  if not p.is_dir() or p.is_symlink() or not p.name.endswith('-fixtures'):continue
  stem=p.name[:-9];logs=[q for q in (LOGS/f'{stem}-tests.log',LOGS/f'{stem}.log') if q.exists()]
  if not logs:continue
  assert p.stat().st_mtime<time.time()-3600
  roots.append(p.name);evidence.append(dict(path=str(logs[0]),sha256=sha(logs[0])))
 open_check(roots);rows=inventory(roots);assert len(roots)==144 and len(rows)<500000
 assert all(r['size']<16*1024**2 for r in rows if r['kind']=='file'),'Unexpected large payload in disposable fixtures'
 with gzip.open(E/'tmp-plan.json.gz','xt') as f:json.dump(dict(roots=roots,files=rows,evidence=evidence,source_sha256=sha(Path(__file__)),archive=str(ARCHIVE),free_before=shutil.disk_usage(R).free,authorization='Latest human asks another way to reclaim capacity while preserving selected and fresh models. Archive recorded inactive pytest fixture trees losslessly, retain sources/test logs and exact original path manifest; never rerun closed tests.'),f,separators=(',',':'))
 print(json.dumps(dict(roots=len(roots),entries=len(rows),allocated=sum(x['allocated'] for x in rows),logical=sum(x['size'] for x in rows if x['kind']=='file'))),flush=True)
def apply():
 with gzip.open(E/'tmp-plan.json.gz','rt') as f:p=json.load(f)
 assert p['source_sha256']==sha(Path(__file__)) and not ARCHIVE.exists() and not (E/'tmp-journal.jsonl').exists()
 open_check(p['roots']);assert inventory(p['roots'])==p['files']
 for r in p['evidence']:assert sha(Path(r['path']))==r['sha256']
 ARCHIVE.parent.mkdir(exist_ok=True)
 with (E/'tmp-zstd.log').open('xb') as err:
  proc=subprocess.Popen(['/opt/homebrew/bin/zstd','-3','-T1','--no-progress','-o',str(ARCHIVE)],stdin=subprocess.PIPE,stderr=err)
  try:
   with tarfile.open(fileobj=proc.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
    for r in p['files']:
     q=BASE/r['path'];assert record(q)==r
     ti=tarfile.TarInfo(r['path']);ti.mode=r['mode'];ti.uid=r['uid'];ti.gid=r['gid'];ti.mtime=r['mtime_ns']/1e9
     if r['kind']=='directory':ti.type=tarfile.DIRTYPE;tar.addfile(ti)
     elif r['kind']=='symlink':ti.type=tarfile.SYMTYPE;ti.linkname=r['target'];tar.addfile(ti)
     else:
      ti.size=r['size']
      with q.open('rb') as f:tar.addfile(ti,f)
     assert (not ARCHIVE.exists() or ARCHIVE.stat().st_size<512*1024**2) and shutil.disk_usage(R).free>=8*1024**3
   proc.stdin.close();assert proc.wait(timeout=60)==0
  finally:
   if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
 with ARCHIVE.open('rb') as f:os.fsync(f.fileno())
 expected={r['path']:r for r in p['files']};seen=set()
 proc=subprocess.Popen(['/opt/homebrew/bin/zstd','-d','--stdout',str(ARCHIVE)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 try:
  with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
   for member in tar:
    r=expected[member.name];assert member.name not in seen;seen.add(member.name)
    assert member.mode==r['mode'] and member.uid==r['uid'] and member.gid==r['gid']
    if r['kind']=='directory':assert member.isdir()
    elif r['kind']=='symlink':assert member.issym() and member.linkname==r['target']
    else:
     assert member.isfile() and member.size==r['size']
     f=tar.extractfile(member);h=hashlib.sha256()
     while chunk:=f.read(1024**2):h.update(chunk)
     assert h.hexdigest()==r['sha256']
  while proc.stdout.read(1024**2):pass
  assert proc.wait(timeout=60)==0 and seen==set(expected)
 finally:
  if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
 archive_sha=sha(ARCHIVE);save(E/'tmp-verified.json',dict(archive=str(ARCHIVE),sha256=archive_sha,size=ARCHIVE.stat().st_size,entries=len(seen),complete_decompression_verified=True,manifest_sha256=sha(E/'tmp-plan.json.gz')))
 open_check(p['roots']);assert inventory(p['roots'])==p['files']
 with (E/'tmp-journal.jsonl').open('x') as log:
  # One durable batch intent binds every exact path/hash before any removal.
  log.write(json.dumps(dict(event='intent_remove_verified_archived_entries',manifest_sha256=sha(E/'tmp-plan.json.gz'),archive_sha256=archive_sha))+'\n');log.flush();os.fsync(log.fileno())
  for r in sorted(p['files'],key=lambda x:len(Path(x['path']).parts),reverse=True):
   q=BASE/r['path']
   if r['kind']=='directory':q.rmdir()
   else:
    assert record(q)==r;q.unlink()
   log.write(json.dumps(dict(event='removed_archived_entry',path=r['path']))+'\n')
  log.flush();os.fsync(log.fileno());sync(BASE)
 assert all(not (BASE/n).exists() for n in p['roots']) and sha(ARCHIVE)==archive_sha
 for r in p['evidence']:assert sha(Path(r['path']))==r['sha256']
 save(E/'tmp-completed.json',dict(status='completed',archive=str(ARCHIVE),archive_sha256=archive_sha,archive_bytes=ARCHIVE.stat().st_size,entries=len(p['files']),roots=len(p['roots']),allocated_before=sum(r['allocated'] for r in p['files']),free_after=shutil.disk_usage(R).free,all_bytes_preserved_in_verified_archive=True,selected_and_fresh_paths_outside_scope=True,restoration='Resolve exact original /private/tmp paths through tmp-plan.json.gz and verified tar.zst. Preserve failed-test fixtures and original test logs; no closed-fixture rerun.'))
 print((E/'tmp-completed.json').read_text(),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('plan','apply')
 with session_lock(GLOBAL_LOCK.parent),CpuPrerequisiteGuard(E/f'tmp-{mode}-guard.json',R,timeout_seconds=900):
  idle();plan() if mode=='plan' else apply()
