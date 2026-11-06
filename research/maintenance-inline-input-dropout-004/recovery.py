"""New regenerated compiler cache after accepted worker pair;independent of CPU retirement."""
import fcntl,hashlib,json,os,shutil,stat,subprocess,re
from pathlib import Path
R=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');E=Path(__file__).resolve().parent;C=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/org.python.python/com.apple.metal')
ALLOW={f'{d}/{base}{n}.{ext}' for d,base in [('16777235_403','functions'),('32024','libraries')] for n in ('','1') for ext in ('data','list')}
def row(p):
 s=p.lstat();assert p.resolve()==p and not p.is_symlink() and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=1,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=h)
def consumers(paths):
 ps=subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True)
 for line in ps.splitlines():
  p=line.strip().split(None,2)
  if len(p)==3 and Path(p[1]).name.lower().startswith('python') and int(p[0])!=os.getpid():
   assert not('-m research.' in p[2] or re.search(r'/(correctness\d+|resume\d+|worker\d+-build-run\d+|worker\d+-import\d+)\.py',p[2])),'Numerical consumer active'
 result=subprocess.run(['lsof','-nP',*paths],capture_output=True,text=True);assert result.returncode==1 and not result.stdout.strip() and not result.stderr.strip()
def save(n,d):
 with (E/n).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with (R/'research_state/external-cache-maintenance.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(E/'manifest.json').exists()
 for n in ('001','002'):assert (R/f'research_runs/tokenizer-embedding-input-dropout-worker-v1-{n}/STOP').is_file()
 review=json.load((R/'research/tokenizer-evidence/embedding-input-dropout-preparation-001/worker002-review.json').open());assert review['status']=='two_generated_workers_independently_accepted'
 old=set()
 for n in ('001','002','003'):
  for r in json.load((R/f'research/maintenance-inline-input-dropout-{n}/manifest.json').open())['entries']:old.add((r['path'],r['inode'],r['mtime_ns'],r['sha256']))
 entries=[];retained=[]
 for p in sorted(C.rglob('*')):
  if not p.is_file() or p.is_symlink():continue
  r=row(p)
  if str(p.relative_to(C)) in ALLOW and (r['path'],r['inode'],r['mtime_ns'],r['sha256']) not in old:entries.append(r)
  else:retained.append(r)
 assert entries;consumers([r['path'] for r in entries]);before=shutil.disk_usage(R).free
 save('manifest.json',dict(entries=entries,retained=retained,reason='New exact compiler-cache generations after completed original/control-import/candidate worker-parent processes. CPU-only research survivor audit may run concurrently under independent lock;no MPS worker active. No installed runtime/library or research payload in allowlist.',required=11050479616,target=13197963264,free_before=before))
 with (E/'journal.jsonl').open('x') as j:
  for i,r in enumerate(entries):
   consumers([v['path'] for v in entries[i:]]);p=Path(r['path']);assert row(p)==r;j.write(json.dumps(dict(event='intent_remove_exact_unused_regenerated_cache',file=r))+'\n');j.flush();os.fsync(j.fileno());p.unlink();fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);j.write(json.dumps(dict(event='removed',path=str(p)))+'\n');j.flush();os.fsync(j.fileno())
 for r in retained:assert row(Path(r['path']))==r
 free=shutil.disk_usage(R).free;result=dict(status='completed',removed_files=len(entries),allocated_removed=sum(r['allocated'] for r in entries),free_before=before,free_after=free,required=11050479616,target=13197963264,
  no_apps_workers_stopped=True,no_research_or_runtime_bytes_modified=True,note='Concurrent research-retirement/application activity can affect free-space delta;allocated removed bytes are exact allowlist totals only.')
 save('completed.json',result);print(json.dumps(result))
