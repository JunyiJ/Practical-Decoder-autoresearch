"""Only the four newly regenerated cache files from the final worker-source proof."""
import fcntl,hashlib,json,os,shutil,stat,subprocess
from pathlib import Path
R=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');E=Path(__file__).resolve().parent;C=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/org.python.python/com.apple.metal')
P=[C/'16777235_403/functions.data',C/'16777235_403/functions.list',C/'32024/libraries.data',C/'32024/libraries.list']
def row(p):
 s=p.lstat();assert p.resolve()==p and not p.is_symlink() and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=1,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=h)
def consumers():
 ps=subprocess.check_output(['ps','-axo','command='],text=True)
 assert not any('python' in s.lower() and ('worker-source-review001.py' in s or '-m research.' in s or '/resume001.py' in s or '/correctness001.py' in s) for s in ps.splitlines())
 p=subprocess.run(['lsof','-nP',*map(str,P)],capture_output=True,text=True);assert p.returncode==1 and not p.stdout.strip() and not p.stderr.strip()
def save(n,d):
 with (E/n).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with (R/'research_state/external-cache-maintenance.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(E/'manifest.json').exists();consumers()
 prior=json.load((R/'research/maintenance-inline-input-dropout-001/manifest.json').open())['entries'];old={(r['path'],r['inode'],r['mtime_ns'],r['sha256']) for r in prior};entries=[row(p) for p in P]
 assert all((r['path'],r['inode'],r['mtime_ns'],r['sha256']) not in old for r in entries)
 save('manifest.json',dict(reason='New actual cache regeneration after final958-source worker-route/RNG proof ended;distinct identity from closed001,unopened disposable Python compiler data/list only.',entries=entries,required=19590082560,target=21737566208,free_before=shutil.disk_usage(R).free))
 with (E/'journal.jsonl').open('x') as log:
  for p,r in zip(P,entries):
   consumers();assert row(p)==r;log.write(json.dumps(dict(event='intent_remove_exact_unused_compiler_cache',file=r))+'\n');log.flush();os.fsync(log.fileno());p.unlink();fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);log.write(json.dumps(dict(event='removed',path=str(p)))+'\n');log.flush();os.fsync(log.fileno())
 free=shutil.disk_usage(R).free;result=dict(status='completed',removed_files=4,allocated_removed=sum(r['allocated'] for r in entries),free_after=free,required=19590082560,target=21737566208,mandatory_deficit=max(0,19590082560-free),target_deficit=max(0,21737566208-free),no_apps_workers_stopped=True,no_research_or_runtime_bytes_modified=True)
 save('completed.json',result);print(json.dumps(result))
