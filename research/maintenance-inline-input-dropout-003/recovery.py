"""Complete only the three unremoved allowlisted cache files after closed002 diagnostic."""
import fcntl,hashlib,json,os,shutil,stat,subprocess
from pathlib import Path
R=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');E=Path(__file__).resolve().parent;P=R/'research/maintenance-inline-input-dropout-002'
def row(p):
 s=p.lstat();assert p.resolve()==p and not p.is_symlink() and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=1,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=h)
def consumers(paths):
 ps=subprocess.check_output(['ps','-axo','command='],text=True)
 assert not any('python' in s.lower() and ('worker-source-review001.py' in s or '-m research.' in s or '/resume001.py' in s or '/correctness001.py' in s) for s in ps.splitlines())
 p=subprocess.run(['lsof','-nP',*paths],capture_output=True,text=True);assert p.returncode==1 and not p.stdout.strip() and not p.stderr.strip()
def save(n,d):
 with (E/n).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with (R/'research_state/external-cache-maintenance.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(E/'manifest.json').exists()
 original=json.load((P/'manifest.json').open());events=[json.loads(s) for s in (P/'journal.jsonl').read_text().splitlines()];removed={e['path'] for e in events if e['event']=='removed'}
 assert len(removed)==1 and all(not Path(p).exists() for p in removed)
 entries=[r for r in original['entries'] if r['path'] not in removed];assert len(entries)==3
 assert all(row(Path(r['path']))==r for r in entries);consumers([r['path'] for r in entries])
 save('manifest.json',dict(entries=entries,original_manifest_sha256=hashlib.sha256((P/'manifest.json').read_bytes()).hexdigest(),original_journal_sha256=hashlib.sha256((P/'journal.jsonl').read_bytes()).hexdigest(),closed_original_exception_preserved=True,already_removed_paths=sorted(removed),reason='Original002 stopped because lsof was asked about an already removed path;new003 checks only still-existing remaining exact files,never reruns002 or restores removed cache.',required=19590082560,target=21737566208))
 with (E/'journal.jsonl').open('x') as j:
  for i,r in enumerate(entries):
   consumers([e['path'] for e in entries[i:]]);p=Path(r['path']);assert row(p)==r;j.write(json.dumps(dict(event='intent_remove_exact_remaining_unused_cache',file=r))+'\n');j.flush();os.fsync(j.fileno());p.unlink();fd=os.open(p.parent,os.O_RDONLY);os.fsync(fd);os.close(fd);j.write(json.dumps(dict(event='removed',path=str(p)))+'\n');j.flush();os.fsync(j.fileno())
 assert all(not Path(r['path']).exists() for r in original['entries']);free=shutil.disk_usage(R).free
 result=dict(status='completed_three_remaining_exact_caches',removed_files=3,allocated_removed=sum(r['allocated'] for r in entries),free_after=free,required=19590082560,target=21737566208,mandatory_deficit=max(0,19590082560-free),target_deficit=max(0,21737566208-free),original002_closed_partial_deletion_preserved=True,no_restore_or_closed_rerun=True,no_apps_workers_stopped=True)
 save('completed.json',result);print(json.dumps(result))
