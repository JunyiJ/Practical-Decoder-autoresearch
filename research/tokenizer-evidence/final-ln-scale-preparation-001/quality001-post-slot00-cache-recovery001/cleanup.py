"""Retire only newly regenerated unused Metal compiler-cache bytes after slot00.

A new exact input set after completed accepted training, not a closed cleanup
rerun. No research payload is deleted; active/pending app data is excluded.
"""
import datetime,fcntl,hashlib,json,os,shutil,stat,subprocess
from pathlib import Path
R=Path.cwd();E=Path(__file__).resolve().parent;B=E.parent
CACHE=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/org.python.python/com.apple.metal')

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def info(p):
 s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
 return dict(path=str(p),size=s.st_size,allocated=s.st_blocks*512,dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=s.st_mode,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,sha256=sha(p),reason='Regenerated disposable Metal compiled function/library cache; accepted slot00 process ended; no Python worker/open consumer; regenerate on demand')
def publish(n,v):
 with (E/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def readref(ref):
 p=Path(ref['path']);assert sha(p)==ref['sha256'];return json.loads(p.read_text())
def consumers(paths):
 processes=subprocess.check_output(['/bin/ps','-axo','pid=,comm='],text=True);seen=[]
 for line in processes.splitlines():
  pid,cmd=line.strip().split(maxsplit=1)
  assert int(pid)==os.getpid() or 'python' not in cmd.lower(),('Python consumer present',pid)
  if int(pid)!=os.getpid():seen.append(int(pid))
 if paths:
  result=subprocess.run(['/usr/sbin/lsof','-nP','-Fpn','--',*[str(p) for p in paths]],capture_output=True,text=True,timeout=30);assert result.returncode==1 and not result.stdout.strip() and not result.stderr.strip()
 return dict(other_pids=seen,no_other_python=True,no_open_target_handles=True)
with open(R/'research_state/external-cache-maintenance.lock','a+b') as ext,open(R/'research_state/tokenizer-worker-lock/.lock','a+b') as work:
 fcntl.flock(ext,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(work,fcntl.LOCK_EX|fcntl.LOCK_NB)
 anchor=json.loads((B/'quality001-accepted-00.json').read_text());receipt=readref(anchor['acceptance']);worker=readref(receipt['worker']);req=worker['acceptance_request'];parent=readref(anchor['parent_guard']);assert receipt['slot']==0 and receipt['artifact_acceptance']['accepted'] and parent['status']=='finished' and parent['reason'] is None
 retained={k:dict(path=v['path'],sha256=sha(Path(v['path']))) for k,v in req['files'].items()};assert all(retained[k]['sha256']==v['sha256'] for k,v in req['files'].items())
 paths=sorted(p for p in CACHE.rglob('*') if p.is_file()) if CACHE.exists() else []
 assert len(paths)<=16 and all(p.name in ('functions.data','functions.list','functions1.data','functions1.list','libraries.data','libraries.list','libraries1.data','libraries1.list') for p in paths)
 rows=[info(p) for p in paths];assert sum(r['allocated'] for r in rows)<=128*1024**2
 old=json.loads((R/'research/maintenance-cache-cleanup-012/manifest.json').read_text());old_identity={(r['path'],r['ino'],r['mtime_ns'],r['sha256']) for r in old['entries']}
 assert all((r['path'],r['ino'],r['mtime_ns'],r['sha256']) not in old_identity for r in rows)
 checked=consumers(paths);before=shutil.disk_usage(R).free
 publish('manifest.json',dict(entries=rows,consumer_check=checked,closed_slot_acceptance=anchor['acceptance'],source=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),affected_retained_bundle=retained,free_before=before,requirement_remaining=11822322392,cleanup_target_remaining=11822322392+2*1024**3,authorization='Standing disposable compiler-cache recovery after no active Python/open consumer. Fresh RAW/H200/source/config/log/metrics/receipts preserved. New inode/hash/mtime identities distinct from closed maintenance012.'))
 assert rows==[info(p) for p in paths];consumers(paths)
 with (E/'journal.jsonl').open('x') as log:
  def event(v):log.write(json.dumps(dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),**v))+'\n');log.flush();os.fsync(log.fileno())
  for p,row in zip(paths,rows):
   assert info(p)==row;event(dict(event='intent_delete_disposable_cache',file=row));p.unlink();d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d);event(dict(event='removed',path=str(p)))
 for binding in retained.values():assert sha(Path(binding['path']))==binding['sha256']
 after=shutil.disk_usage(R).free;value=dict(status='completed_new_regenerated_cache_recovery',files=len(rows),allocated_bytes_removed=sum(r['allocated'] for r in rows),affected_retained_bundle_files_verified=len(retained),free_before=before,free_after=after,required_remaining_bytes=11822322392,remaining_deficit=max(0,11822322392-after),extra_target_deficit=max(0,11822322392+2*1024**3-after),research_payloads_modified=False,apps_or_workers_stopped=False)
 publish('completed.json',value);print(json.dumps(value),flush=True)
