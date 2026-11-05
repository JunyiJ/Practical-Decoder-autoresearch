"""New bounded external-cache recovery during the active first fresh trajectory.

Never takes the busy research lock, touches research payloads, stops a consumer,
or reuses a closed maintenance script. Installed apps/updaters/data are excluded.
"""
import fcntl,hashlib,json,os,shutil,stat,subprocess,time
from pathlib import Path
R=Path.cwd();B=Path(__file__).resolve().parent;E=B/'quality001-active-capacity-recovery001';E.mkdir()
HOME_DIR=Path.home()
ROOTS=[HOME_DIR/'Library/Caches/Homebrew/downloads',HOME_DIR/'Library/Caches/pip',HOME_DIR/'.cache/pip',HOME_DIR/'.cache/uv',HOME_DIR/'.npm/_cacache',HOME_DIR/'.npm/_npx',HOME_DIR/'Library/Caches/go-build',HOME_DIR/'Library/Caches/typescript']

def publish(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def rec(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
 return dict(path=str(p),size=s.st_size,allocated=s.st_blocks*512,uid=s.st_uid,gid=s.st_gid,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,links=s.st_nlink,sha256=sha(p))
fd=os.open(R/'research_state/external-cache-maintenance.lock',os.O_RDWR|os.O_CREAT,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
 raw=subprocess.check_output(['/bin/ps','-axo','pid=,command='],text=True)
 consumers=[]
 for line in raw.splitlines():
  fields=line.strip().split(None,1)
  if len(fields)!=2 or int(fields[0])==os.getpid():continue
  cmd=fields[1];tags=[s for s in (' -m pip ','pip install','npm install','npm ci','npx ','brew install','uv pip','uv sync','go build','go test','typingsInstaller','tsserver') if s in cmd]
  if tags:consumers.append(dict(pid=int(fields[0]),tags=tags))
 inventory=[];eligible=[];blocked=[];seen=set();started=time.monotonic()
 for root in ROOTS:
  if not root.exists():inventory.append(dict(root=str(root),status='absent'));continue
  assert not root.is_symlink();count=0;allocated=0
  for folder,dirs,names in os.walk(root,followlinks=False):
   dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]
   for name in names:
    assert time.monotonic()-started<120
    p=Path(folder)/name;s=p.lstat()
    if not stat.S_ISREG(s.st_mode):continue
    inode=(s.st_dev,s.st_ino)
    if inode in seen:continue
    seen.add(inode);count+=1;allocated+=s.st_blocks*512
    if s.st_uid!=os.getuid() or s.st_nlink!=1 or not os.access(p,os.W_OK) or not os.access(p.parent,os.W_OK):blocked.append(dict(path=str(p),uid=s.st_uid,size=s.st_size,allocated=s.st_blocks*512,reason='Existing ownership/link/normal filesystem access blocker'));continue
    if consumers:continue
    eligible.append(dict(**rec(p),reason='Disposable package/download/compiler cache; no package/build/type-acquisition consumer; no installed runtime,app settings,credentials,models or data'))
  inventory.append(dict(root=str(root),files=count,unique_allocated_bytes=allocated))
 assert sum(x['allocated'] for x in eligible)<=512*1024**2 and len(eligible)<=10000
 opened=[]
 if eligible:
  check=subprocess.run(['/usr/sbin/lsof','-nP','-Fpn','--',*[x['path'] for x in eligible]],capture_output=True,text=True,timeout=30)
  assert check.returncode==1 and not check.stdout.strip() and not check.stderr.strip(),'External cache has an open consumer'
 before=shutil.disk_usage(R).free
 publish(E/'manifest.json',dict(operation='new_active_quality_external_cache_recovery',source=dict(path=str(Path(__file__).resolve()),sha256=sha(Path(__file__))),inventory=inventory,eligible=eligible,blocked=blocked,consumer_processes=consumers,free_before=before,required_full_free_bytes=12381585408,cleanup_target_bytes=14529069056,research_payloads_untouched=True,scope='Recheck newly available package/build cache candidates only after changed forward capacity. Other classes remain exhausted per maintenance012,active Sparkle updater remains protected; no duplicate closed cleanup or app/process stopping.'))
 with (E/'journal.jsonl').open('x') as journal:
  def event(v):journal.write(json.dumps(v)+'\n');journal.flush();os.fsync(journal.fileno())
  for row in eligible:
   p=Path(row['path']);assert rec(p)=={k:v for k,v in row.items() if k!='reason'};event(dict(event='intent',file=row));p.unlink();d=os.open(p.parent,os.O_RDONLY);os.fsync(d);os.close(d);event(dict(event='removed',path=row['path']))
 after=shutil.disk_usage(R).free
 publish(E/'completed.json',dict(status='completed_new_eligible_cache_recovery' if eligible else 'verified_current_external_options_exhausted',removed_files=len(eligible),allocated_bytes_removed=sum(x['allocated'] for x in eligible),free_before=before,free_after=after,forward_full_budget_deficit=max(0,12381585408-after),cleanup_target_deficit=max(0,14529069056-after),research_payloads_modified=False,apps_or_workers_stopped=False,known_remaining_classes='Root-owned npm/admin-blocked unused Keynote;active/pending updaters;unmatched browser archives;live app caches/shared runtimes/active containers;all selected/fresh research payloads protected. No new proof permits their deletion.',next_action='Leave active guarded slot00 running. On clean independent acceptance,measure remaining full budget before slot01; recover on new eligible evidence or wait if still insufficient. Do not retry closed slot or launch below remaining full budget.'))
 print(json.dumps(json.loads((E/'completed.json').read_text())),flush=True)
finally:os.close(fd)
