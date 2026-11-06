"""Fresh human disk-cleanup request: exact new unopened disposable cache payloads only."""
import fcntl,hashlib,json,os,re,shutil,stat,subprocess
from pathlib import Path
R=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');E=Path(__file__).resolve().parent
REQUIRED=14893920256;TARGET=17039945728
METAL=Path('/private/var/folders/kq/88lvc8ps3s5_4jpn83v68f8c0000gn/C/org.python.python/com.apple.metal');PIP=Path.home()/'Library/Caches/pip'
ROOTS=[METAL,PIP,*map(Path,[str(Path.home()/'.cache/pip'),str(Path.home()/'.cache/uv'),str(Path.home()/'.npm/_cacache'),str(Path.home()/'Library/Caches/Homebrew'),str(Path.home()/'anaconda3/pkgs'),str(Path.home()/'Library/Caches/Google/Chrome'),str(Path.home()/'Library/Application Support/Code/CachedData')])]
ALLOWED_METAL={f'{d}/{base}{n}.{ext}' for d,base in [('16777235_403','functions'),('32024','libraries')] for n in ('','1') for ext in ('data','list')}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def row(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and p.resolve()==p and s.st_uid==os.getuid() and s.st_nlink==1
 return dict(path=str(p),device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=sha(p))
def save(name,d):
 with (E/name).open('x') as f:json.dump(d,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def opens():
 p=subprocess.run(['lsof','-nP','-Fn'],capture_output=True,text=True);assert p.returncode==0 and not p.stderr,p.stderr
 return {s[1:] for s in p.stdout.splitlines() if s.startswith('n')}
def processes():return subprocess.check_output(['ps','-axo','pid=,comm=,args='],text=True)
def active_mps(ps):
 rows=[]
 for line in ps.splitlines():
  parts=line.strip().split(None,2)
  if len(parts)==3 and Path(parts[1]).name.lower().startswith('python') and int(parts[0])!=os.getpid() and ('-m research.' in parts[2] or re.search(r'/(correctness\d+|resume\d+|resume-metadata\d+|diagnostic\d+|review\d+)\.py',parts[2])):rows.append(line)
 return rows
def syncdir(p):
 fd=os.open(p,os.O_RDONLY);os.fsync(fd);os.close(fd)
with (R/'research_state/external-cache-maintenance.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(E/'manifest.json').exists();ps=processes();busy=active_mps(ps);opened=opens()
 installers=[s for s in ps.splitlines() if re.search(r'\b(pip|uv|conda|mamba|brew) (install|update|upgrade|download|build|sync)',s)]
 old=[]
 for folder in (R/'research').glob('maintenance*'):
  p=folder/'manifest.json'
  if p.exists():
   d=json.load(p.open());old.extend(d.get('entries',[]) if isinstance(d.get('entries'),list) else [])
 old_ids={(r.get('path'),r.get('inode',r.get('ino')),r.get('mtime_ns'),r.get('sha256')) for r in old}
 inventory=[];candidates=[];retained_metal=[]
 for root in ROOTS:
  unique={};errors=[];root_owned=0
  if root.exists():
   for parent,dirs,names in os.walk(root,followlinks=False,onerror=lambda e:errors.append(str(e))):
    dirs[:]=[d for d in dirs if not(Path(parent)/d).is_symlink()]
    for name in names:
     p=Path(parent)/name
     if p.is_symlink():continue
     s=p.lstat()
     if not stat.S_ISREG(s.st_mode):continue
     unique[(s.st_dev,s.st_ino)]=s.st_blocks*512
     if s.st_uid!=os.getuid():root_owned+=1;continue
     reason=None
     if root==METAL and not busy and str(p.relative_to(root)) in ALLOWED_METAL and str(p) not in opened:
      reason='New regenerated per-user Python Metal function/library compiler data/list cache;workers ended,no open consumer;not an installed runtime/metallib/source/checkpoint.'
     elif root==PIP and not installers and p.parts[len(PIP.parts)] in ('http-v2','wheels') and str(p) not in opened:
      reason='Completed unopened pip download/wheel cache,no installer;installed environments remain intact.'
     elif root==Path.home()/'Library/Caches/Homebrew' and not installers and p.parent==root/'api' and p.name in ('formula.jws.json','cask.jws.json') and str(p) not in opened:
      d=json.load(p.open())
      if set(d)=={'payload','signatures'} and isinstance(d['payload'],str) and d['signatures']:reason='Regenerable signed API catalog;no brew process,not an installer/updater payload.'
     elif root==Path.home()/'Library/Caches/Google/Chrome' and re.fullmatch('[0-9a-f]{16}_0',p.name) and str(p) not in opened:
      with p.open('rb') as f:magic=f.read(8)
      if magic==bytes.fromhex('305c72a71b6dfbfc'):reason='Unopened self-identifying Chromium SimpleCache payload in cache-only root;no profile/index/session/extension/local-storage/settings bytes. A later cache miss is regenerable.'
     # Unknown archives/downloads, shared runtimes and app data are inventoried
     # without inferring disposal from size, suffix or absent process alone.
     if reason and s.st_nlink==1:
      item=row(p);identity=(item['path'],item['inode'],item['mtime_ns'],item['sha256'])
      if identity not in old_ids:candidates.append(dict(**item,reason=reason))
     elif root==METAL and s.st_nlink==1:retained_metal.append(row(p))
  inventory.append(dict(root=str(root),unique_allocated=sum(unique.values()),files=len(unique),root_owned_files=root_owned,errors=errors))
 apps=[]
 for p in (Path('/Applications/Keynote.app'),):
  if p.exists():
   s=p.stat();apps.append(dict(path=str(p),uid=s.st_uid,gid=s.st_gid,mode=oct(s.st_mode&0o777),normal_user_writable=os.access(p,os.W_OK),decision='Human-unused identification retained;administrator/access blocker unchanged,no permission bypass.'))
 before=shutil.disk_usage(R).free;candidates.sort(key=lambda r:r['allocated'],reverse=True)
 save('manifest.json',dict(required=REQUIRED,target=TARGET,free_before=before,reason='New direct human disk-cleanup request. Exact new cache identities, unique allocation, no consumers, ordinary permission checks; independent of coordinated CPU-only historical checkpoint audit.',inventory=sorted(inventory,key=lambda r:r['unique_allocated'],reverse=True),entries=candidates,retained_metal=retained_metal,active_mps=busy,active_installers=installers,unused_app_access=apps,
  other_classes='Current body24/finalscale12/H100/storage004 remain protected. Human separately authorizes retirement of exact superseded historical archive/H200 pairs through a NEW research-lock audit;this external-cache pass touches none of them. Closed probes are never rerun. Installed/shared runtimes,active containers,pending updaters,unmatched archives/downloads,personal files and live app caches preserved. New closed metadata handled separately under research lock.'))
 removed=[];skipped=[]
 with (E/'journal.jsonl').open('x') as j:
  def event(d):j.write(json.dumps(d)+'\n');j.flush();os.fsync(j.fileno())
  for c in candidates:
   p=Path(c['path']);current=processes();nowopen=opens()
   if str(p) in nowopen or (p.is_relative_to(METAL) and active_mps(current)) or not p.exists() or row(p)!={k:v for k,v in c.items() if k!='reason'}:skipped.append(str(p));continue
   event(dict(event='intent_remove_exact_verified_disposable_cache',file=c));p.unlink();syncdir(p.parent);event(dict(event='removed',path=str(p)));removed.append(c)
 for r in retained_metal:assert row(Path(r['path']))==r
 free=shutil.disk_usage(R).free
 result=dict(status='completed',removed_files=len(removed),allocated_removed=sum(r['allocated'] for r in removed),skipped=skipped,free_before=before,free_after=free,required=REQUIRED,target=TARGET,mandatory_deficit=max(0,REQUIRED-free),target_deficit=max(0,TARGET-free),retained_metal_hashes_verified=len(retained_metal),no_apps_workers_stopped=True,no_research_payload_or_runtime_modified=True)
 save('completed.json',result);print(json.dumps(result))
