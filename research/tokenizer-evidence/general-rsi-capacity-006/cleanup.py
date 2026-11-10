"""Bounded external-cache recovery for the remaining frozen general RSI panel."""
import datetime,fcntl,hashlib,json,os,re,stat,subprocess
from pathlib import Path
ROOT=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI')
OUT=Path(__file__).resolve().parent
BASE=Path('/Users/junyi/Library/Caches/Google/Chrome/Default')
BASES=[BASE/'Cache/Cache_Data',BASE/'Code Cache/js',BASE/'Code Cache/wasm',Path('/Users/junyi/Library/Caches/Codex')]
PEAK=1394319870
RESERVE=10737418240
MAGIC=bytes.fromhex('305c72a71b6dfbfc')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def row(p):
 s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1
 return dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,sha256=sha(p))
def opens():
 r=subprocess.run(['/usr/sbin/lsof','-nP','-Fn'],capture_output=True,text=True)
 assert r.returncode==0 and not r.stderr,r.stderr
 return {s[1:] for s in r.stdout.splitlines() if s.startswith('n')}
def capacity():
 s=os.statvfs(ROOT);free=s.f_bavail*s.f_frsize
 retained=sum(p.stat().st_size for p in (ROOT/'research_runs/general-rsi-exposure-001').glob('slot-*/*.pt') if p.name in ('raw.pt','h200.pt'))
 required=RESERVE+max(0,PEAK-retained)
 return dict(free_bytes=free,retained_panel_bytes=retained,required_free_bytes=required,deficit_bytes=max(0,required-free))
def save(name,value):
 with open(OUT/name,'x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
with open(ROOT/'research_state/external-cache-maintenance.lock','a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'manifest.json').exists()
 before=capacity();assert before['deficit_bytes']>0
 protected=[ROOT/'research/tokenizer-evidence/general-rsi-panel-001/plan.json',ROOT/'research/tokenizer-evidence/general-rsi-panel-001/selection.json',ROOT/'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json',ROOT/'research_runs/byte-input-quality-v1-001/10/h200.pt',ROOT/'research_runs/byte-input-quality-v1-001/10/raw.pt']
 protected += sorted((ROOT/'research_runs/general-rsi-panel-001').glob('slot-0[4-9]/*.pt'))
 protected += [ROOT/'research/tokenizer-evidence/general-rsi-exposure-001/plan.json']
 protected += sorted((ROOT/'research_runs/general-rsi-exposure-001').glob('slot-*/*.pt'))
 protected_hashes={str(p):sha(p) for p in protected}
 open_paths=opens();rows=[]
 for base in BASES:
  for p in base.rglob('*'):
   if p.is_symlink() or not p.is_file() or not re.fullmatch('[0-9a-f]{16}_0',p.name) or str(p) in open_paths:continue
   with p.open('rb') as f:header=f.read(8)
   if header==MAGIC:rows.append(row(p))
 rows.sort(key=lambda x:x['allocated'],reverse=True)
 # Bounded new pass: enough allocation for current deficit plus 64MiB churn.
 selected=[];allocated=0
 for e in rows:
  selected.append(e);allocated+=e['allocated']
  if allocated>=before['deficit_bytes']+(64<<20):break
 save('manifest.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Newly regenerated unopened Chromium HTTP/compiled-code cache payloads. Exact magic, allowlist, inode/hash and consumer checks. Active research and all user/application state excluded.',initial_observation={'at':'2026-11-10T03:01:57Z','free_bytes':11444920320,'required_free_bytes':11477326780,'deficit_bytes':32406460},before=before,entries=selected,protected_sha256=protected_hashes,research_lock_policy='Worker PID15837 retains research lock; independent external-cache lock only. No research payload deletion.'))
 removed=[];skipped=[]
 with open(OUT/'deletion-journal.jsonl','x') as j:
  def log(v):j.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**v})+'\n');j.flush();os.fsync(j.fileno())
  open_paths=opens()
  for i,e in enumerate(selected):
   if i%100==0:open_paths=opens()
   p=Path(e['path'])
   if str(p) in open_paths or not p.exists() or row(p)!=e:skipped.append(str(p));continue
   log(dict(event='intent',entry=e));p.unlink();removed.append(e);log(dict(event='removed',path=str(p)))
  for base in {Path(e['path']).parent for e in removed}:
   fd=os.open(base,os.O_RDONLY);os.fsync(fd);os.close(fd)
  for p,h in protected_hashes.items():assert sha(Path(p))==h,p
  after=capacity()
  result=dict(status='recovered' if after['deficit_bytes']==0 else 'partial',before=before,after=after,removed_files=len(removed),removed_allocated_bytes=sum(e['allocated'] for e in removed),skipped_changed_or_open=skipped,protected_files_verified=len(protected_hashes),research_payloads_modified=False,note='Full forecast includes serialization overlap and 10GiB reserve (8GiB floor +2GiB headroom); concurrent writes affect observed free delta.')
  log(dict(event='complete',**result));save('completed.json',result);print(json.dumps(result))
