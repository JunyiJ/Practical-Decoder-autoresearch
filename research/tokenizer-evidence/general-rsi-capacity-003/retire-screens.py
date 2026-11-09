"""Execute the previously audited obsolete LR-screen retirement, once."""
import datetime,fcntl,hashlib,json,os,re,shutil,stat,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent
AUDIT=OUT/'screen-retirement-audit.json'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(n,v):
 with (OUT/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def capacity():
 p=json.loads((ROOT/'research/tokenizer-evidence/general-rsi-exposure-001/plan.json').read_text())
 r=sum(x.stat().st_size for x in (ROOT/'research_runs/general-rsi-exposure-001').glob('slot-*/*.pt') if x.name in ('raw.pt','h200.pt'))
 req=p['reserve_bytes']+max(0,p['incremental_peak_bytes']-r);free=shutil.disk_usage(ROOT).free
 return dict(free_bytes=free,required_free_bytes=req,deficit_bytes=max(0,req-free))
def opened():
 p=subprocess.run(['/usr/sbin/lsof','-nP','-Fn'],capture_output=True,text=True);assert p.returncode==0 and not p.stderr,p.stderr
 return {x[1:] for x in p.stdout.splitlines() if x.startswith('n')}
def check(e):
 p=Path(e['path']);s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==os.getuid()
 assert s.st_size==e['size'] and s.st_ino==e['inode'] and s.st_dev==e['device'] and s.st_mtime_ns==e['mtime_ns'] and sha(p)==e['sha256']
with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert not (OUT/'screen-retirement-completed.json').exists()
 ps=subprocess.check_output(['ps','-axo','pid,command'],text=True)
 workers=[x for x in ps.splitlines() if re.search(r'python.* -m research\..*(worker|pilot|qualification)',x,re.I)]
 assert not workers,workers
 before=capacity();assert before['deficit_bytes']>0
 a=json.loads(AUDIT.read_text());protected=dict(a['preserve_evidence_sha256'])
 for study,slots in [('general-rsi-panel-001',range(4,10)),('general-rsi-exposure-001',range(2))]:
  for i in slots:
   d=ROOT/f'research/tokenizer-evidence/{study}/slot-{i:02d}'
   r=json.loads((d/'result.json').read_text());assert r['status']=='completed'
   for e in r['endpoints']:protected[e['weights']['path']]=e['weights']['sha256']
 for path in ['research_runs/byte-input-quality-v1-001/10/raw.pt','research_runs/byte-input-quality-v1-001/10/h200.pt','research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json','research/tokenizer-evidence/general-rsi-exposure-001/plan.json']:
  protected[str(ROOT/path)]=sha(ROOT/path)
 for p,h in protected.items():assert sha(p)==h,p
 entries=a['entries'];assert len(entries)==8
 for e in entries:check(e);assert e['path'] not in protected
 op=opened();assert not any(e['path'] in op for e in entries)
 save('screen-retirement-manifest.json',dict(audit_sha256=sha(AUDIT),before=before,entries=entries,protected_sha256=protected,process_check={"matching_research_workers":workers,"total_process_rows":len(ps.splitlines())}))
 removed=[]
 with (OUT/'screen-retirement-journal.jsonl').open('x') as j:
  def log(v):j.write(json.dumps({'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),**v})+'\n');j.flush();os.fsync(j.fileno())
  for e in entries:
   check(e);assert e['path'] not in opened();log(dict(event='intent',entry=e));Path(e['path']).unlink();removed.append(e['path']);log(dict(event='removed',path=e['path']))
  for d in {Path(p).parent for p in removed}:
   fd=os.open(d,os.O_RDONLY);os.fsync(fd);os.close(fd)
  for p,h in protected.items():assert sha(p)==h,p
  after=capacity();result=dict(status='recovered' if after['deficit_bytes']==0 else 'partial',before=before,after=after,removed_files=len(removed),removed_allocated_bytes=sum(e['allocated_bytes'] for e in entries),protected_files_verified=len(protected),selected_and_fresh_weights_preserved=True)
  log(dict(event='complete',**result));save('screen-retirement-completed.json',result);print(json.dumps(result))
