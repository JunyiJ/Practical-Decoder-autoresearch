"""Exact one-shot retirement of closed obsolete numerical RAW fixtures."""
import fcntl,hashlib,json,os,shutil,stat,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(ROOT))
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
OUT=Path(__file__).resolve().parent
BASE=OUT.parent
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()
def put(name,value):
 with (OUT/name).open('x') as f:json.dump(value,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def exact(r):
 p=Path(r['path']);s=p.lstat()
 assert p==p.resolve() and p.is_relative_to(ROOT/'research_runs') and not any(x.is_symlink() for x in [p,*p.parents])
 assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size==r['size'] and sha(p)==r['sha256']
 return p,s
def consumers(paths):
 proc=subprocess.run(['ps','-axo','pid=,comm=,args='],capture_output=True,text=True,check=True)
 lines=[]
 for line in proc.stdout.splitlines():
  fields=line.strip().split(None,2)
  if len(fields)<3 or int(fields[0])==os.getpid():continue
  if 'python' in fields[1].lower() and ('research.' in fields[2] or 'LLM-pretraining-RSI' in fields[2]):lines.append(line)
 assert not lines,lines
 opened=subprocess.run(['/usr/sbin/lsof','-Fpn','--',*paths],capture_output=True,text=True)
 assert opened.returncode in (0,1) and not opened.stdout and not opened.stderr,(opened.returncode,opened.stdout,opened.stderr)
 return dict(active_research_python=lines,lsof_returncode=opened.returncode,open_file_consumers=[])

def run():
 plan=json.loads((OUT/'plan.json').read_text());items=plan['allowlist'];paths=[r['path'] for r in items]
 assert len(paths)==len(set(paths))==38
 assert {Path(p).parent.name for p in paths}=={'byte-input-state-math001','byte-input-state-math002','byte-input-state-math003','byte-input-training-math001'}
 for rel in plan['stop_paths']:
  value=json.loads((BASE/rel).read_text());assert value['no_replay'] and value['status'].startswith(('complete_','closed_failed_'))
 for rel in plan['acceptance_paths']:assert json.loads((BASE/rel).read_text())['accepted']
 current=json.loads((BASE/'pilot-math002/plan.json').read_text())
 assert not set(paths)&{r['path'] for k in ['sources','assets','prerequisites'] for r in current[k]}
 roots=[ROOT/'research',ROOT/'src',Path('/Users/junyi/.codex/automations')]
 search=subprocess.run(['rg','-l','byte-input-state-math00[123]|byte-input-training-math001',*[str(p) for p in roots],'--glob','*.py','--glob','*.toml'],capture_output=True,text=True)
 assert search.returncode==0 and not search.stderr
 found=set(search.stdout.splitlines());found.discard(str(Path(__file__).resolve()))
 wanted={str(ROOT/f'research/byte_input_state_proof_v{i}/entry.py') for i in (1,2,3)}|{str(ROOT/'research/byte_input_training_proof_v1/entry.py')}
 assert found==wanted,found
 checks=consumers(paths)
 # Inventory by size, including incomplete files, rather than by extension.
 large=[]
 for parent,dirs,files in os.walk(ROOT/'research_runs',followlinks=False):
  dirs[:]=[d for d in dirs if not (Path(parent)/d).is_symlink()]
  for name in files:
   p=Path(parent)/name;s=p.lstat()
   if s.st_size>=100*1024**2:large.append(dict(path=str(p),size=s.st_size,eligible=str(p) in paths,incomplete=name.endswith('.tmp')))
 put('large-file-inventory.json',sorted(large,key=lambda x:-x['size']))
 preserved={}
 for p in sorted(BASE.rglob('*')):
  if p.is_file() and not p.is_relative_to(OUT):preserved[str(p)]=dict(size=p.stat().st_size,sha256=sha(p))
 # Preserve every still-needed binary explicitly, including latest pilot parents.
 for rel in ['training-acceptance001/local-payload-manifest.json','inference-acceptance001/local-payload-manifest.json','pilot-acceptance002/local-payload-manifest.json']:
  for r in json.loads((BASE/rel).read_text())['payloads']:
   if r['path'] not in paths:
    p,s=exact(r);preserved[str(p)]=dict(size=s.st_size,sha256=r['sha256'])
 originals={}
 for r in items:
  p,s=exact(r);originals[str(p)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]
 put('retained-evidence-manifest.json',preserved)
 free=shutil.disk_usage(ROOT).free
 put('audit.json',dict(accepted=True,free_before=free,deficit_before=max(0,plan['required_free_bytes']-free),
  expected_logical_recovery=sum(r['size'] for r in items),consumers=checks,source_consumers=sorted(found),
  dependency_closure=plan['closure'],policy_exception=plan['authority'],retained_files=len(preserved)))
 consumers(paths)
 with (OUT/'deletion-journal.jsonl').open('x') as journal:
  def record(x):journal.write(json.dumps(x,sort_keys=True)+'\n');journal.flush();os.fsync(journal.fileno())
  for r in items:
   p,s=exact(r);assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==originals[str(p)]
   record(dict(event='authorized_exact_delete',**r));p.unlink()
   fd=os.open(p.parent,os.O_RDONLY)
   try:os.fsync(fd)
   finally:os.close(fd)
   assert not p.exists();record(dict(event='deleted_verified',**r))
 for path,r in preserved.items():
  p=Path(path);assert p.stat().st_size==r['size'] and sha(p)==r['sha256'],path
 after=shutil.disk_usage(ROOT).free
 put('result.json',dict(completed=True,deleted_files=len(items),logical_bytes=sum(r['size'] for r in items),
  free_before=free,free_after=after,observed_free_change=after-free,required_free_bytes=plan['required_free_bytes'],
  remaining_deficit=max(0,plan['required_free_bytes']-after),capacity_satisfied=after>=plan['required_free_bytes'],
  retained_evidence_verified=len(preserved),next_action=plan['next_action']))

if __name__=='__main__':
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (OUT/'attempt.json').exists();put('attempt.json',dict(pid=os.getpid(),no_replay=True,plan_sha256=sha(OUT/'plan.json')))
  try:
   with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=300,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3):run()
   put('STOP',dict(status='complete_original_retirement',no_replay=True))
  except BaseException as exc:
   put('STOP',dict(status='closed_failed_original_retirement',error=repr(exc),no_replay=True));raise
