"""User-authorized lossless cold archive. Never trains, loads tensors or loses bytes.

One archive at a time; verify every streamed member before deleting any loose
copy. Selected/fresh/failure weights are excluded. All receipts remain intact.
"""
import argparse,hashlib,json,os,resource,shutil,subprocess,sys,tarfile,threading,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
from src.research.protocol import file_digest
B=Path(__file__).resolve().parent;E=B/'worker-archive001';COLD=ROOT/'research_archives/mla-score-worker-001'
RESERVE=8*1024**3;TARGET=13690208256;ZSTD='/opt/homebrew/bin/zstd'

def publish(path,value):
 with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def journal(value):
 with (E/'journal.jsonl').open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
def syncdir(path):
 fd=os.open(path,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def check(path,v):
 assert not path.is_symlink() and path.is_file() and path.stat().st_nlink==1
 assert path.stat().st_size==v['size'] and file_digest(path)==v['sha256']

def monitored(proc,done,errors,archive):
 started=time.monotonic()
 try:
  while not done.wait(.5):
   if proc.poll() is not None:return
   p=subprocess.run(['/bin/ps','-o','rss=','-p',str(proc.pid)],capture_output=True,text=True,timeout=5)
   rss=int(p.stdout.strip() or 0)*1024
   own=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
   if time.monotonic()-started>600 or rss+own>=4*1024**3 or shutil.disk_usage(ROOT).free<RESERVE or (archive.exists() and archive.stat().st_size>2300*1024**2):
    raise RuntimeError('Archive wall/RSS/disk/output bound')
 except BaseException as exc:
  errors.append(repr(exc));proc.terminate()

def run_stream(argv,archive,operation):
 done=threading.Event();errors=[]
 # Only the known compressor/decompressor subprocess is supervised/stopped.
 with (E/'subprocess.log').open('ab') as err:
  proc=subprocess.Popen(argv,stdin=subprocess.PIPE if operation=='create' else subprocess.DEVNULL,
       stdout=subprocess.DEVNULL if operation=='create' else subprocess.PIPE,stderr=err)
  thread=threading.Thread(target=monitored,args=(proc,done,errors,archive),daemon=True);thread.start()
  return proc,done,thread,errors

def finish(state):
 proc,done,thread,errors=state
 try:code=proc.wait(timeout=30)
 finally:
  if proc.poll() is None:proc.kill();proc.wait()
  done.set();thread.join()
 assert code==0 and not errors,(code,errors)

def create_archive(archive,files):
 assert not archive.exists() and shutil.disk_usage(ROOT).free>=RESERVE+max(v['size'] for v in files.values())+512*1024**2
 state=run_stream([ZSTD,'-3','--long=30','-T1','--no-progress','-o',str(archive)],archive,'create')
 proc=state[0]
 try:
  with tarfile.open(fileobj=proc.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
   for n,v in files.items():
    check(ROOT/n,v)
    tar.add(ROOT/n,arcname=n,recursive=False)
  proc.stdin.close();finish(state)
 except BaseException:
  proc.terminate();proc.wait(timeout=10);state[1].set();state[2].join();raise
 with archive.open('rb') as f:os.fsync(f.fileno())
 syncdir(archive.parent)

def verify_archive(archive,files):
 state=run_stream([ZSTD,'-d','--long=30','--stdout',str(archive)],archive,'verify');proc=state[0];seen={}
 try:
  with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
   for member in tar:
    assert member.isfile() and member.name in files and member.name not in seen
    expected=files[member.name];assert member.size==expected['size']
    h=hashlib.sha256();size=0
    stream=tar.extractfile(member)
    while data:=stream.read(4*1024**2):h.update(data);size+=len(data)
    assert size==expected['size'] and h.hexdigest()==expected['sha256']
    seen[member.name]=dict(size=size,sha256=h.hexdigest())
  while proc.stdout.read(4*1024**2):pass
  proc.stdout.close();finish(state)
 except BaseException:
  proc.terminate();proc.wait(timeout=10);state[1].set();state[2].join();raise
 assert set(seen)==set(files)
 return seen

def ref(p):return dict(path=str(p),sha256=file_digest(p))
def verify(r):
 p=Path(r['path']);assert file_digest(p)==r['sha256'];return p
def bound(r):return json.loads(verify(r).read_text())
def record(path):
 assert not path.is_symlink() and path.is_file()
 before=path.stat();h=file_digest(path);after=path.stat()
 assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 return dict(path=str(path),size=after.st_size,sha256=h,device=after.st_dev,inode=after.st_ino,mtime_ns=after.st_mtime_ns)
def all_paths():
 paths=[]
 for parent in ('research_runs','research_archives'):
  for p in sorted((ROOT/parent).rglob('*')):
   if '__pycache__' in p.parts:continue
   assert not p.is_symlink()
   if p.is_file():paths.append(p)
 return paths

def plan():
 E.mkdir();current=json.loads((B/'trainer-worker001.json').read_text());timing=json.loads((B/'trained-inference001.json').read_text())
 assert current['status']=='real_bounded_worker_parent_acceptance_passed' and timing['status']=='generated_trained_score_pair_inference_passed'
 old_path=ROOT/'research/tokenizer-evidence/ffn-structure-preparation-001/trainer-worker003.json';old=json.loads(old_path.read_text())
 assert old['status']=='actual_serial_trainer_and_parent_acceptance_passed'
 rows=list(current['arms'])+[r for r in old['arms'] if r['arm']=='ungated_silu' and not r['imported_control']]
 assert len(rows)==3
 files={}
 for row in rows:
  for key in ('parent','h200'):
   binding=row['binding'][key];path=verify(binding);r=record(path)
   files[str(path.relative_to(ROOT))]=dict(size=r['size'],sha256=r['sha256'])
 assert len(files)==6 and sum(v['size'] for v in files.values())==1677471074
 metadata=[];protected=[]
 for path in all_paths():
  if str(path.relative_to(ROOT)) in files:continue
  row=record(path);(protected if path.suffix=='.pt' or path.is_relative_to(ROOT/'research_archives') else metadata).append(row)
 shards={}
 for group,rows in [('protected',protected),('metadata',metadata)]:
  shards[group]=[]
  for start in range(0,len(rows),500):
   path=E/f'{group}-{start//500:03d}.json';publish(path,rows[start:start+500]);shards[group].append(ref(path))
 publish(E/'plan.json',dict(status='planned_exact_successful_worker_lossless_archive',source=ref(Path(__file__)),policy=ref(ROOT/'research/TOKENIZER_CHECKPOINT_RETENTION.md'),
  eligibility=[ref(B/'trainer-worker001.json'),ref(B/'trained-inference001.json'),ref(old_path)],files=files,inventory_shards=shards,
  reason='Two newly accepted score worker raw/H200 pairs plus the successful ungated FFN worker003 pair. Failed worker002 control is excluded. No selected or fresh scientific weights. Preserve exact bytes by full decompression verification before loose-copy retirement.',
  restoration='Use archive member manifest or restore exact absent paths through verified staging with sufficient reserve. Never overwrite, discard bytes or rerun closed fixtures.',
  free_bytes=shutil.disk_usage(ROOT).free,limits=dict(reserve=RESERVE,archive_bytes=2300*1024**2,rss=4*1024**3,subprocess_seconds=600)))
 print(json.dumps(dict(status='planned',files=6,logical_bytes=sum(v['size'] for v in files.values()),protected=len(protected),metadata=len(metadata))),flush=True)

def apply():
 p=json.loads((E/'plan.json').read_text());verify(p['source']);verify(p['policy'])
 for r in p['eligibility']:verify(r)
 survivors=[r for refs in p['inventory_shards'].values() for shard in refs for r in bound(shard)]
 for r in survivors:
  path=Path(r['path']);s=path.stat();assert not path.is_symlink()
  assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(r['device'],r['inode'],r['size'],r['mtime_ns'])
 assert set(map(str,all_paths()))=={r['path'] for r in survivors}|{str(ROOT/n) for n in p['files']}
 COLD.mkdir();archive=COLD/'successful-workers.tar.zst';create_archive(archive,p['files']);members=verify_archive(archive,p['files'])
 receipt=dict(archive=ref(archive),archive_size=archive.stat().st_size,members=members,verified_before_removal=True)
 publish(E/'verified.json',receipt)
 for n,v in p['files'].items():check(ROOT/n,v)
 for n,v in p['files'].items():
  check(ROOT/n,v);journal(dict(event='intent_remove_verified_loose_copy',path=n,sha256=v['sha256'],archive=receipt['archive']))
  (ROOT/n).unlink();syncdir((ROOT/n).parent);journal(dict(event='removed_verified_loose_copy',path=n))
 assert file_digest(archive)==receipt['archive']['sha256']
 for r in survivors:assert record(Path(r['path']))==r
 assert set(map(str,all_paths()))=={r['path'] for r in survivors}|{str(archive)}
 result=dict(status='completed',plan=ref(E/'plan.json'),archive=receipt,archived_files=6,
  net_logical_reclaimed_bytes=sum(v['size'] for v in p['files'].values())-receipt['archive_size'],free_bytes=shutil.disk_usage(ROOT).free,
  retained_files_verified=len(survivors),no_checkpoint_bytes_lost=True,selected_fresh_failed_unchanged=True,training=False,holdout_consumed=False)
 publish(E/'completed.json',result);print(json.dumps({k:v for k,v in result.items() if k!='archive'}),flush=True)

if __name__=='__main__':
 from research.ffn_quality_v1.support import idle
 mode=sys.argv[1];assert mode in ('plan','apply')
 with session_lock(ROOT/'research_state/tokenizer-worker-lock'):
  idle()
  with CpuPrerequisiteGuard(B/'worker-archive001-plan-guard.json' if mode=='plan' else E/'apply-guard.json',ROOT,timeout_seconds=600 if mode=='plan' else 1800):
   plan() if mode=='plan' else apply()
