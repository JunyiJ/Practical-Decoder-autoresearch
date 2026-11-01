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
B=Path(__file__).resolve().parent;E=B/'synthetic-archive001';COLD=ROOT/'research_archives/mla-score-synthetic-001'
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
def bound(r):
 p=Path(r['path']);assert file_digest(p)==r['sha256'];return json.loads(p.read_text())
def inventory(manifest):
 return {kind:[row for reference in refs for row in bound(reference)] for kind,refs in manifest['inventory_shards'].items()}
def immutable(row):
 p=Path(row['path']);s=p.stat();assert not p.is_symlink()
 assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['size'],row['mtime_ns'])

def plan():
 E.mkdir();COLD.mkdir()
 review_path=B/'serialization-capacity002.json';review=json.loads(review_path.read_text())
 candidates=review['authorized_next_capacity_recovery']['files'];assert len(candidates)==12
 eligibility=[B/'checkpoint-resume001.json',ROOT/'research/tokenizer-evidence/ffn-structure-preparation-001/checkpoint-resume001.json']
 for path in eligibility:assert json.loads(path.read_text())['status']=='serialized_mps_repeat_resume_h200_passed'
 receipt_path=B/'ema-pilot-retention/completed.json';receipt=json.loads(receipt_path.read_text())
 assert receipt['status']=='completed' and bound(receipt['guard'])['status']=='finished'
 m=bound(receipt['manifest']);rows=inventory(m);groups={}
 remaining={r['path']:r for r in rows['protected']}
 for row in candidates:
  key='score' if '/tokenizer-mla-score-checkpoint-' in row['path'] else 'ffn'
  source=remaining.pop(row['path']);assert source['size']==row['size'] and source['sha256']==row['sha256'];immutable(source)
  n=str(Path(row['path']).relative_to(ROOT));check(ROOT/n,row);groups.setdefault(key,{})[n]=dict(size=row['size'],sha256=row['sha256'])
 assert set(groups)=={'score','ffn'} and all(len(g)==6 for g in groups.values())
 for row in [*remaining.values(),*rows['preserved_metadata']]:immutable(row)
 publish(E/'plan.json',dict(status='exact_lossless_archival_planned',script=ref(Path(__file__)),policy=ref(ROOT/'research/TOKENIZER_CHECKPOINT_RETENTION.md'),eligibility=[ref(p) for p in eligibility],
  inventory_manifest=receipt['manifest'],inventory_verified_receipt=ref(receipt_path),groups=groups,
  rationale='Successful synthetic resume prerequisites only. Preserve exact bytes by verified cold archive; no selected/fresh/failed files. Current score fixtures become archive-resolvable, not discarded; restore exact bytes if downstream APIs require loose paths.',
  archive='PAX tar + zstd level3 long30, one compression thread. Verify every decompressed member before any loose-copy removal.',
  reserve_bytes=RESERVE,max_archive_bytes=2300*1024**2,subprocess_and_parent_rss_limit=4*1024**3,per_subprocess_timeout_seconds=600,
  restoration='Restore only absent allowlisted member paths through staging with size/hash verification; never overwrite or reopen closed fixtures.',free_bytes=shutil.disk_usage(ROOT).free))
 print(json.dumps(dict(status='planned',files=12,bytes=sum(r['size'] for r in candidates))),flush=True)

def apply():
 p=json.loads((E/'plan.json').read_text());bound(p['script']);bound(p['policy']);bound(p['inventory_verified_receipt'])
 for r in p['eligibility']:bound(r)
 rows=inventory(bound(p['inventory_manifest']));moved={str(ROOT/n) for g in p['groups'].values() for n in g}
 survivors=[r for r in rows['protected'] if r['path'] not in moved]+rows['preserved_metadata']
 for r in [*rows['protected'],*rows['preserved_metadata']]:immutable(r)
 assert not (E/'journal.jsonl').exists();receipts=[]
 for group,files in p['groups'].items():
  archive=COLD/(group+'.tar.zst');before=shutil.disk_usage(ROOT).free
  create_archive(archive,files);members=verify_archive(archive,files)
  receipt=dict(group=group,archive=ref(archive),archive_size=archive.stat().st_size,members=members,verified_before_removal=True)
  publish(E/(group+'-verified.json'),receipt)
  for n,v in files.items():check(ROOT/n,v)
  for n,v in files.items():
   check(ROOT/n,v);journal(dict(event='intent_remove_verified_loose_copy',path=n,sha256=v['sha256'],archive=receipt['archive']))
   (ROOT/n).unlink();syncdir((ROOT/n).parent);journal(dict(event='removed_verified_loose_copy',path=n))
  assert file_digest(archive)==receipt['archive']['sha256']
  receipt.update(logical_bytes=sum(v['size'] for v in files.values()),net_logical_reclaimed_bytes=sum(v['size'] for v in files.values())-receipt['archive_size'],free_before=before,free_after=shutil.disk_usage(ROOT).free)
  receipts.append(receipt);print(json.dumps({k:v for k,v in receipt.items() if k!='members'}),flush=True)
 for r in survivors:immutable(r);check(Path(r['path']),r)
 for n in moved:assert not Path(n).exists()
 publish(E/'completed.json',dict(status='completed',plan=ref(E/'plan.json'),archives=receipts,archived_files=12,no_checkpoint_bytes_lost=True,
  selected_fresh_failure_weights_unchanged=True,retained_files_verified=len(survivors),net_logical_reclaimed_bytes=sum(r['net_logical_reclaimed_bytes'] for r in receipts),free_bytes=shutil.disk_usage(ROOT).free,
  next='Finish real worker/parent and inference acceptance; future audits resolve exact archived members or restore with verified hashes. Recompute capacity before finite freezes.',training=False,holdout_consumed=False))
 print(json.dumps({k:v for k,v in json.loads((E/'completed.json').read_text()).items() if k!='archives'}),flush=True)

if __name__=='__main__':
 from research.ffn_quality_v1.support import idle
 mode=sys.argv[1];assert mode in ('plan','apply')
 with session_lock(ROOT/'research_state/tokenizer-worker-lock'):
  idle()
  if mode=='plan':
   # Guard output parent must exist before entry; keep plan() exclusive creation.
   with CpuPrerequisiteGuard(B/'synthetic-archive001-plan-guard.json',ROOT,timeout_seconds=600):plan()
  else:
   with CpuPrerequisiteGuard(E/'apply-guard.json',ROOT,timeout_seconds=1800):apply()
