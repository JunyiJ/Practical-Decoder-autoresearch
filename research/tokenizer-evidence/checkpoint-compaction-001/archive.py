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
E=Path(__file__).resolve().parent;COLD=ROOT/'research_archives/checkpoint-compaction-001'
RESERVE=8*1024**3;TARGET=30631412484+1024**3;ZSTD='/opt/homebrew/bin/zstd'

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
   if time.monotonic()-started>600 or rss+own>=4*1024**3 or shutil.disk_usage(ROOT).free<RESERVE or (archive.exists() and archive.stat().st_size>12*1024**3):
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

def plan():
 old=ROOT/'research/tokenizer-evidence/wide-hybrid-preparation-001/checkpoint-time-handoff-source-applicability-v2-review.json'
 prior=json.loads(old.read_text());weights=prior['retention']['checkpoints']
 groups={}
 for group,prefix in [('numerical','research_runs/tokenizer-wide-hybrid-numerical-v2-001/'),('campaign002-completed','research_runs/tokenizer-wide-hybrid-002/'),('campaign003-completed','research_runs/tokenizer-wide-hybrid-003/')]:
  groupfiles={n:dict(size=v['size'],sha256=v['sha256']) for n,v in sorted(weights.items()) if n.startswith(prefix) and n.endswith('.pt')}
  assert len(groupfiles)=={'numerical':9,'campaign002-completed':10,'campaign003-completed':16}[group]
  for n,v in groupfiles.items():check(ROOT/n,v)
  groups[group]=groupfiles
 selected={n for values in groups.values() for n in values}
 protected={n:dict(size=v['size'],sha256=v['sha256']) for n,v in weights.items() if n not in selected}
 for n,v in protected.items():check(ROOT/n,v)
 metadata={}
 for root in (ROOT/'research_runs').glob('tokenizer-*'):
  if not root.is_dir():continue
  for p in root.rglob('*'):
   if p.is_file() and not p.name.endswith(('.pt','.pt.tmp')) and p.name!='.lock' and '__pycache__' not in p.parts:
    assert not p.is_symlink();metadata[str(p.relative_to(ROOT))]=dict(size=p.stat().st_size,sha256=file_digest(p))
 COLD.mkdir(parents=True,exist_ok=False)
 publish(E/'plan.json',dict(status='planned',authorization='Human requested safe archival/removal and proactive storage recovery; preserve selected models, fresh confirmation pairs and failure evidence.',
  source_inventory_sha256=file_digest(old),script_sha256=file_digest(Path(__file__)),groups=groups,protected=protected,metadata=metadata,
  reserve_bytes=RESERVE,target_free_bytes=TARGET,free_bytes=shutil.disk_usage(ROOT).free,
  archive_format='tar PAX + zstd level3 long-window30 single worker; no lossy conversion; member path/size/SHA256 verified before loose-copy removal',
  rationale='Nine completed numerical resume prerequisites and26 successfully accepted non-fresh, non-selected screen weights from closed incomplete campaigns are cold-archived, not discarded. Failed partials, selected screens and all fresh pairs stay loose and unchanged. No old controller is reopened.',
  restoration='Decompress archive with zstd -d --long=30 --stdout; restore only manifest-listed members into a staging directory, verify size/SHA256, then exclusively publish to the original absent paths. Never overwrite a path or rerun a campaign.'))
 print(json.dumps(dict(status='planned',files=sum(map(len,groups.values())),logical_bytes=sum(v['size'] for g in groups.values() for v in g.values()),protected=len(protected),metadata=len(metadata))),flush=True)

def apply():
 p=json.loads((E/'plan.json').read_text());assert file_digest(Path(__file__))==p['script_sha256']
 assert not (E/'journal.jsonl').exists()
 # Recover scratch space only: both exact duplicates are independently pinned.
 probe=json.loads((E/'sparse-probe.json').read_text());scratch=ROOT/probe['copy']
 check(scratch,probe);check(ROOT/probe['source'],probe);scratch.unlink()
 journal(dict(event='removed_verified_probe_duplicate',path=probe['copy']))
 receipts=[]
 for group,files in p['groups'].items():
  if shutil.disk_usage(ROOT).free>=TARGET:break
  archive=COLD/(group+'.tar.zst')
  before=shutil.disk_usage(ROOT).free;create_archive(archive,files);members=verify_archive(archive,files)
  receipt=dict(group=group,archive=str(archive.relative_to(ROOT)),archive_size=archive.stat().st_size,archive_sha256=file_digest(archive),members=members,verified_before_removal=True)
  publish(E/(group+'-archive.json'),receipt)
  journal(dict(event='archive_verified',**{k:v for k,v in receipt.items() if k!='members'}))
  # All originals remain intact until the complete archive verification passed.
  for n,v in files.items():check(ROOT/n,v)
  for n,v in files.items():
   check(ROOT/n,v);(ROOT/n).unlink();syncdir((ROOT/n).parent);journal(dict(event='loose_copy_removed',path=n,original_sha256=v['sha256'],archive=receipt['archive']))
  assert file_digest(archive)==receipt['archive_sha256']
  receipt.update(logical_bytes=sum(v['size'] for v in files.values()),net_logical_reclaimed_bytes=sum(v['size'] for v in files.values())-receipt['archive_size'],free_before=before,free_after=shutil.disk_usage(ROOT).free)
  receipts.append(receipt)
  print(json.dumps({k:v for k,v in receipt.items() if k!='members'}),flush=True)
 for n,v in p['protected'].items():check(ROOT/n,v)
 for n,v in p['metadata'].items():
  f=ROOT/n;assert f.stat().st_size==v['size'] and file_digest(f)==v['sha256']
 archived={n for receipt in receipts for n in receipt['members']}
 for group,files in p['groups'].items():
  for n,v in files.items():
   if n in archived:assert not (ROOT/n).exists()
   else:check(ROOT/n,v)
 publish(E/'completed.json',dict(status='completed',archives=receipts,archived_files=len(archived),protected_loose_files_verified=len(p['protected']),noncheckpoint_files_verified=len(p['metadata']),
  net_logical_reclaimed_bytes=sum(x['net_logical_reclaimed_bytes'] for x in receipts),free_bytes=shutil.disk_usage(ROOT).free,required_campaign_free_bytes=30631412484,
  no_checkpoint_bytes_lost=True,selected_and_fresh_unchanged=True,failure_evidence_unchanged=True,training=False,holdout_consumed=False))
 print((E/'completed.json').read_text()[:500],flush=True)

if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('plan','apply')
 with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/(mode+'-guard.json'),ROOT,timeout_seconds=1800 if mode=='apply' else 600):
  plan() if mode=='plan' else apply()
