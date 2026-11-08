"""One original bounded scientific slot. No retry of an entered slot."""
import fcntl,hashlib,json,os,sys
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from .contract import OUTPUT_CAP,slots
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001/quality001'
WEIGHTS=ROOT/'research_runs/byte-input-quality-v1-001'
TORCH=None
OUT=None
WRITTEN=0
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()
def pins(records):
 for r in records:
  p=Path(r['path']);assert p.stat().st_size==r['size'] and sha(p)==r['sha256'],str(p)
def publish(name,value):
 global WRITTEN
 raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();WRITTEN+=len(raw)
 assert WRITTEN<=16*1024**2,'Per-slot metadata cap'
 with (OUT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

class Guard(CpuPrerequisiteGuard):
 def _sample(self):
  sample,reason=super()._sample();paths=[p for p in WEIGHTS.rglob('*') if p.is_file()]
  if any(p.is_symlink() for p in paths):reason='unsafe_output'
  sample['retained_weight_bytes']=sum(p.stat().st_size for p in paths)
  sample['metadata_allowance_bytes']=512*1024**2
  sample['charged_output_bytes']=sample['retained_weight_bytes']+sample['metadata_allowance_bytes']
  if sample['charged_output_bytes']>OUTPUT_CAP:reason='output_limit'
  if TORCH is not None:
   sample['mps_driver_bytes']=TORCH.mps.driver_allocated_memory()
   if sample['mps_driver_bytes']>=6*1024**3:reason='mps_limit'
  return sample,reason

def main(expected,index):
 global TORCH,OUT
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:raise SystemExit(75)
  assert type(index)is int and 0<=index<18 and sha(BASE/'plan.json')==expected
  plan=json.loads((BASE/'plan.json').read_text());admission=json.loads((BASE/'admission/review.json').read_text())
  assert admission['accepted'] and admission['plan_sha256']==expected
  assert json.loads((BASE/'admission/guard.json').read_text())['status']=='finished'
  spec=dict(slots()[index])
  from .controller import acceptance,selection
  accepted=[acceptance(i) for i in range(index)]
  if index>=9:spec['learning_rate']=selection(accepted[:9])[spec['variant']]
  retained=sum(p.stat().st_size for p in WEIGHTS.rglob('*') if p.is_file())
  stat=os.statvfs(ROOT);assert stat.f_bavail*stat.f_frsize>=20*1024**3-retained,'Complete remaining allocation plus reserve/headroom'
  OUT=BASE/'slots'/str(index).zfill(2);archives=WEIGHTS/str(index).zfill(2)
  OUT.mkdir();archives.mkdir();publish('attempt.json',dict(pid=os.getpid(),plan_sha256=expected,spec=spec,no_replay=True))
  try:
   with Guard(OUT/'guard.json',ROOT,timeout_seconds=7200,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,poll_seconds=.5):
    pins(plan['sources']);pins(plan['assets']);pins(plan['prerequisites']);assert sys.flags.optimize==0
    for key,value in plan['environment'].items():assert os.environ.get(key)==value,key
    import torch
    TORCH=torch
    from .driver import run
    run(plan,spec,archives,publish)
    from .review import run as saved_review
    publish('review.json',saved_review(plan,spec,OUT))
    pins(plan['sources']);pins(plan['assets']);pins(plan['prerequisites'])
   publish('STOP',dict(status='complete_original_scientific_slot',no_replay=True))
  except BaseException as exc:
   if not (OUT/'STOP').exists():publish('STOP',dict(status='closed_failed_original_scientific_slot',error=repr(exc),no_replay=True))
   raise

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]))
