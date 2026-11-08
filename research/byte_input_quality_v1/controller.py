"""Sequential18-slot controller; barriers before selection and confirmation."""
import fcntl,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from .contract import DESIGN,slots,select
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001/quality001'
WEIGHTS=ROOT/'research_runs/byte-input-quality-v1-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()
def read(p):return json.loads(p.read_text())
def put(p,value):
 with p.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
def acceptance(index):
 out=BASE/'slots'/str(index).zfill(2);record=read(out/'acceptance.json')
 assert record['accepted'] and record['slot']==index
 for name,digest in record['evidence'].items():assert sha(out/name)==digest,name
 assert read(out/'guard.json')['status']=='finished' and read(out/'STOP')['status']=='complete_original_scientific_slot'
 row=read(out/'review.json');assert row['accepted'] and row['slot']==index
 for artifact in [row['parent'],row['h200']]:
  p=Path(artifact['path']);assert p.is_file() and not p.is_symlink() and p.stat().st_size==artifact['size']
 return row
def accept(index):
 out=BASE/'slots'/str(index).zfill(2)
 assert read(out/'guard.json')['status']=='finished' and read(out/'STOP')['status']=='complete_original_scientific_slot'
 row=read(out/'review.json');assert row['accepted'] and row['slot']==index
 for r in [row['parent'],row['h200']]:assert sha(Path(r['path']))==r['sha256']
 evidence={n:sha(out/n) for n in ['attempt.json','result.json','artifacts.json','validation.json','inference.json','review.json','guard.json','STOP']}
 put(out/'acceptance.json',dict(accepted=True,slot=index,evidence=evidence,no_adoption=True))
 return acceptance(index)
def selection(rows):
 chosen=select(rows);path=BASE/'selection.json'
 expected=dict(learning_rates=chosen,screens=[dict(slot=r['slot'],variant=r['variant'],learning_rate=r['learning_rate'],ce=r['ce'],review_sha256=sha(BASE/'slots'/str(r['slot']).zfill(2)/'review.json')) for r in rows])
 assert read(path)==expected,'Frozen complete screen selection differs'
 return chosen
def choose(rows):
 chosen=select(rows)
 # Common paired sampler trajectory across all rates and arms within screening.
 assert all(r['sampler_hashes']==rows[0]['sampler_hashes'] for r in rows)
 put(BASE/'selection.json',dict(learning_rates=chosen,screens=[dict(slot=r['slot'],variant=r['variant'],learning_rate=r['learning_rate'],ce=r['ce'],review_sha256=sha(BASE/'slots'/str(r['slot']).zfill(2)/'review.json')) for r in rows]))
 selection(rows)
def final(plan):
 rows=[acceptance(i) for i in range(18)];selection(rows[:9]);pairs=[]
 for j in range(3):
  group=rows[9+3*j:12+3*j];assert {r['variant'] for r in group}==set(DESIGN['arms'])
  assert all(r['bundle']=='fresh'+str(j) and r['sampler_hashes']==group[0]['sampler_hashes'] for r in group)
  pairs.append({r['variant']:read(BASE/'slots'/str(r['slot']).zfill(2)/'validation.json')['report'] for r in group})
 for row in rows:
  for r in [row['parent'],row['h200']]:assert sha(Path(r['path']))==r['sha256']
 from .statistics import calculate
 result=calculate(pairs,plan['bootstrap_seed'])
 result.update(accepted_complete_panel=True,trajectories=18,updates=21600,selection=read(BASE/'selection.json'),
  all_slot_acceptances=[dict(slot=i,sha256=sha(BASE/'slots'/str(i).zfill(2)/'acceptance.json')) for i in range(18)])
 put(BASE/'final.json',result)

def main(expected):
 with (BASE/'.controller.lock').open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert sha(BASE/'plan.json')==expected and not (BASE/'controller-attempt.json').exists()
  plan=read(BASE/'plan.json');assert plan['design']==DESIGN and read(BASE/'admission/review.json')['accepted']
  put(BASE/'controller-attempt.json',dict(pid=os.getpid(),plan_sha256=expected,no_replay=True,started_unix=time.time()))
  started=time.monotonic()
  try:
   rows=[]
   for index in range(18):
    assert time.monotonic()-started+7200<=130800,'Complete campaign wall budget exhausted before next slot'
    if index==9:choose(rows)
    while True:
     code=subprocess.call([sys.executable,'-m','research.byte_input_quality_v1.entry',expected,str(index)],cwd=ROOT)
     if code!=75:break
     assert time.monotonic()-started+7200<=130800
     time.sleep(20)  # Busy shared lock before entry only; never retry an attempt.
    assert code==0,('Scientific slot failed; preserve and diagnose, no automatic retry',index,code)
    rows.append(accept(index))
    put(BASE/('completed-'+str(index).zfill(2)+'.json'),dict(slot=index,review=rows[-1],elapsed_seconds=time.monotonic()-started))
   from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
   with CpuPrerequisiteGuard(BASE/'statistics-guard.json',ROOT,timeout_seconds=300,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3):final(plan)
   put(BASE/'STOP',dict(status='complete_original_quality_panel_pending_adoption_review',no_replay=True))
  except BaseException as exc:
   if not (BASE/'STOP').exists():put(BASE/'STOP',dict(status='closed_failed_original_quality_controller',error=repr(exc),no_replay=True))
   raise

if __name__=='__main__':main(sys.argv[1])
