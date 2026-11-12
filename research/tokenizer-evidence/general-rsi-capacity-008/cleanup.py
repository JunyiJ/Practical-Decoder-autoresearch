"""Retire exactly accepted, obsolete engineering payloads after dependency closure."""
import fcntl,json,hashlib,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
 return h.hexdigest()
def put(p,v):
 with Path(p).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def journal(v):
 with (OUT/'deletion-journal.jsonl').open('a') as f:f.write(json.dumps(v)+'\n');f.flush();os.fsync(f.fileno())

def main():
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  process=subprocess.check_output(['ps','-axo','pid,ppid,etime,command'],text=True)
  workers=[r for r in process.splitlines() if 'python' in r.lower() and any(s in r for s in ('general_rsi','tokenizer')) and 'cleanup.py' not in r]
  assert not workers,workers
  targets=[];protected={};evidence_roots=[]
  def add(receipt,evidence,reason):
   p=Path(receipt['path']);assert p.resolve()==p and not p.is_symlink() and p.stat().st_nlink==1
   st=p.stat();assert sha(p)==receipt['sha256']
   if 'size' in receipt:assert st.st_size==receipt['size']
   targets.append(dict(path=str(p),size=st.st_size,sha256=receipt['sha256'],inode=st.st_ino,evidence=str(evidence),reason=reason))
  for kind in ['pilot','worker']:
   root=ROOT/f'research_runs/tokenizer-readout-projection-{kind}-v1-001';evidence_roots.append(root)
   for slot in range(3):
    p=root/f'evidence/pilots/{slot:02}/accepted.json';a=read(p)['acceptance']
    assert a['status']=='accepted' and a['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked']
    assert a['step']==(20 if kind=='pilot' else 4)
    add(a['binding']['parent'],p,'Accepted engineering prerequisite only. Projection QUALITY001 is CLOSED_FAILED_INCOMPLETE; no retry or downstream consumption authorized. Preserve every scientific and failed quality checkpoint.')
  closure=ROOT/'research/TOKENIZER_READOUT_PROJECTION_RESULTS.md';assert 'CLOSED_FAILED_INCOMPLETE' in closure.read_text();protected[str(closure)]=sha(closure)
  for dirname in ['general-rsi-time-qualification-001','general-rsi-composer-fp32-quality-001/qualification']:
   p=ROOT/'research/tokenizer-evidence'/dirname/'result.json';d=read(p)
   assert d.get('qualified') or d.get('integration_qualified')
   evidence_roots.append(p.parent)
   for arm in d['arms']:add(arm['checkpoint'],p,'Completed exact restart prerequisite; downstream panels finished. Scientific workers initialize fresh models and read qualification evidence, not these engineering checkpoints.')
  root=ROOT/'research/tokenizer-evidence/general-rsi-worker-qualification-001';evidence_roots.append(root)
  assert read(root/'result.json')['qualified']
  d=read(root/'slot-00/result.json');assert d['status']=='completed'
  for item in d['endpoints']:add(item['weights'],root/'slot-00/result.json','Completed four-update assembled-worker fixture; never scientific selection. Downstream native panels finished.')
  # Current parents and fresh general confirmations remain protected, as do the
  # selected Shakespeare references and all original acceptance/source evidence.
  for campaign in ['general-rsi-native-horizon-001','general-rsi-composer-fp32-quality-001']:
   root=ROOT/'research/tokenizer-evidence'/campaign;evidence_roots.append(root)
   for result in root.glob('slot-*/result.json'):
    for e in read(result)['endpoints']:
     r=e['weights'];assert sha(r['path'])==r['sha256'];protected[r['path']]=r['sha256']
  recipe=read(ROOT/'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json')
  for name in ['parent','h200']:
   r=recipe['representative'][name];assert sha(r['path'])==r['sha256'];protected[r['path']]=r['sha256']
  excluded={t['path'] for t in targets}
  for root in evidence_roots:
   for p in root.rglob('*'):
    if p.is_file() and str(p) not in excluded:protected[str(p)]=sha(p)
  # Freeze active direction references and explicitly document historical-only consumers.
  for name in ['research_state/general-rsi-state.json','research/GENERAL_RSI_NEXT_CAMPAIGNS.md','research/GENERAL_RSI_PSEUDO_REPETITION_NEXT.md','research/general_rsi_pseudo_training_v1.py']:
   p=ROOT/name;protected[str(p)]=sha(p)
  consumers={}
  for kind in ['pilot','worker']:
   pattern=f'tokenizer-readout-projection-{kind}-v1-001'
   r=subprocess.run(['rg','-l','--glob','*.py','--glob','*.json','--glob','*.md',pattern,'research','src','research_state'],cwd=ROOT,capture_output=True,text=True)
   assert r.returncode in (0,1);consumers[pattern]=r.stdout.splitlines()
  free=shutil.disk_usage(ROOT).free;required=19058917376;target=required+2*2**30
  assert len(targets)==12
  for t in targets:
   r=subprocess.run(['/usr/sbin/lsof','-t','--',t['path']],capture_output=True,text=True);assert r.returncode==1 and not r.stdout and not r.stderr
  manifest=dict(targets=targets,protected=protected,consumers=consumers,source_sha256=sha(__file__),free_before=free,required_free_bytes=required,target_with_extra_headroom=target,
   consumer_review='Projection references are closed prerequisite/closure records and source snapshots, not active work. Current pseudo study loads selected native-horizon slot01/03/05 time_ema parents only. No target is a parent, final fresh confirmation, selected weight or failure artifact.',
   authority='Standing human AGENTS.md authorization for exact obsolete concluded engineering payloads; narrowly supersedes older blanket retain-all statements. Original records stay unchanged.',
   forecast='All twelve128MiB endpoints + two512MiB active serialization slots +256MiB metadata +5GiB failed-write allowance +10GiB reserve. New qualification payloads already retained and charged in free space, not charged twice.',
   process_snapshot=process,scientific_panel_admitted=False)
  put(OUT/'manifest.json',manifest)
  for t in targets:
   p=Path(t['path']);assert p.stat().st_ino==t['inode'] and p.stat().st_size==t['size'] and sha(p)==t['sha256']
   journal(dict(event='before_unlink',**t));p.unlink();journal(dict(event='unlinked',**t))
  assert all(sha(p)==h for p,h in protected.items())
  after=shutil.disk_usage(ROOT).free
  put(OUT/'completed.json',dict(removed_bytes=sum(t['size'] for t in targets),removed_files=len(targets),protected_files_verified=len(protected),free_after=after,required_free_bytes=required,target_with_extra_headroom=target,remaining_deficit_bytes=max(0,target-after),manifest_sha256=sha(OUT/'manifest.json'),scientific_panel_admitted=False))
  print((OUT/'completed.json').read_text())
if __name__=='__main__':main()
