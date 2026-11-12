"""Exact retirement of accepted obsolete pilot/worker payloads from closed studies."""
import fcntl,json,hashlib,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];OUT=Path(__file__).resolve().parent

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
def processes():
 text=subprocess.check_output(['ps','-axo','pid,ppid,etime,command'],text=True)
 workers=[r for r in text.splitlines() if 'python' in r.lower() and any(s in r for s in ('general_rsi','tokenizer')) and int(r.split()[0])!=os.getpid()]
 assert not workers,workers
 return text
def unused(path):
 r=subprocess.run(['/usr/sbin/lsof','-t','--',str(path)],capture_output=True,text=True);assert r.returncode==1 and not r.stdout and not r.stderr

def main(action):
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);process=processes()
  if action=='prepare':
   targets=[];protected={};consumers={};roots=[]
   for family in ('readout-mixture','mla-gate'):
    quality=ROOT/f'research_runs/tokenizer-{family}-quality-v1-001';stop=read(quality/'STOP')
    assert stop['status']=='complete_finite_scientific_panel' and stop['never_rerun_training_inference_review']
    assert sha(stop['review']['path'])==stop['review']['sha256'];protected[stop['review']['path']]=stop['review']['sha256'];roots.append(quality)
    for kind in ('pilot','worker'):
     root=ROOT/f'research_runs/tokenizer-{family}-{kind}-v1-001';roots.append(root)
     for slot in range(3):
      evidence=root/f'evidence/pilots/{slot:02d}/accepted.json';a=read(evidence)['acceptance']
      assert a['status']=='accepted' and a['complete_raw_optimizer_ema_rng_sampler_clipping_h200_checked'] and a['step']==(20 if kind=='pilot' else 4)
      for endpoint in ('parent','h200'):
       r=a['binding'][endpoint];path=Path(r['path']);assert path.resolve()==path and path.parent==root/f'evidence/pilots/{slot:02d}/checkpoints' and not path.is_symlink()
       st=path.stat();assert st.st_nlink==1 and sha(path)==r['sha256'];unused(path)
       targets.append(dict(path=str(path),size=st.st_size,sha256=r['sha256'],inode=st.st_ino,evidence=str(evidence),endpoint=endpoint,reason='Accepted engineering pilot/worker checkpoint. Its downstream finite scientific panel is complete; no active consumer. Never a selected or fresh scientific confirmation.'))
     pattern=root.name
     r=subprocess.run(['rg','-l','--glob','*.py','--glob','*.json','--glob','*.md',pattern,'research','src','research_state'],cwd=ROOT,capture_output=True,text=True);assert r.returncode in (0,1);consumers[pattern]=r.stdout.splitlines()
   excluded={t['path'] for t in targets}
   for root in roots:
    for path in root.rglob('*'):
     if path.is_file() and str(path) not in excluded:protected[str(path)]=sha(path)
   # Current selected/fresh general endpoints, active new prerequisites, and Shakespeare reference.
   for campaign in ['general-rsi-native-horizon-001','general-rsi-composer-fp32-quality-001','general-rsi-pseudo-repetition-001']:
    for path in (ROOT/'research_runs'/campaign).rglob('*.pt'):protected[str(path)]=sha(path)
   for name in ['general-rsi-context-length-pilot-001','general-rsi-context-length-integration-001']:
    for path in (ROOT/'research_runs'/name).rglob('*'):
     if path.is_file():protected[str(path)]=sha(path)
   recipe=read(ROOT/'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json')
   for n in ['parent','h200']:
    r=recipe['representative'][n];assert sha(r['path'])==r['sha256'];protected[r['path']]=r['sha256']
   for name in ['research/GENERAL_RSI_NEXT_CAMPAIGNS.md','research_state/general-rsi-state.json','research/TOKENIZER_MLA_GATE_RESULTS.md','research/TOKENIZER_READOUT_MIXTURE_RESULTS.md']:
    path=ROOT/name;protected[str(path)]=sha(path)
   assert len(targets)==24
   required=19058917376;target=required+2*2**30;free=shutil.disk_usage(ROOT).free
   put(OUT/'manifest.json',dict(targets=targets,protected=protected,consumers=consumers,process_snapshot=process,source_sha256=sha(__file__),free_before=free,required_free_bytes=required,target_with_extra_headroom=target,projected_reclaimed_bytes=sum(t['size'] for t in targets),authority='Latest human AGENTS.md authorizes exact obsolete completed pilots/worker payload retirement. Narrow exception supersedes older retain-all prerequisite wording in the two closure reports. Preserve all original metadata and every scientific checkpoint.',consumer_review='Consumers are historical preparation/acceptance/closure references and source snapshots for the two terminal closed studies. Current native context work constructs fresh models and uses neither family nor these payloads. Fresh quality confirmations, selected models and all failures excluded.',forecast='Prospective six paired context runs: twelve128MiB endpoint caps + two512MiB active/temporary restart slots +256MiB metadata +5GiB failed-output allowance +10GiB reserve +2GiB extra headroom. Existing retained outputs already charged in free space.',scientific_panel_admitted=False))
   print(json.dumps(dict(files=len(targets),bytes=sum(t['size'] for t in targets),free=free,target=target,projected_free=free+sum(t['size'] for t in targets))))
  elif action=='execute':
   m=read(OUT/'manifest.json');assert sha(__file__)==m['source_sha256']
   for t in m['targets']:
    p=Path(t['path']);assert not p.is_symlink() and p.stat().st_nlink==1 and p.stat().st_ino==t['inode'] and p.stat().st_size==t['size'] and sha(p)==t['sha256'];unused(p)
   assert all(sha(p)==h for p,h in m['protected'].items())
   for t in m['targets']:
    p=Path(t['path']);assert p.stat().st_ino==t['inode'] and sha(p)==t['sha256'];journal(dict(event='before_unlink',**t));p.unlink();journal(dict(event='unlinked',**t))
   assert all(sha(p)==h for p,h in m['protected'].items())
   after=shutil.disk_usage(ROOT).free
   put(OUT/'completed.json',dict(removed_files=len(m['targets']),removed_bytes=sum(t['size'] for t in m['targets']),protected_files_verified=len(m['protected']),free_after=after,required_free_bytes=m['required_free_bytes'],target_with_extra_headroom=m['target_with_extra_headroom'],remaining_deficit_bytes=max(0,m['target_with_extra_headroom']-after),manifest_sha256=sha(OUT/'manifest.json'),scientific_panel_admitted=False))
   print((OUT/'completed.json').read_text())
  else:raise ValueError(action)
if __name__=='__main__':main(sys.argv[1])
