import json,subprocess,hashlib,shutil,os,time
from pathlib import Path
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.adamw_epsilon_quality_v1.support import idle
R=Path.cwd();E=Path(__file__).resolve().parent

def git(*a):return subprocess.check_output(['git',*a],cwd=R)
def save(name,d):
 with (E/name).open('x') as f:json.dump(d,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def objects():return sorted(git('cat-file','--batch-all-objects','--batch-check=%(objectname) %(objecttype) %(objectsize)').decode().splitlines())
def state():return dict(refs=git('show-ref').decode(),head=git('rev-parse','HEAD').decode(),index=hashlib.sha256((R/'.git/index').read_bytes()).hexdigest(),unstaged=hashlib.sha256(git('diff','--binary')).hexdigest())
with session_lock(GLOBAL_LOCK.parent),CpuPrerequisiteGuard(E/'guard.json',R,timeout_seconds=300):
 idle();assert not (R/'.git/index.lock').exists() and not (R/'.git/objects/pack/tmp_pack').exists()
 before=objects();snap=state();free=shutil.disk_usage(R).free
 save('plan.json',dict(operation='lossless_git_object_repack',authorization='Human storage cleanup for RSI iterations; preserve every Git object/ref/index/worktree',objects=before,state=snap,free_before=free,argv=['git','-c','pack.threads=1','-c','pack.windowMemory=128m','repack','-a','-d','--keep-unreachable'],no_prune=True,study_payloads_modified=False))
 with (E/'journal.jsonl').open('x') as j:
  j.write(json.dumps(dict(event='intent_lossless_repack',objects=len(before)))+'\n');j.flush();os.fsync(j.fileno())
  p=subprocess.run(['git','-c','pack.threads=1','-c','pack.windowMemory=128m','repack','-a','-d','--keep-unreachable'],cwd=R,capture_output=True,text=True,timeout=240)
  (E/'git.log').write_text(p.stdout+p.stderr);assert p.returncode==0
  assert objects()==before and state()==snap
  j.write(json.dumps(dict(event='verified_all_objects_refs_index_worktree_unchanged',objects=len(before)))+'\n');j.flush();os.fsync(j.fileno())
 save('completed.json',dict(status='verified_lossless_git_repack',all_objects_preserved=len(before),refs_index_worktree_unchanged=True,no_prune=True,free_before=free,free_after=shutil.disk_usage(R).free,study_payloads_modified=False))
 print((E/'completed.json').read_text())
