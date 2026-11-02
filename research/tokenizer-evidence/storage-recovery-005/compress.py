"""Lossless transparent compression of exact historical text; weights untouched."""
import importlib.util,json,os,shutil,sys
from pathlib import Path
R=Path(__file__).resolve().parents[3];sys.path.insert(0,str(R));E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('old',R/'research/tokenizer-evidence/filesystem-compression-001/compress.py');a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
from research.ffn_norm_quality_v1.support import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock

def selected():
 out={}
 for root in ('research_runs','research_state','research'):
  for p in sorted((R/root).rglob('*')):
   if E in p.parents or '__pycache__' in p.parts or 'maintenance-cache-cleanup-002' in p.parts or not p.is_file() or p.is_symlink() or p.suffix not in ('.md','.json','.jsonl','.log','.txt'):continue
   if p.name in ('tokenizer-quality-progress.json','TOKENIZER_RESUME_STATE.json'):continue
   s=p.stat()
   if s.st_size>=128*1024 and s.st_blocks*512>s.st_size*.8 and not(s.st_flags&32):out[str(p.relative_to(R))]=a.info(p)
 return out

def plan():
 files=selected();assert files
 protected={}
 for root in ('research_runs','research_archives','research_state','research','src','data'):
  for p in sorted((R/root).rglob('*')):
   n=str(p.relative_to(R))
   if n in files or E in p.parents or '__pycache__' in p.parts or 'maintenance-cache-cleanup-002' in p.parts or not p.is_file() or p.is_symlink() or p.name=='.lock' or 'tokenizer-worker-lock' in p.parts or p.name=='tokenizer-quality-progress.json':continue
   protected[n]=a.info(p)
 a.publish(E/'compression-plan.json',dict(files=files,protected=protected,source=a.info(Path(__file__)),free_before=shutil.disk_usage(R).free,authorization='Latest human: find another way to free remaining space without deleting selected models or fresh confirmations. Lossless APFS compression preserves original paths and exact readable bytes; no weights selected.',required_free=9210691584))
 print(json.dumps(dict(files=len(files),allocated=sum(x['allocated_bytes'] for x in files.values()),protected=len(protected))),flush=True)
def apply():
 p=json.loads((E/'compression-plan.json').read_text());a.check(Path(__file__),p['source'],True)
 for n,v in p['files'].items():a.check(R/n,v,True)
 for n,v in p['protected'].items():a.check(R/n,v,True)
 reclaimed=0
 with (E/'compression-journal.jsonl').open('x') as log:
  def emit(row):log.write(json.dumps(row)+'\n');log.flush();os.fsync(log.fileno())
  for n,v in p['files'].items():
   src=R/n;tmp=src.with_name(src.name+'.storage-recovery-005.tmp');assert not tmp.exists() and shutil.disk_usage(R).free>=8*1024**3+v['size']
   a.check(src,v,True)
   a.subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(src),str(tmp)],check=True,timeout=30)
   w=a.check(tmp,v)
   with tmp.open('rb') as f:os.fsync(f.fileno())
   a.check(src,v,True)
   if w['allocated_bytes']>=v['allocated_bytes']:
    emit(dict(event='verified_copy_no_savings',path=n));tmp.unlink();a.syncdir(tmp.parent);continue
   assert w['flags']&32
   emit(dict(event='verified_staged_copy',path=n,sha256=v['sha256'],allocated_before=v['allocated_bytes'],allocated_after=w['allocated_bytes']))
   os.replace(tmp,src);a.syncdir(src.parent);a.check(src,v);reclaimed+=v['allocated_bytes']-w['allocated_bytes']
   emit(dict(event='replacement_verified',path=n,sha256=v['sha256']))
 for n,v in p['files'].items():a.check(R/n,v)
 for n,v in p['protected'].items():a.check(R/n,v,True)
 a.publish(E/'compression-completed.json',dict(status='completed',files=len(p['files']),allocated_bytes_reclaimed=reclaimed,free_after=shutil.disk_usage(R).free,protected_files_verified=len(p['protected']),original_paths_and_bytes_preserved=True,weights_unchanged=True))
 print((E/'compression-completed.json').read_text(),flush=True)
if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('plan','apply')
 with session_lock(GLOBAL_LOCK.parent),CpuPrerequisiteGuard(E/f'compression-{mode}-guard.json',R,timeout_seconds=900):
  idle();plan() if mode=='plan' else apply()
