"""New regularization budget: compress only newly closed exact metadata paths."""
import importlib.util,json,os,shutil,stat,subprocess
from pathlib import Path
R=Path.cwd();E=Path(__file__).resolve().parent;REQUIRED=14892462080;TARGET=17039945728
s=importlib.util.spec_from_file_location('compression_primitives',R/'research/tokenizer-evidence/filesystem-compression-001/compress.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle

def info(p):
 st=p.lstat();assert not p.is_symlink() and st.st_uid==os.getuid() and st.st_nlink==1 and stat.S_ISREG(st.st_mode)
 return dict(a.info(p),device=st.st_dev,inode=st.st_ino,nlink=st.st_nlink,reason='Newly closed human-authorized old-checkpoint audit/retirement metadata only;lossless readable bytes/path preserved,no tensor payload')
def check(p,v,physical=False):
 a.check(p,v)
 if physical:assert info(p)==v
with session_lock(GLOBAL_LOCK.parent):
 idle();assert not(E/'plan.json').exists()
 with CpuPrerequisiteGuard(E/'guard.json',R,timeout_seconds=300,poll_seconds=.25):
  assert json.load((R/'research/tokenizer-evidence/human-old-checkpoint-retirement-002/completed.json').open())['deleted_files']==404
  assert json.load((R/'research/tokenizer-evidence/human-old-checkpoint-retirement-003/completed.json').open())['deleted_files']==3
  closed=set()
  for name in ('human-old-checkpoint-retirement-001','human-old-checkpoint-retirement-002','human-old-checkpoint-retirement-003'):
   for p in (R/'research/tokenizer-evidence'/name).rglob('*'):
    if p.is_file() and p.suffix in ('.json','.jsonl','.log'):closed.add(p)
  for p in (R/'research/maintenance-human-disk-cleanup-001').rglob('*'):
   if p.is_file() and p.suffix in ('.json','.jsonl','.log'):closed.add(p)
  attempted=set()
  for folder in (R/'research/tokenizer-evidence').iterdir():
   if not folder.is_dir() or folder==E or not folder.name.startswith(('storage-recovery-','filesystem-compression-')):continue
   for p in folder.glob('*plan*.json'):
    d=json.load(p.open());files=d.get('files',{}) if isinstance(d,dict) else {}
    if isinstance(files,dict):attempted.update(str((R/n).resolve()) if not Path(n).is_absolute() else n for n in files)
    elif isinstance(files,list):attempted.update(str((R/v['path']).resolve()) if not Path(v['path']).is_absolute() else v['path'] for v in files if isinstance(v,dict) and isinstance(v.get('path'),str))
  files={str(p):info(p) for p in sorted(closed) if p.stat().st_size>=4096 and not(p.stat().st_flags&32) and str(p) not in attempted}
  assert 0<len(files)<=1000 and sum(v['allocated_bytes'] for v in files.values())<=512*1024**2
  protected={}
  body=json.load((R/'research/tokenizer-evidence/body-allocation-preparation-001/quality001-closure001.json').open())
  scale=json.load((R/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-closure.json').open())
  refs=[*body['protected_original_payloads'],*scale['weights']]
  for r in refs:
   p=Path(r['path']);v=info(p);assert v['sha256']==r['sha256'];protected[str(p)]=v
  assert len(protected)==36
  cp=subprocess.run(['lsof','-nP','-Fpn','--',*files],capture_output=True,text=True,timeout=30);assert cp.returncode==1 and not cp.stdout.strip() and not cp.stderr.strip(),'Closed metadata has consumer'
  a.publish(E/'plan.json',dict(operation='lossless_APFS_new_closed_metadata_compression',source=info(Path(__file__)),files=files,protected_affected_models=protected,
    required=REQUIRED,target=TARGET,free_before=shutil.disk_usage(R).free,authorization='Human requests maximum safe disk cleanup; only newly closed retirement metadata is losslessly compressed. Old archive/H200 exceptions are documented in the separate exact404+3 retirement manifests.;all original readable bytes/paths preserved;no checkpoint modification or retired restoration.'))
  saved=0
  with (E/'journal.jsonl').open('x') as log:
   def event(d):log.write(json.dumps(d)+'\n');log.flush();os.fsync(log.fileno())
   for name,v in files.items():
    p=Path(name);check(p,v,True);tmp=p.with_name(p.name+'.storage041.tmp');assert not tmp.exists() and shutil.disk_usage(R).free>=8*1024**3+v['size']
    subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(p),str(tmp)],check=True,timeout=30);w=a.check(tmp,v)
    with tmp.open('rb') as f:os.fsync(f.fileno())
    check(p,v,True)
    if w['allocated_bytes']>=v['allocated_bytes']:event(dict(event='verified_no_savings',path=name));tmp.unlink();a.syncdir(p.parent);continue
    assert w['flags']&32;event(dict(event='verified_staged_copy',path=name,sha256=v['sha256'],before=v['allocated_bytes'],after=w['allocated_bytes']))
    os.replace(tmp,p);a.syncdir(p.parent);check(p,v);saved+=v['allocated_bytes']-w['allocated_bytes'];event(dict(event='replacement_verified',path=name,sha256=v['sha256']))
  for name,v in files.items():check(Path(name),v)
  for name,v in protected.items():check(Path(name),v,True)
  free=shutil.disk_usage(R).free;a.publish(E/'completed.json',dict(status='completed_lossless_metadata_compression',files=len(files),allocated_bytes_reclaimed=saved,
   protected_payload_hashes_verified=len(protected),all_paths_and_readable_bytes_unchanged=True,no_weights_modified=True,required=REQUIRED,target=TARGET,free_after=free,mandatory_deficit=max(0,REQUIRED-free),target_deficit=max(0,TARGET-free)))
  print((E/'completed.json').read_text())
