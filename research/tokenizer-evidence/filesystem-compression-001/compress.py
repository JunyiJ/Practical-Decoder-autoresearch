"""Lossless APFS compression of explicit historical text snapshots, never weights.

Retains original paths and byte streams; stages and hashes each copy before
atomic replacement. No recursive removal and no alteration to source/training.
"""
import json, os, shutil, stat, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent
REQUIRED=30631412484

def publish(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def syncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def info(p):
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
 return dict(size=s.st_size,sha256=file_digest(p),mode=stat.S_IMODE(s.st_mode),mtime_ns=s.st_mtime_ns,uid=s.st_uid,gid=s.st_gid,allocated_bytes=s.st_blocks*512,flags=s.st_flags)
def check(p,v,physical=False):
 w=info(p)
 assert all(w[k]==v[k] for k in ('size','sha256','mode','mtime_ns','uid','gid')),(str(p),w,v)
 if physical:assert w==v,(str(p),w,v)
 return w

def plan():
 files={str(p):info(ROOT/p) for p in sorted(Path('research_state').rglob('*knowledge*snapshot*.md')) if p.is_file() and not p.is_symlink()}
 assert len(files)==175 and sum(v['size'] for v in files.values())==2037153240
 assert all(v['flags']==0 for v in files.values())
 protected={}
 for base in ('research_runs','research_archives','research_state','data','src'):
  for p in sorted((ROOT/base).rglob('*')):
   n=str(p.relative_to(ROOT))
   if not p.is_file() or n in files or '__pycache__' in p.parts or p.name=='.lock' or 'tokenizer-worker-lock' in p.parts:continue
   if p.name.startswith('apfs-compression-probe-'):continue
   protected[n]=info(p)
 old=json.loads((E.parent/'checkpoint-compaction-002/plan.json').read_text())
 for n,v in old['protected'].items():assert protected[n]['sha256']==v['sha256'] and protected[n]['size']==v['size']
 publish(E/'plan.json',dict(authorization='Human requests another safe way to free storage, preserving selected models, every fresh pair and failure evidence.',operation='APFS transparent compression of historical text only; original paths, sizes, SHA256, mode, mtime and ownership preserved',script_sha256=file_digest(Path(__file__)),files=files,protected=protected,protected_tokenizer_weights=159,free_before=shutil.disk_usage(ROOT).free,required_free_bytes=REQUIRED,reserve_bytes=8*1024**3,probe=dict(sandbox_copy_invalid_original_unchanged=True,escalated_copy=json.loads((ROOT/'research_state/apfs-compression-probe-002.json').read_text())),restoration='Ordinary reads transparently decompress. To remove compression, copy with ditto --nohfsCompression --nopreserveHFSCompression --noclone to a new staging path, verify manifest bytes, then atomically replace. Reserve uncompressed space first. Never rerun old campaigns.'))
 print(json.dumps(dict(status='planned',files=len(files),logical_bytes=sum(v['size'] for v in files.values()),protected=len(protected))),flush=True)

def apply():
 p=json.loads((E/'plan.json').read_text());assert file_digest(Path(__file__))==p['script_sha256']
 assert not (E/'journal.jsonl').exists()
 for n,v in p['files'].items():check(ROOT/n,v,True)
 for n,v in p['protected'].items():check(ROOT/n,v,True)
 before=shutil.disk_usage(ROOT).free; receipts=[]
 with (E/'journal.jsonl').open('x') as log:
  for n,v in p['files'].items():
   src=ROOT/n;tmp=src.with_name(src.name+'.apfs-compression-001.tmp');assert not tmp.exists()
   assert shutil.disk_usage(ROOT).free>8*1024**3+v['size']
   check(src,v,True)
   subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(src),str(tmp)],check=True,timeout=30)
   w=check(tmp,v)
   assert w['flags']&32 and w['allocated_bytes']<v['allocated_bytes']
   with tmp.open('rb') as f:os.fsync(f.fileno())
   check(src,v,True)
   log.write(json.dumps(dict(event='verified_staged_copy',path=n,sha256=v['sha256'],allocated_before=v['allocated_bytes'],allocated_after=w['allocated_bytes']))+'\n');log.flush();os.fsync(log.fileno())
   os.replace(tmp,src);syncdir(src.parent);check(src,v)
   r=dict(path=n,sha256=v['sha256'],allocated_before=v['allocated_bytes'],allocated_after=w['allocated_bytes']);receipts.append(r)
   log.write(json.dumps(dict(event='replacement_verified',**r))+'\n');log.flush();os.fsync(log.fileno())
 for n,v in p['files'].items():check(ROOT/n,v)
 for n,v in p['protected'].items():check(ROOT/n,v,True)
 publish(E/'completed.json',dict(status='completed',files=len(receipts),logical_bytes_unchanged=sum(v['size'] for v in p['files'].values()),allocated_bytes_reclaimed=sum(r['allocated_before']-r['allocated_after'] for r in receipts),free_before=before,free_after=shutil.disk_usage(ROOT).free,required_free_bytes=REQUIRED,protected_files_verified=len(p['protected']),protected_tokenizer_weights_verified=159,all_original_paths_and_readable_bytes_preserved=True,training=False,holdout_consumed=False,manifest_sha256=file_digest(E/'plan.json'),journal_sha256=file_digest(E/'journal.jsonl')))
 print((E/'completed.json').read_text(),flush=True)

if __name__=='__main__':
 mode=sys.argv[1];assert mode in ('plan','apply')
 with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/(mode+'-guard.json'),ROOT,timeout_seconds=1800):
  plan() if mode=='plan' else apply()
