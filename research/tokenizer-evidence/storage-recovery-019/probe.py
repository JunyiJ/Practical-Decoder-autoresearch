"""Copy-only transparent H200 compression probe; original selected/fresh unchanged."""
import hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
from research.adamw_beta2_quality_v1.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
E=Path(__file__).resolve().parent;R=E.parents[2]
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def bound(v):
    p=Path(v['path']);assert sha(p)==v['sha256'];return p
def info(p):
    s=p.stat();return dict(size=s.st_size,allocated=s.st_blocks*512,flags=s.st_flags,sha256=sha(p))
def publish(name,d):
    raw=(json.dumps(d,indent=2)+'\n').encode();assert len(raw)<1024**2
    with (E/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
with session_lock(GLOBAL_LOCK.parent),CpuPrerequisiteGuard(E/'probe-guard.json',R,timeout_seconds=180):
    idle();p=json.loads((E/'probe-plan.json').read_text());src=bound(p['original'])
    assert json.loads(bound(p['terminal']).read_text())['status']=='closed_not_adopted'
    accepted=json.loads(bound(p['acceptance']).read_text());assert accepted['artifact_acceptance']['accepted']
    assert accepted['artifact_acceptance']['files']['h200']==p['original']
    before=info(src);assert before['size']==p['size']
    target=E/'copy-only-h200.pt';assert not target.exists()
    assert shutil.disk_usage(R).free>=p['reserve_bytes']+p['max_output_bytes']+16*1024**2
    subprocess.run([sys.executable,'-c',"import os,resource,sys; cap=int(sys.argv[1]); resource.setrlimit(resource.RLIMIT_FSIZE,(cap,cap)); os.execv('/usr/bin/ditto',['ditto','--hfsCompression','--noclone',sys.argv[2],sys.argv[3]])",str(p['max_output_bytes']),str(src),str(target)],check=True,timeout=120)
    copy=info(target);assert copy['size']==before['size'] and copy['sha256']==before['sha256']
    assert info(src)==before
    publish('probe-verified.json',dict(source=dict(path=str(Path(__file__)),sha256=sha(Path(__file__))),original=p['original'],before=before,copy=copy,original_unchanged=True))
    target.unlink()
    fd=os.open(E,os.O_RDONLY);os.fsync(fd);os.close(fd)
    publish('probe-result.json',dict(status='verified_copy_only',original_unchanged=True,temporary_copy_removed=not target.exists(),allocated_saving=before['allocated']-copy['allocated'],transparent_compression=bool(copy['flags']&32),free_bytes=shutil.disk_usage(R).free))
    print((E/'probe-result.json').read_text())
