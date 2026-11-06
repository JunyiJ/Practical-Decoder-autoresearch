"""Portable immutable metadata only; original artifacts and STOPs unchanged."""
import hashlib,json,os,shutil
from pathlib import Path
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
R=Path.cwd();B=Path(__file__).resolve().parent;O=B/'complete-metadata-mirror001'
def ref(p):
 s=p.lstat();assert p.is_file() and not p.is_symlink() and s.st_nlink==1
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 return dict(path=str(p),size=s.st_size,sha256=h)
with session_lock(GLOBAL_LOCK.parent):
 idle();assert not O.exists();assert (B/'quality001-pilot-retirement001/completed.json').is_file()
 roots=[R/'research_runs'/n for n in ('tokenizer-embedding-input-dropout-worker-v1-001','tokenizer-embedding-input-dropout-worker-v1-002','tokenizer-embedding-input-dropout-pilot-v1-001')]
 files=[]
 for root in roots:
  assert (root/'STOP').is_file()
  for p in sorted(root.rglob('*')):
   if p.is_file() and (p.name=='STOP' or p.suffix in ('.json','.jsonl','.log')):
    assert 'checkpoints' not in p.parts and p.stat().st_size<8*1024**2
    files.append((p,O/root.name/p.relative_to(root)))
 O.mkdir();rows=[]
 for p,dst in files:
  original=ref(p);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dst);copy=ref(dst)
  assert original['size']==copy['size'] and original['sha256']==copy['sha256'];assert ref(p)==original
  rows.append(dict(original=original,copy=copy))
 result=dict(status='complete_metadata_mirrored',source=ref(Path(__file__)),files=rows,metadata_files=len(rows),weight_copies=0,original_failure_and_STOP_preserved=True,retired_payloads_resolve_via_later_exact_manifests=True)
 with (B/'complete-metadata-mirror001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({k:v for k,v in result.items() if k!='files'}))
