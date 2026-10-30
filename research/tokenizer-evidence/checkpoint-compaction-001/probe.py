import json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'probe-guard.json',ROOT,timeout_seconds=240):
 source=ROOT/'research_runs/tokenizer-wide-hybrid-numerical-v2-001/03/partial.pt'
 out=ROOT/'research_state/checkpoint-storage-compaction-001';out.mkdir(exist_ok=False)
 target=out/'probe.pt'
 before=source.stat();h=file_digest(source);start=time.monotonic()
 command=['/usr/bin/ditto','--noclone','--hfsCompression',str(source),str(target)]
 p=subprocess.run(command,capture_output=True,text=True,timeout=180)
 assert p.returncode==0,(p.returncode,p.stderr)
 after=target.stat();assert after.st_size==before.st_size and file_digest(target)==h and file_digest(source)==h
 result=dict(source=str(source.relative_to(ROOT)),copy=str(target.relative_to(ROOT)),size=before.st_size,sha256=h,
  source_allocated_bytes=before.st_blocks*512,copy_allocated_bytes=after.st_blocks*512,copy_flags=after.st_flags,
  source_mtime_ns=before.st_mtime_ns,copy_mtime_ns=after.st_mtime_ns,
  elapsed_seconds=time.monotonic()-start,command=command,stdout=p.stdout,stderr=p.stderr,source_unchanged=True)
 with (E/'probe.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
