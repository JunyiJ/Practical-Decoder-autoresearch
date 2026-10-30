import json,os,shutil,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_driver import session_lock
E=Path(__file__).resolve().parent
with session_lock(ROOT/'research_state/tokenizer-worker-lock'),CpuPrerequisiteGuard(E/'sparse-probe-guard.json',ROOT,timeout_seconds=240):
 first=json.loads((E/'probe.json').read_text());source=ROOT/first['source'];prior=ROOT/first['copy']
 assert file_digest(source)==first['sha256'] and file_digest(prior)==first['sha256']
 prior.unlink() # Only the verified scratch duplicate created by probe.py.
 target=ROOT/'research_state/checkpoint-storage-compaction-001/sparse-probe.pt'
 zero=b'\0'*65536;skipped=0;start=time.monotonic()
 with source.open('rb') as inp,target.open('xb') as out:
  while data:=inp.read(len(zero)):
   if data==zero:out.seek(len(data),1);skipped+=len(data)
   else:out.write(data)
  out.truncate(source.stat().st_size);out.flush();os.fsync(out.fileno())
 shutil.copystat(source,target)
 assert target.stat().st_size==source.stat().st_size and file_digest(target)==first['sha256'] and file_digest(source)==first['sha256']
 result=dict(source=first['source'],copy=str(target.relative_to(ROOT)),size=source.stat().st_size,sha256=first['sha256'],
  source_allocated_bytes=source.stat().st_blocks*512,copy_allocated_bytes=target.stat().st_blocks*512,
  zero_bytes_skipped=skipped,elapsed_seconds=time.monotonic()-start,source_unchanged=True,deleted_scratch_duplicate=first['copy'])
 with (E/'sparse-probe.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result),flush=True)
