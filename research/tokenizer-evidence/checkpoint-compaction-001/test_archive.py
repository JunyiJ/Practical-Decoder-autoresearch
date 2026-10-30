import importlib.util,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
E=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('archive',E/'archive.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with m.session_lock(ROOT/'research_state/tokenizer-worker-lock'),m.CpuPrerequisiteGuard(E/'archive-test-guard.json',ROOT,timeout_seconds=180):
 out=ROOT/'research_state/checkpoint-storage-compaction-001/archive-test';out.mkdir(exist_ok=False)
 files={}
 for name,data in [('zeros.bin',b'\0'*131072),('pattern.bin',bytes(range(256))*513)]:
  p=out/name;p.write_bytes(data);files[str(p.relative_to(ROOT))]=dict(size=len(data),sha256=m.file_digest(p))
 archive=out/'test.tar.zst';m.create_archive(archive,files);assert m.verify_archive(archive,files)==files
 bad={n:dict(v) for n,v in files.items()};bad[next(iter(bad))]['sha256']='0'*64
 rejected=False
 try:m.verify_archive(archive,bad)
 except AssertionError:rejected=True
 assert rejected and all(m.file_digest(ROOT/n)==v['sha256'] for n,v in files.items())
 m.publish(E/'archive-test.json',dict(roundtrip_verified=True,incorrect_member_hash_rejected=True,originals_unchanged=True,archive_bytes=archive.stat().st_size))
 print('Archive roundtrip and mismatched-member rejection passed',flush=True)
