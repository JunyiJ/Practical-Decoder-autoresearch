"""Bounded single-thread level9 lossless archive encoder; inherited verification."""
import shutil,tarfile

def install(a):
 def create(archive,files):
  assert not archive.exists() and shutil.disk_usage(a.ROOT).free>=a.RESERVE+max(v['size'] for v in files.values())+512*1024**2
  state=a.run_stream([a.ZSTD,'-9','--long=30','-T1','--no-progress','-o',str(archive)],archive,'create');proc=state[0]
  try:
   with tarfile.open(fileobj=proc.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
    for n,v in files.items():a.check(a.ROOT/n,v);tar.add(a.ROOT/n,arcname=n,recursive=False)
   proc.stdin.close();a.finish(state)
  except BaseException:
   proc.terminate();proc.wait(timeout=10);state[1].set();state[2].join();raise
  with archive.open('rb') as f:a.os.fsync(f.fileno())
  a.syncdir(archive.parent)
 a.create_archive=create
