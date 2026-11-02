"""Bounded single-stream codec: tighter kernel-enforced output, same reserve/RSS/time."""
import os, resource, shutil, subprocess, tarfile, threading

ALLOWED={'research_runs/tokenizer-ema-horizon-v1-001/evidence/quality/00/checkpoints/final_001200_1793493546634603000.pt': {'size': 559083893, 'sha256': '63d70861767f3fe501786a7aed6a7829532509d79c0abbae6bf82f3e2badc6f5'}, 'research_runs/tokenizer-ema-horizon-v1-001/evidence/quality/01/checkpoints/final_001200_1793493959470750000.pt': {'size': 559083893, 'sha256': '8e18a53c0719af25288da9bca0dcb91e142018f329a359e01b8fb298b1ba9169'}, 'research_runs/tokenizer-ema-horizon-v1-001/evidence/quality/02/checkpoints/final_001200_1793494330535578000.pt': {'size': 559083893, 'sha256': '1a124f3209f75d93a067fc972dbfc2004f99edc25edf07ccff9755d12d6f74d1'}}
CAP=512*1024**2
METADATA=16*1024**2


def limit_output():
    resource.setrlimit(resource.RLIMIT_FSIZE,(CAP,CAP))


def install(a):
    def create(archive,files):
        # Input is streamed directly from the retained original, with no duplicate
        # RAW or staging tar. The only new large object is capped at512MiB by the
        # kernel. The old generic input-size+512MiB estimate is not used for this
        # separate path; the8GiB reserve and all monitored limits remain intact.
        assert not archive.exists() and len(files)==1
        assert all(name in ALLOWED and value==ALLOWED[name] for name,value in files.items())
        assert shutil.disk_usage(a.ROOT).free >= a.RESERVE+CAP+METADATA
        done=threading.Event();errors=[]
        with (a.E/'subprocess.log').open('ab') as err:
            proc=subprocess.Popen([a.ZSTD,'-9','--long=30','-T1','--no-progress','-o',str(archive)],
                stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,stderr=err,preexec_fn=limit_output)
        thread=threading.Thread(target=a.monitored,args=(proc,done,errors,archive),daemon=True);thread.start()
        state=(proc,done,thread,errors)
        try:
            with tarfile.open(fileobj=proc.stdin,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                for name,value in files.items():
                    a.check(a.ROOT/name,value);tar.add(a.ROOT/name,arcname=name,recursive=False)
            proc.stdin.close();a.finish(state)
        except BaseException:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
            done.set();thread.join();raise
        assert archive.stat().st_size <= CAP
        assert shutil.disk_usage(a.ROOT).free >= a.RESERVE
        with archive.open('rb') as f:os.fsync(f.fileno())
        a.syncdir(archive.parent)
    a.create_archive=create
