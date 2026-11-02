"""Copy-only lossless compression probe. Original checkpoint never replaced."""
import hashlib,importlib.util,json,os,shutil,subprocess,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.parallel_residual_campaign_v1.pilot import idle
s=importlib.util.spec_from_file_location('compression_checks',R/'research/tokenizer-evidence/filesystem-compression-001/compress.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
with session_lock(GLOBAL_LOCK.parent),CpuPrerequisiteGuard(E/'probe-guard.json',R,timeout_seconds=180):
 idle();assert not (E/'probe.json').exists()
 accepted=R/'research_runs/tokenizer-decay-quality-v2-001/evidence/quality/00/accepted.json'
 receipt=json.loads(accepted.read_text());assert receipt['artifact_acceptance']['accepted']
 assert (accepted.parents[3]/'STOP').exists()
 binding=receipt['artifact_acceptance']['files']['parent'];src=Path(binding['path']);v=a.info(src)
 assert v['sha256']==binding['sha256'] and v['size']<512*1024**2
 assert shutil.disk_usage(R).free>=8*1024**3+v['size']
 tmp=E/'verified-copy.pt.tmp';assert not tmp.exists();started=time.monotonic()
 a.publish(E/'probe-plan.json',dict(source=a.info(Path(__file__)),accepted_path=str(accepted),accepted_sha256=a.info(accepted)['sha256'],original_path=str(src),original=v,temporary_path=str(tmp),operation='Copy-only APFS transparent compression; do not replace or remove original.'))
 subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(src),str(tmp)],check=True,timeout=90)
 w=a.check(tmp,v);a.check(src,v,True)
 with tmp.open('rb') as f:os.fsync(f.fileno())
 a.publish(E/'probe-copy-verified.json',dict(original_sha256=v['sha256'],temporary_sha256=w['sha256'],temporary=str(tmp),size=w['size'],original_unchanged=True))
 tmp.unlink();a.syncdir(E);a.check(src,v,True)
 a.publish(E/'probe.json',dict(status='verified_copy_only',original_path=str(src),original=v,compressed_copy=w,allocated_saving_bytes=v['allocated_bytes']-w['allocated_bytes'],elapsed_seconds=time.monotonic()-started,original_unchanged=True,temporary_removed=True,scientific_training=False,free_bytes=shutil.disk_usage(R).free))
 print(json.dumps(dict(status='verified_copy_only',saving=v['allocated_bytes']-w['allocated_bytes'],seconds=time.monotonic()-started)))
