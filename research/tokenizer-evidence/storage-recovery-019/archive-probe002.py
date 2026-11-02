"""Copy-only zstd archive feasibility: original selected/fresh payload unchanged."""
import importlib.util,json,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parents[3];E=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('archive_primitives',R/'research/tokenizer-evidence/mla-score-preparation-001/archive-workers.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a);a.E=E

spec=importlib.util.spec_from_file_location('codec',E/'archive_codec.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec);codec.install(a)
from research.final_ln_bias_campaign_v1.pilot import idle
with a.session_lock(R/'research_state/tokenizer-worker-lock'),a.CpuPrerequisiteGuard(E/'archive-probe002-guard.json',R,timeout_seconds=180):
 idle()
 # Exercise the same OS output limiter at1KiB before the real512MiB capped stream.
 import resource,subprocess,sys
 limited=E/'output-cap-selftest.bin';assert not limited.exists()
 def small_cap():resource.setrlimit(resource.RLIMIT_FSIZE,(1024,1024))
 with limited.open('xb') as out:
  test=subprocess.run([sys.executable,'-c',"import os;os.write(1,b'x'*1024);os.write(1,b'y')"],stdout=out,stderr=subprocess.PIPE,preexec_fn=small_cap,timeout=10)
 assert test.returncode!=0 and limited.stat().st_size==1024
 a.publish(E/'output-cap-selftest.json',dict(status='passed',limit_bytes=1024,bytes_written=limited.stat().st_size,returncode=test.returncode,production_cap_bytes=codec.CAP,reserve_bytes=a.RESERVE))
 limited.unlink();a.syncdir(E)
 probe=json.loads((E/'raw-probe-plan.json').read_text());src=Path(probe['original']['path']);before=a.record(src);v=before
 assert before['sha256']==probe['original']['sha256'] and before['size']==probe['size']
 a.verify(probe['acceptance']);a.verify(probe['terminal']);a.verify(probe['h200'])
 relative=str(src.relative_to(R));files={relative:dict(size=v['size'],sha256=v['sha256'])};archive=E/'copy-only-probe.tar.zst'
 started=time.monotonic();a.create_archive(archive,files);a.verify_archive(archive,files)
 assert a.record(src)==before
 size=archive.stat().st_size;h=a.file_digest(archive)
 a.publish(E/'archive-probe-verified.json',dict(archive=str(archive),archive_bytes=size,archive_sha256=h,original=before,verified_members=files,original_unchanged=True))
 archive.unlink();a.syncdir(E);assert a.record(src)==before
 a.publish(E/'archive-probe.json',dict(status='verified_copy_only',original_unchanged=True,temporary_archive_removed=True,original_bytes=v['size'],archive_bytes=size,logical_saving_bytes=v['size']-size,seconds=time.monotonic()-started,free_bytes=shutil.disk_usage(R).free,scope='Feasibility only; no original selected/fresh checkpoint removed or replaced.'))
 print((E/'archive-probe.json').read_text())
