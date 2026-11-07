"""NEW pre-Torch TOP proof entry guard SOURCE; numerical execution UNADMITTED.

The original NEVER-run proof retains its own lock/attempt/MPS/phase/output/STOP.
This entry adds elapsed/RSS/reserve/output coverage before importing Torch.
"""
import time
BOOT_MONOTONIC = time.monotonic()
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
PROOF = ROOT / 'research/tokenizer-evidence/future-token-preparation-001/top-model-math001.py'
RUN_RELATIVE = 'research/tokenizer-evidence/future-token-preparation-001/top-model-math001'
OUTER_RESOURCES = {'elapsed_seconds':600,'RSS_bytes':4294967296,'reserve_bytes':8589934592,'output_bytes':536870912}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_observation(run):
    paths = list(run.iterdir())
    unsafe = any(path.is_symlink() or not path.is_file() for path in paths)
    return sum(path.stat().st_size for path in paths if path.is_file() and not path.is_symlink()), unsafe


def write(run, name, value):
    raw = (json.dumps(value,sort_keys=True,allow_nan=False) + '\n').encode()
    size, unsafe = output_observation(run)
    if unsafe or size + len(raw) > OUTER_RESOURCES['output_bytes']: raise ValueError('Original whole output cap or flat-file ownership differs')
    with (run / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


class OuterGuard(CpuPrerequisiteGuard):
    def __init__(self, run):
        super().__init__(run / 'outer-guard.json',ROOT,timeout_seconds=600,poll_seconds=.25)
        self.run = run

    def _sample(self):
        observation, reason = super()._sample()
        size, unsafe = output_observation(self.run)
        elapsed = time.monotonic() - BOOT_MONOTONIC
        observation.update(output_bytes=size,elapsed_since_entry_seconds=elapsed,unsafe_output_entry=unsafe)
        if reason is None and elapsed >= 600: reason = 'whole_entry_wall_time_limit'
        if reason is None and unsafe: reason = 'unsafe_nonflat_or_symlink_output'
        if reason is None and size >= OUTER_RESOURCES['output_bytes']: reason = 'whole_JSON_output_limit'
        return observation, reason


def main():
    if len(sys.argv) != 3: raise ValueError('Exact NEW original numerical plan and SHA required')
    path = Path(sys.argv[1]).resolve(); run = path.parent
    if path != ROOT / RUN_RELATIVE / 'plan.json': raise ValueError('Exact original TOP math output root required')
    if any((run / name).exists() for name in ('outer-attempt.json','outer-guard.json','outer-STOP','attempt.json','STOP','result.json')):
        raise ValueError('Original command closed or attempted; no replay/retry')
    original_argv = list(sys.argv)
    write(run,'outer-attempt.json',{'pid':os.getpid(),'plan_sha256':sys.argv[2],'no_replay':True,'no_retry':True,
                                 'guard_scope':'Before Torch import, whole elapsed/RSS/reserve/output; inner proof owns shared lock'})
    try:
        # Guard starts BEFORE plan/source validation and BEFORE runpy can import
        # the proof, StrictMpsGuard or Torch. First sample includes BOOT time.
        with OuterGuard(run):
            if sha(path) != original_argv[2]: raise ValueError('Original numerical plan SHA differs')
            plan = json.loads(path.read_text())
            if plan.get('execution_admitted') is not True: raise ValueError('SOURCE ONLY; numerical admission absent')
            if plan['outer_resources'] != OUTER_RESOURCES: raise ValueError('Whole-process guard resources differ')
            self_path = str(Path(__file__).resolve().relative_to(ROOT))
            if plan['outer_entry'] != self_path or plan['sources'].get(self_path) != sha(__file__): raise ValueError('Outer entry pin differs')
            if plan['sources'].get(str(PROOF.relative_to(ROOT))) != sha(PROOF): raise ValueError('Original NEVER-run proof pin differs')
            admission = json.loads((ROOT / plan['runtime_admission_review']).read_text())
            if (admission['status'] != 'complete_NEW_TOP_model_math_runtime_admission'
                    or admission['accepted'] is not True or admission['numerical_plan_sha256'] != original_argv[2]):
                raise ValueError('Independent original runtime admission missing')
            # The original proof obtains the shared nonblocking worker lock and
            # writes its own attempt BEFORE importing StrictMpsGuard/Torch.
            sys.argv = [str(PROOF),str(path),original_argv[2]]
            runpy.run_path(str(PROOF),run_name='__main__')
            stop = json.loads((run / 'STOP').read_text())
            result = json.loads((run / 'result.json').read_text())
            if stop['status'] != 'complete_NEW_TOP_model_math_finite_scope_only' or result['status'] != stop['status']:
                raise ValueError('Inner original proof not complete')
        write(run,'outer-STOP',{'status':'complete_original_NEW_TOP_outer_guarded_math_entry',
                              'no_replay':True,'no_retry':True,'model_quality_or_state_qualification':False})
    except BaseException as error:
        if not (run / 'outer-STOP').exists():
            write(run,'outer-STOP',{'status':'closed_failed_original_NEW_TOP_outer_entry','error':repr(error),'no_replay':True,'no_retry':True})
        raise
    finally:
        sys.argv = original_argv


if __name__ == '__main__': main()
