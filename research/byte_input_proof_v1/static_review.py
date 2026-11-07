"""Original source-only consistency gate; not functional admission."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001/resume-model-proof001'


def save(name, record):
    payload = (json.dumps(record, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.iterdir()) + len(payload) < 2097152
    with (OUT/name).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert {p.name for p in OUT.iterdir()} == {'plan.json'}
        raw = (OUT/'plan.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected
        plan = json.loads(raw)
        save('attempt.json', dict(pid=os.getpid(), plan_sha256=expected, scope='static_only'))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120,
                                     rss_limit_bytes=4294967296, reserve_bytes=8589934592,
                                     poll_seconds=.25):
                trees = {}
                for pin in plan['sources'] + plan['old_v1_unchanged_expected']:
                    path = Path(pin['path'])
                    path = path if path.is_absolute() else ROOT/path
                    data = path.read_bytes()
                    assert len(data) == pin['size'] and hashlib.sha256(data).hexdigest() == pin['sha256']
                    tree = ast.parse(data, filename=str(path))
                    compile(tree, str(path), 'exec')  # Compile only, never exec/import.
                    trees[str(path.relative_to(ROOT))] = tree
                old = trees['research/byte_input_model_v1/optimizer.py']
                new = trees['research/byte_input_model_v2/optimizer.py']
                old_accesses = [n for n in ast.walk(old) if isinstance(n, ast.Attribute) and n.attr == 'state']
                new_accesses = [n for n in ast.walk(new) if isinstance(n, ast.Attribute) and n.attr == 'state']
                assert not old_accesses and new_accesses
                cls = next(n for n in new.body if isinstance(n, ast.ClassDef))
                methods = {n.name:n for n in cls.body if isinstance(n, ast.FunctionDef)}
                step = methods['step']
                assert isinstance(step.body[0], ast.Expr)
                assert ast.unparse(step.body[0]) == 'self.validate(require_gradients=True)'
                validate = ast.unparse(methods['validate'])
                assert 'self._validate_state(device)' in validate
                assert 'self._steps += 1' in ast.unparse(step.body[-1])
                # Literal budget accounts for two updates and both child states,
                # without assuming compression savings. Exact runtime costs await
                # the actual driver and must not be inferred from this arithmetic.
                parameter_coordinates = 2*28492416 + 4*16351859
                logit_coordinates = 6*8*128*50257
                # Conservative bound: <=3 state elements per parameter per step.
                coordinates = logit_coordinates + 3*parameter_coordinates + 6*parameter_coordinates
                worst_raw = coordinates*9
                framing_allowance = 268435456
                cap = 17179869184
                assert worst_raw + framing_allowance < cap
                stat = os.statvfs(ROOT)
                free = stat.f_bavail*stat.f_frsize
                entry = cap + 8589934592 + 2147483648
                save('review.json', dict(
                    accepted_source_consistency=True, functional_admission=False,
                    original_optimizer_state_accesses=len(old_accesses),
                    new_optimizer_state_accesses=len(new_accesses),
                    old_v1_pins_preserved=True, new_roles='UNASSIGNED',
                    source_count=len(plan['sources']),
                    full_coordinate_uncompressed_upper_bytes=worst_raw,
                    framing_codebook_metadata_allowance=framing_allowance,
                    prospective_proof_output_cap=cap, prospective_entry_bytes=entry,
                    observed_free_bytes=free, prospective_deficit=max(0, entry-free),
                    live_graph_RSS_MPS_and_time='UNQUALIFIED until implemented driver',
                    next_action='Implement guarded driver and rejection transactions; then complete runtime/role/static admission'))
            save('STOP', dict(status='complete_original_static_consistency_check', no_replay=True))
        except BaseException as exc:
            save('STOP', dict(status='closed_failed_static_consistency_check', error=repr(exc), no_replay=True))
            raise


if __name__ == '__main__':
    main(sys.argv[1])
