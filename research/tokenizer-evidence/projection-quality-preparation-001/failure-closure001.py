"""One-time read-only closure of the original incomplete projection panel."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
B = Path(__file__).resolve().parent
OUT = B / 'failure-closure001'
RUN = ROOT / 'research_runs/tokenizer-readout-projection-quality-v1-001'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def put(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) + len(raw) < 8*1024**2
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def main(plan_sha):
    plan_path = OUT / 'plan.json'
    assert sha(plan_path) == plan_sha
    plan = read(plan_path)
    assert sha(Path(__file__)) == plan['source_sha256']
    assert not (OUT / 'attempt.json').exists() and not (OUT / 'STOP').exists()
    locks = []
    for path in [ROOT/'research_state/tokenizer-worker-lock/.lock', RUN/'.lock']:
        f = path.open('r+')
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        locks.append(f)
    put(OUT/'attempt.json', dict(plan_sha256=plan_sha, pid=os.getpid(), numerical_calls=False))
    with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
        pins = read(B/'admission-independent-review001/plan.json')['all_execution_and_proof_sources']
        assert len(pins) == 1164
        for path, digest in pins.items():
            assert sha(ROOT/path) == digest, path
        stop = read(RUN/'STOP')
        assert stop == {'error': "ValueError('Complete CPU/MPS native CE/coverage differs')", 'mode': 'advance', 'no_retry': True}
        assert sorted(p.name for p in (RUN/'evidence/quality').iterdir()) == [f'{i:02d}' for i in range(8)]
        assert not any((B/name).exists() for name in plan['absent_scientific_outputs'])
        archives, rows = [], []
        for slot in range(8):
            d = RUN/f'evidence/quality/{slot:02d}'
            w = read(d/'worker.json')
            assert w['step'] == 1200 and w['targets'] == 1228800 and w['ema_snapshots'] == 41
            assert set(w['parameter_gradient_steps'].values()) == {1200}
            spec = read(d/'request.json')['spec']
            assert spec['slot'] == slot and w['config'] == spec['config']
            proc = read(d/'process/process.json')
            assert proc['status'] == 'completed' and proc['returncode'] == 0
            wg = read(RUN/f'control/worker-quality-{slot:02d}-guard.json')
            assert wg['status'] == 'finished' and wg['reason'] is None
            pg = read(RUN/f'control/parent-quality-{slot:02d}-guard.json')
            row = dict(slot=slot, arm=spec['arm'], lr=w['config']['training']['learning_rate'], steps=1200,
                       targets=1228800, worker_completed=True, independently_accepted=slot < 7,
                       parent_guard_status=pg['status'], parent_guard_reason=pg['reason'])
            if slot < 7:
                a = read(d/'accepted.json')
                assert a['slot'] == slot and a['worker']['sha256'] == sha(d/'worker.json')
                assert pg['status'] == 'finished' and pg['reason'] is None
                row['cpu_mps_ce_differences'] = {k: a['acceptance']['full_validation'][k]['cpu_mps_ce_difference'] for k in ('raw', 'h200')}
                assert all(v <= 3e-5 for v in row['cpu_mps_ce_differences'].values())
            else:
                assert not (d/'accepted.json').exists()
                assert pg['status'] == 'fixture_exception' and pg['reason'] == stop['error']
                assert (d/'cpu-raw-guard.json').exists() and not (d/'cpu-h200-guard.json').exists()
            rows.append(row)
            for key in ('parent', 'h200'):
                ref = w[key]
                path = Path(ref['path'])
                assert path == d/'checkpoints'/f'{key}.pt' and not path.is_symlink()
                s = path.stat()
                assert s.st_nlink == 1 and s.st_size == ref['size'] and sha(path) == ref['sha256']
                after = path.stat()
                assert (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns) == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                archives.append(dict(**ref, device=s.st_dev, inode=s.st_ino, links=s.st_nlink,
                                     allocated_bytes=s.st_blocks*512, slot=slot, kind=key,
                                     retention='protected_original_no_deletion_or_retirement_admission'))
        metadata = []
        for path in sorted(RUN.rglob('*')):
            if not path.is_file() or path.name == '.lock' or path.suffix in ('.pt', '.tmp'):
                continue
            assert not path.is_symlink()
            raw = path.read_bytes()
            dest = OUT/'original'/path.relative_to(RUN)
            dest.parent.mkdir(parents=True, exist_ok=True)
            assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file()) + len(raw) < 8*1024**2
            with dest.open('xb') as f:
                f.write(raw)
                f.flush()
                os.fsync(f.fileno())
            metadata.append(dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
        assert sha(RUN/'STOP') == plan['root_stop_sha256']
        result = dict(status='CLOSED_FAILED_INCOMPLETE_CPU_MPS_RAW_ACCEPTANCE', sources_preserved=1164,
                      original_controller_exit_code=1, original_controller_pid=48758, original_exec_session=2476,
                      failed_slot=7, failed_arm='linear_projection1', failed_lr=.0025,
                      accepted_slots=list(range(7)), unattempted_slots=list(range(8, 18)), rows=rows,
                      original_metadata=metadata, retained_archives=archives,
                      retained_archive_bytes=sum(a['size'] for a in archives),
                      failure_phase='RAW CPU/MPS CE/target comparison before H200 scoring/cache acceptance',
                      exact_failed_ce_values_or_comparison_operand='NOT_PERSISTED_UNAVAILABLE_NO_RECONSTRUCTION',
                      scientific_selection=False, fresh_confirmation=False, bootstrap=False, adoption=False,
                      native_incumbent_unchanged=True, holdout_closed=True,
                      no_model_tensor_load_scoring_training_rng_census_calls=True,
                      no_retry_no_stop_clear_no_source_or_tolerance_rescue=True,
                      closure_creates_no_retirement_eligibility=True)
        put(OUT/'review.json', result)
    put(OUT/'STOP', dict(status='complete_one_time_read_only_failed_panel_closure', no_replay=True))
    print(json.dumps({k: result[k] for k in ('status', 'accepted_slots', 'unattempted_slots', 'retained_archive_bytes')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan-sha', required=True)
    main(p.parse_args().plan_sha)
