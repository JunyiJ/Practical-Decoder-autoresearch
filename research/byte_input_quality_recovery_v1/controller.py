"""Continue the exact original panel via saved slot14 and unused slots15–17."""
import fcntl
import os
import subprocess
import sys
import time
from pathlib import Path

from .common import ROOT, ORIGINAL, BASE, read, put, sha, slot_dir, acceptance, accept
from research.byte_input_quality_v1.controller import selection


def final(plan):
    rows = [acceptance(i) for i in range(18)]; selection(rows[:9]); pairs = []
    for bundle in range(3):
        group = rows[9+3*bundle:12+3*bundle]
        assert {r['variant'] for r in group} == {'native', 'raw_bytes', 'nonlexical_codes'}
        assert all(r['bundle'] == f'fresh{bundle}' and r['sampler_hashes'] == group[0]['sampler_hashes'] for r in group)
        pairs.append({r['variant']: read(slot_dir(r['slot'])/'validation.json')['report'] for r in group})
    for row in rows:
        for artifact in (row['parent'], row['h200']):
            assert sha(Path(artifact['path'])) == artifact['sha256']
    from research.byte_input_quality_v1.statistics import calculate
    result = calculate(pairs, plan['bootstrap_seed'])
    result.update(accepted_complete_panel=True, trajectories=18, updates=21600,
        selection=read(ORIGINAL/'selection.json'),
        all_slot_acceptances=[dict(slot=i, path=str(slot_dir(i)/'acceptance.json'),
                                  sha256=sha(slot_dir(i)/'acceptance.json')) for i in range(18)],
        execution_recovery=read(BASE/'plan.json'), original_failed_controller_not_reclassified=True,
        independent_final_acceptance_required=True, adopted=False)
    put(BASE/'final.json', result)


def main(expected):
    with (BASE/'.controller.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert sha(BASE/'plan.json') == expected and not (BASE/'controller-attempt.json').exists()
        recovery = read(BASE/'plan.json'); plan = read(ORIGINAL/'plan.json')
        original_elapsed = max(recovery['original_elapsed_seconds'],
            time.time()-read(ORIGINAL/'controller-attempt.json')['started_unix'])
        put(BASE/'controller-attempt.json', dict(pid=os.getpid(), recovery_plan_sha256=expected,
            original_plan_sha256=recovery['original_plan_sha256'], started_unix=time.time(), no_replay=True))
        started = time.monotonic()
        try:
            for index in range(14, 18):
                assert original_elapsed + time.monotonic()-started+7200 <= 130800
                while True:
                    code = subprocess.call([sys.executable, '-m', 'research.byte_input_quality_recovery_v1.entry',
                                            expected, str(index)], cwd=ROOT)
                    if code != 75:
                        break
                    assert original_elapsed + time.monotonic()-started+7200 <= 130800
                    time.sleep(20)
                assert code == 0, ('Recovery entry failed; preserve and diagnose, no automatic retry', index, code)
                row = accept(index)
                put(BASE/f'completed-{index:02}.json', dict(slot=index, review=row, elapsed_seconds=time.monotonic()-started))
            from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
            with CpuPrerequisiteGuard(BASE/'statistics-guard.json', ROOT, timeout_seconds=300,
                                     rss_limit_bytes=4*1024**3, reserve_bytes=8*1024**3):
                final(plan)
            put(BASE/'STOP', dict(status='complete_recovered_quality_panel_pending_independent_review', no_replay=True))
        except BaseException as exc:
            if not (BASE/'STOP').exists():
                put(BASE/'STOP', dict(status='closed_failed_recovery_controller', error=repr(exc), no_replay=True))
            raise


if __name__ == '__main__':
    main(sys.argv[1])
