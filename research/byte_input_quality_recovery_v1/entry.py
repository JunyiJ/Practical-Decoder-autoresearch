"""New recovery entries: finish saved slot14; run only never-entered slots15–17."""
import fcntl
import os
import sys
from pathlib import Path

from research.byte_input_quality_v1.entry import pins
from research.byte_input_quality_v1.contract import slots
from research.byte_input_quality_v1.controller import selection
from .common import ROOT, ORIGINAL, BASE, OLD_WEIGHTS, WEIGHTS, read, put, sha, acceptance
from .guard import Guard, retained_bytes


def main(expected, index):
    assert index in (14, 15, 16, 17) and sha(BASE/'plan.json') == expected
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(75)
        recovery = read(BASE/'plan.json'); plan = read(ORIGINAL/'plan.json')
        assert sha(ORIGINAL/'plan.json') == recovery['original_plan_sha256']
        assert read(ORIGINAL/'STOP')['status'] == 'closed_failed_original_quality_controller'
        pins(recovery['sources']); pins(recovery['prerequisites'])
        accepted = [acceptance(i) for i in range(index)]
        spec = dict(slots()[index]); spec['learning_rate'] = selection(accepted[:9])[spec['variant']]
        if index > 14:
            assert not (ORIGINAL/'slots'/f'{index:02}'/'attempt.json').exists()
        retained = retained_bytes([OLD_WEIGHTS, WEIGHTS]); space = os.statvfs(ROOT)
        assert space.f_bavail*space.f_frsize >= 20*1024**3-retained
        out = BASE/'slots'/f'{index:02}'; archives = WEIGHTS/f'{index:02}'
        out.mkdir(); archives.mkdir()
        written = 0
        def publish(name, value):
            nonlocal written
            import json
            written += len((json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode())
            assert written <= 16*1024**2
            put(out/name, value)
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=recovery['original_plan_sha256'],
            recovery_plan_sha256=expected, spec=spec, no_replay=True))
        previous_elapsed = recovery['failed_guard_elapsed_seconds'] if index == 14 else 0
        try:
            with Guard(out/'guard.json', ROOT, weight_roots=[OLD_WEIGHTS, WEIGHTS],
                       timeout_seconds=7200-previous_elapsed, rss_limit_bytes=4*1024**3,
                       reserve_bytes=8*1024**3, poll_seconds=.5) as guard:
                pins(plan['sources']); pins(plan['assets']); pins(plan['prerequisites'])
                assert sys.flags.optimize == 0
                for key, value in plan['environment'].items():
                    assert os.environ.get(key) == value
                import torch
                guard.torch = torch
                lineage = dict(original_plan_sha256=recovery['original_plan_sha256'],
                    recovery_plan_sha256=expected, original_failed_controller=recovery['original_stop'],
                    original_scientific_rules_unchanged=True, new_training_updates=0 if index == 14 else 1200)
                if index == 14:
                    failure = read(Path(recovery['failure_record']))
                    for record in failure['original_evidence']:
                        pins([record])
                    for pattern in ('step*.json', 'diagnostic*.json'):
                        for source in sorted((ORIGINAL/'slots/14').glob(pattern)):
                            raw = source.read_bytes(); written += len(raw)
                            assert written <= 16*1024**2
                            with (out/source.name).open('xb') as destination:
                                destination.write(raw); destination.flush(); os.fsync(destination.fileno())
                            assert sha(out/source.name) == sha(source)
                    lineage.update(original_failed_slot=str(ORIGINAL/'slots/14'),
                        raw_receipt=failure['raw_receipt'], original_guard_remains_failed=True,
                        original_guard_elapsed_seconds=previous_elapsed,
                        resource_scope='Original guard observed training until its inventory error; recovery guard covers postprocessing. Original terminal guard failure is retained, not rewritten as passed.')
                    publish('lineage.json', lineage)
                    from .driver import run
                    run(plan, spec, archives, publish, failure['raw_receipt'], ORIGINAL/'slots/14')
                else:
                    publish('lineage.json', lineage)
                    from research.byte_input_quality_v1.driver import run
                    run(plan, spec, archives, publish)
                from research.byte_input_quality_v1.review import run as review
                publish('review.json', review(plan, spec, out))
                pins(plan['sources']); pins(plan['assets']); pins(plan['prerequisites'])
                pins(recovery['sources']); pins(recovery['prerequisites'])
            assert previous_elapsed + read(out/'guard.json')['elapsed_seconds'] < 7200
            publish('STOP', dict(status='complete_recovery_scientific_slot', no_replay=True,
                                original_failed_slot_not_reclassified=index == 14))
        except BaseException as exc:
            if not (out/'STOP').exists():
                publish('STOP', dict(status='closed_failed_recovery_scientific_slot', error=repr(exc), no_replay=True))
            raise


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]))
