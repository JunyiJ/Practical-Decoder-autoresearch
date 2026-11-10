"""Full-model engineering qualification of the separately identified reuse path."""
import fcntl
import gc
import hashlib
import json
import os
import shutil
import signal
import statistics
import time
import traceback
import torch
from research.general_rsi_transfer_data_v1 import ROOT, read, put, sha, contract
from research import general_rsi_transfer_worker_v1 as loader
from research.general_rsi_training_v2 import Context, write_checkpoint, read_checkpoint
from research.general_rsi_reuse_runtime_v1 import ReuseContext
from research.general_rsi_composer_reuse_worker_v1 import compare, digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

OUT = ROOT/'research/tokenizer-evidence/general-rsi-reuse-full-model-001'


def tree_hashes(value, prefix=''):
    if isinstance(value, torch.Tensor):
        return {prefix: dict(shape=list(value.shape), dtype=str(value.dtype), sha256=digest(value))}
    if isinstance(value, dict):
        result = {prefix: dict(keys=list(value))}
        for key, child in value.items():
            result.update(tree_hashes(child, prefix+'/'+str(key)))
        return result
    if isinstance(value, (tuple, list)):
        result = {prefix: dict(type=type(value).__name__, length=len(value))}
        for i, child in enumerate(value):
            result.update(tree_hashes(child, prefix+'/'+str(i)))
        return result
    return {prefix: value}


def gradients(context):
    sampler = context.data.view.generators['train']
    before = sampler.get_state().clone()
    x, y = context.data.batch()
    output = context.model.forward_train(x, y)
    output.loss.backward()
    result = (output.loss.detach().cpu(), {n: p.grad.detach().cpu().clone() for n, p in context.model.ordered_parameters()})
    context.optimizer.zero_grad()
    sampler.set_state(before)
    return result


def run():
    cfg = read(OUT/'plan.json')
    for path, h in cfg['sources'].items():
        assert sha(ROOT/path) == h, path
    remaining = cfg['total_guard_seconds']-sum(read(p)['elapsed_seconds'] for p in OUT.glob('guard-*.json'))
    assert remaining > 0
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free > cfg['reserve_bytes']+cfg['retained_output_cap_bytes']
        with CpuPrerequisiteGuard(OUT/'guard-01.json', ROOT, timeout_seconds=remaining,
                                 rss_limit_bytes=4*1024**3, reserve_bytes=cfg['reserve_bytes']):
            dest = OUT/'attempt-01'
            dest.mkdir()
            statepath = ROOT/'research_state/general-rsi-state.json'
            state = read(statepath)
            state.update(status='COMPOSER_REUSE_FULL_MODEL_RUNNING', active_worker_pid=os.getpid(), active_session=None)
            loader.replace(statepath, state)
            try:
                protocol = contract()
                loader.validate_sources(protocol)
                book = loader.setup(protocol)
                torch.use_deterministic_algorithms(True)
                args = ('raw_bytes', .0025, 40, cfg['roles'], book, sha(OUT/'plan.json'))
                baseline = Context(*args)
                candidate = ReuseContext(*args)
                initial_a = {n: digest(p) for n, p in baseline.model.ordered_parameters()}
                initial_b = {n: digest(p) for n, p in candidate.model.ordered_parameters()}
                assert initial_a == initial_b and baseline.model.identity != candidate.model.identity
                assert baseline.base_lrs == candidate.base_lrs
                old_groups = baseline.capture()['children']
                new_groups = candidate.capture()['children']
                assert old_groups == new_groups
                loader.check()
                a, ag = gradients(baseline)
                b, bg = gradients(candidate)
                checks = dict(loss=compare(b, a, cfg['gradient_tolerance']),
                              gradients={n: compare(bg[n], ag[n], cfg['gradient_tolerance']) for n in ag})
                put(dest/'initial-gradient-comparison.json', checks)
                passed = checks['loss']['passed'] and all(v['passed'] for v in checks['gradients'].values())
                if not passed:
                    receipt = write_checkpoint(dest/'first-failure-gradients.pt', dict(baseline_loss=a, reuse_loss=b, baseline_gradients=ag, reuse_gradients=bg))
                    put(dest/'first-failure-gradients-receipt.json', receipt)
                    put(dest/'decision.json', dict(passed=False, reason='initial_full_model_gradient_gate', training_updates=0))
                    return
                del ag, bg, old_groups, new_groups
                times = {'baseline': [], 'reuse': []}
                rows = []
                for step in range(1, 41):
                    loader.check()
                    order = [('baseline', baseline), ('reuse', candidate)]
                    if step % 2 == 0:
                        order.reverse()
                    step_rows = {}
                    for name, context in order:
                        torch.mps.synchronize()
                        start = time.monotonic()
                        row = context.update()
                        torch.mps.synchronize()
                        elapsed = time.monotonic()-start
                        if step > 2:
                            times[name].append(elapsed)
                        step_rows[name] = dict(row, update_seconds=elapsed)
                    assert step_rows['baseline']['batch_sha256'] == step_rows['reuse']['batch_sha256']
                    rows.append(dict(step=step, arms=step_rows))
                    if step == 1:
                        ap = dict(baseline.model.ordered_parameters())
                        bp = dict(candidate.model.ordered_parameters())
                        delta = {n: compare(bp[n].detach().cpu(), ap[n].detach().cpu(), cfg['first_update_parameter_tolerance']) for n in ap}
                        put(dest/'first-update-parameter-comparison.json', delta)
                        if not all(v['passed'] for v in delta.values()):
                            for name, context in [('baseline', baseline), ('reuse', candidate)]:
                                receipt = write_checkpoint(dest/f'first-failure-{name}.pt', context.capture())
                                put(dest/f'first-failure-{name}-receipt.json', receipt)
                            put(dest/'updates.json', rows)
                            put(dest/'decision.json', dict(passed=False, reason='first_actual_optimizer_update_gate', training_updates_per_arm=1))
                            return
                        del ap, bp
                    if step == 30:
                        restart = write_checkpoint(dest/'reuse-step-30.pt', candidate.capture())
                        put(dest/'restart-receipt.json', restart)
                    if step % 10 == 0:
                        print('Paired updates', step, flush=True)
                expected = candidate.capture()
                assert expected['snapshots'] == 2 and len(expected['ema']) == len(expected['parameters'])
                expected_hashes = tree_hashes(expected)
                put(dest/'uninterrupted-state-hashes.json', expected_hashes)
                put(dest/'updates.json', rows)
                # Late trajectory differences are reported, never called bitwise equivalence.
                final_parameter_max_delta = max(float((p.detach().cpu()-dict(candidate.model.ordered_parameters())[n].detach().cpu()).abs().max()) for n, p in baseline.model.ordered_parameters())
                del baseline, candidate
                gc.collect()
                torch.mps.empty_cache()
                restored = ReuseContext(*args)
                payload = read_checkpoint(restart)
                restored.restore(payload)
                assert tree_hashes(restored.capture()) == tree_hashes(payload)
                del payload
                for _ in range(10):
                    loader.check()
                    restored.update()
                actual = restored.capture()
                actual_hashes = tree_hashes(actual)
                put(dest/'restored-state-hashes.json', actual_hashes)
                replay = actual_hashes == expected_hashes
                if not replay:
                    for name, payload in [('uninterrupted', expected), ('restored', actual)]:
                        receipt = write_checkpoint(dest/f'failure-{name}.pt', payload)
                        put(dest/f'failure-{name}-receipt.json', receipt)
                ratio = statistics.median(times['reuse'])/statistics.median(times['baseline'])
                put(dest/'timing.json', dict(seconds=times, median_ratio=ratio, boundary='synchronized complete Context.update, including sampling, optimizer, gradient clearing, schedule, EMA and batch receipt bookkeeping', warmup_updates_per_arm=2))
                put(dest/'decision.json', dict(passed=replay and ratio <= cfg['maximum_median_update_ratio'],
                    reason='passed' if replay and ratio <= cfg['maximum_median_update_ratio'] else 'restart_or_whole_update_benefit_gate',
                    replay_bitwise=replay, state_entries=len(expected_hashes), primary_updates_per_arm=40, replay_updates=10,
                    first_actual_optimizer_update_gate_passed=True, final_parameter_max_abs_delta_descriptive=final_parameter_max_delta,
                    median_whole_update_ratio=ratio, snapshots=actual['snapshots'], quality_claim=False, test_opened=False))
                print('Replay', replay, 'median whole-update ratio', ratio, flush=True)
            except BaseException as error:
                put(dest/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), worker_sha256=sha(__file__)))
                raise
            finally:
                state = read(statepath)
                state.update(status='COMPOSER_REUSE_FULL_MODEL_REVIEW_PENDING', active_worker_pid=None, active_session=None)
                loader.replace(statepath, state)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, loader.stop)
    signal.signal(signal.SIGTERM, loader.stop)
    run()
