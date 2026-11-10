"""Frozen component correctness/repeatability/timing qualification, no training."""
import fcntl
import hashlib
import json
import os
import shutil
import signal
import statistics
import time
import traceback
import numpy as np
import torch
from research.byte_composer_v2 import ByteComposer
from research.general_rsi_composer_reuse_v1 import ExpandUnique, compose
from research.general_rsi_data_v1 import GeneralData
from research.general_rsi_transfer_data_v1 import ROOT, read, put, sha, contract
from research import general_rsi_transfer_worker_v1 as loader
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

OUT = ROOT/'research/tokenizer-evidence/general-rsi-composer-reuse-001'


def digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def compare(actual, expected, cfg):
    delta = (actual.double()-expected.double()).abs()
    limit = cfg['atol'] + cfg['rtol']*expected.double().abs()
    return dict(passed=bool(torch.isfinite(actual).all() and (delta <= limit).all()),
                max_abs_delta=float(delta.max()), max_tolerance_ratio=float((delta/limit).max()),
                actual_sha256=digest(actual), expected_sha256=digest(expected))


def batches():
    cases = [('all_unique', torch.arange(1024)), ('all_repeated', torch.full((1024,), 50256)),
             ('alternating', torch.tensor([0, 50256]*512))]
    prior = read(ROOT/'research/tokenizer-evidence/general-rsi-equal-time-001/plan.json')
    census = read(ROOT/'research/tokenizer-evidence/general-rsi-composer-efficiency-preparation-001/batch-reuse.json')
    for slot in [4, 6, 8]:
        roles = prior['slots'][slot]['roles']
        data = GeneralData(roles['train_sampler'], roles['validation_sampler'], 'cpu')
        x, y = data.batch()
        expected = next(r for r in census['records'] if r['bundle'] == prior['slots'][slot]['bundle'] and r['step'] == 1)
        assert hashlib.sha256(x.numpy().tobytes()+y.numpy().tobytes()).hexdigest() == expected['batch_sha256']
        cases.append((f'historical_slot_{slot:02d}_step_1', x.reshape(-1)))
    return cases


def run():
    cfg = read(OUT/'plan.json')
    for path, h in cfg['sources'].items():
        assert sha(ROOT/path) == h, path
    remaining = cfg['total_guard_seconds'] - sum(read(p)['elapsed_seconds'] for p in OUT.glob('guard-*.json'))
    assert remaining > 0
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert shutil.disk_usage(ROOT).free > cfg['reserve_bytes'] + cfg['retained_output_cap_bytes']
        with CpuPrerequisiteGuard(OUT/'guard-01.json', ROOT, timeout_seconds=remaining,
                                 rss_limit_bytes=4*1024**3, reserve_bytes=cfg['reserve_bytes']):
            dest = OUT/'attempt-01'
            dest.mkdir()
            statepath = ROOT/'research_state/general-rsi-state.json'
            state = read(statepath)
            state.update(status='COMPOSER_REUSE_COMPONENT_RUNNING', active_worker_pid=os.getpid(), active_session=None)
            loader.replace(statepath, state)
            try:
                protocol = contract()
                loader.validate_sources(protocol)
                book = loader.setup(protocol)
                torch.use_deterministic_algorithms(True)
                codes = torch.from_numpy(np.frombuffer(book.codes, dtype='<u2').astype('i8').reshape(50257, 130)).to('mps')
                lengths = torch.from_numpy(np.frombuffer(book.lengths, dtype='<u2').astype('i8')).to('mps')
                model = ByteComposer(cfg['initializer_seed']).to('mps')
                initial = {name: digest(p) for name, p in model.named_parameters()}
                cpu_rng = torch.get_rng_state().clone()
                mps_rng = torch.mps.get_rng_state().clone()
                generator = torch.Generator(device='cpu').manual_seed(cfg['upstream_seed'])
                upstream = (torch.randn((1024, 256), generator=generator)/512).to('mps')
                records = []

                def execute(ids, reuse, capture=True):
                    loader.check()
                    model.zero_grad(set_to_none=True)
                    torch.mps.synchronize()
                    start = time.monotonic()
                    values = compose(model, codes, lengths, ids, reuse)
                    values.backward(upstream)
                    torch.mps.synchronize()
                    seconds = time.monotonic()-start
                    if not capture:
                        return seconds
                    return seconds, values.detach().cpu(), {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()}

                all_cases = [(name, ids.to('mps')) for name, ids in batches()]
                for name, ids in all_cases:
                    loader.check()
                    unique, inverse = torch.unique(ids, sorted=True, return_inverse=True)
                    leaf = torch.zeros((len(unique), 256), device='mps', requires_grad=True)
                    ExpandUnique.apply(leaf, inverse).backward(upstream)
                    oracle = torch.zeros((len(unique), 256), dtype=torch.float64)
                    for index, row in zip(inverse.cpu().tolist(), upstream.cpu().double(), strict=True):
                        oracle[index] += row
                    aggregation = compare(leaf.grad.cpu(), oracle, cfg['oracle_tolerance'])
                    first = {}
                    repeat = {}
                    for reuse in [False, True]:
                        for rep in range(cfg['correctness_repeats']):
                            seconds, values, grads = execute(ids, reuse)
                            hashes = [digest(values)] + [digest(v) for v in grads.values()]
                            if rep == 0:
                                first[reuse] = (values, grads, hashes)
                                repeat[reuse] = True
                            else:
                                repeat[reuse] &= hashes == first[reuse][2]
                    a, ag, _ = first[False]
                    b, bg, _ = first[True]
                    output = compare(b, a, cfg['numerical_tolerance'])
                    gradients = {n: compare(bg[n], ag[n], cfg['numerical_tolerance']) for n in ag}
                    immutable = initial == {n: digest(p) for n, p in model.named_parameters()}
                    rng_unchanged = torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
                    passed = aggregation['passed'] and output['passed'] and all(v['passed'] for v in gradients.values()) and all(repeat.values()) and immutable and rng_unchanged
                    record = dict(case=name, positions=1024, unique_ids=len(unique), aggregation=aggregation, output=output,
                                  gradients=gradients, repeat_bitwise={str(k): v for k, v in repeat.items()},
                                  parameter_immutable=immutable, global_rng_unchanged=rng_unchanged, passed=passed)
                    put(dest/f'{name}-correctness.json', record)
                    records.append(record)
                    print(name, 'correctness', passed, flush=True)
                    if not passed:
                        with (dest/'first-failure-tensors.npz').open('xb') as stream:
                            np.savez(stream, baseline_output=a.numpy(), reuse_output=b.numpy(),
                                     **{f'baseline_{n}': v.numpy() for n, v in ag.items()},
                                     **{f'reuse_{n}': v.numpy() for n, v in bg.items()})
                        put(dest/'decision.json', dict(passed=False, reason='component_correctness_or_repeatability_gate',
                            failed_case=name, timing_performed=False, production_changed=False, quality_claim=False))
                        return
                timings = []
                for name, ids in all_cases:
                    for reuse in [False, True]*cfg['warmup_pairs']:
                        execute(ids, reuse, False)
                    times = {False: [], True: []}
                    for reuse in [False, True, True, False]*cfg['timing_abba_blocks']:
                        times[reuse].append(execute(ids, reuse, False))
                    ratio = statistics.median(times[True])/statistics.median(times[False])
                    row = dict(case=name, baseline_seconds=times[False], reuse_seconds=times[True], median_ratio=ratio)
                    timings.append(row)
                    put(dest/f'{name}-timing.json', row)
                    print(name, 'median reuse/baseline', ratio, flush=True)
                real = [r['median_ratio'] for r in timings if r['case'].startswith('historical')]
                benefit = statistics.median(real) <= cfg['maximum_real_median_ratio'] and max(real) < 1 and timings[0]['median_ratio'] <= cfg['maximum_unique_ratio']
                put(dest/'decision.json', dict(passed=benefit, reason='passed' if benefit else 'practical_benefit_gate',
                    correctness_passed=True, median_real_ratio=statistics.median(real),
                    all_unique_ratio=timings[0]['median_ratio'], production_changed=False, quality_claim=False))
            except BaseException as error:
                put(dest/'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), worker_sha256=sha(__file__)))
                raise
            finally:
                state = read(statepath)
                state.update(status='COMPOSER_REUSE_COMPONENT_REVIEW_PENDING', active_worker_pid=None, active_session=None)
                loader.replace(statepath, state)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, loader.stop)
    signal.signal(signal.SIGTERM, loader.stop)
    run()
