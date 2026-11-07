"""One new literal forward diagnostic, never a failed-panel replay or acceptance."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import torch
from research.readout_forward_sensitivity_v1 import evaluate, ARMS
from research.readout_forward_sensitivity_oracle_v1 import fixture, evaluate as oracle, CASES
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix('')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(raw) < 4*1024**2
    with (OUT/name).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def difference(actual, expected):
    x = actual.detach().cpu().double()
    y = torch.tensor(expected, dtype=torch.float64)
    assert x.shape == y.shape and bool(torch.isfinite(x).all())
    return float((x-y).abs().max())


def main(plan_sha):
    assert sha(OUT/'plan.json') == plan_sha
    plan = json.loads((OUT/'plan.json').read_text())
    for path, digest in plan['sources'].items():
        assert sha(ROOT/path) == digest, path
    assert not (OUT/'attempt.json').exists() and not (OUT/'STOP').exists()
    assert torch.get_num_threads() == 4 and torch.backends.mps.is_available()
    assert all(os.environ.get(k) == v for k, v in plan['environment'].items())
    lock = (ROOT/'research_state/tokenizer-worker-lock/.lock').open('r+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    publish('attempt.json', dict(plan_sha256=plan_sha, pid=os.getpid(), trained_checkpoint_reads=False))
    guard = StrictMpsGuard(OUT/'mps-guard.json', ROOT)
    guard.limits['timeout_seconds'] = 600
    with guard:
        cpu_rng = torch.get_rng_state().clone()
        mps_rng = torch.mps.get_rng_state().clone()
        expected = {}
        with CpuPrerequisiteGuard(OUT/'cpu-oracle-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            for name in CASES:
                case = fixture(name)
                for key in ('x', 'w', 'a', 'gamma', 'beta'):
                    value = torch.tensor(case[key], dtype=torch.float32)
                    assert value.double().tolist() == case[key], 'Every literal must be exactly FP32 representable'
                for arm in ARMS:
                    expected[name, arm] = oracle(case, arm)
            for center, shifted in [('narrow_centered', 'narrow_shifted'),
                                    ('narrow_gain16_centered', 'narrow_gain16_shifted'),
                                    ('narrow_gain64_centered', 'narrow_gain64_shifted')]:
                for arm in ARMS:
                    assert expected[center, arm] == expected[shifted, arm], 'Scalar translation-invariant control differs'
        rows = []
        for device, dtype in [('cpu', torch.float64), ('cpu', torch.float32), ('mps', torch.float32)]:
            ctx = CpuPrerequisiteGuard(OUT/f'cpu-{str(dtype).split(".")[-1]}-guard.json', ROOT,
                                      timeout_seconds=120, poll_seconds=.25) if device == 'cpu' else __import__('contextlib').nullcontext()
            with ctx:
                outputs = {}
                for name in CASES:
                    case = fixture(name)
                    values = {k: torch.tensor(case[k], device=device, dtype=dtype)
                              for k in ('x', 'w', 'a', 'gamma', 'beta')}
                    values['targets'] = torch.tensor(case['targets'], device=device, dtype=torch.int64)
                    before = {k: v.cpu().clone() for k, v in values.items()}
                    for arm in ARMS:
                        result = evaluate(**values, arm=arm, normalize=case['normalize'])
                        errors = {k: difference(v, expected[name, arm][k]) for k, v in result.items()}
                        row = dict(case=name, arm=arm, device=device, dtype=str(dtype), stage_max_absolute_errors=errors,
                                   ce=float(result['nll'].double().mean().cpu()),
                                   scalar_ce=math.fsum(expected[name, arm]['nll'])/4,
                                   diagnostic_only_not_scientific_acceptance=True)
                        rows.append(row)
                        outputs[name, arm] = {k: v.cpu().clone() for k, v in result.items()}
                        publish(f'case-{len(rows):02d}.json', row)
                    assert all(torch.equal(v.cpu(), before[k]) and v.grad is None for k, v in values.items())
                    del result, values, before
                translated = []
                for center, shifted in [('narrow_centered', 'narrow_shifted'),
                                        ('narrow_gain16_centered', 'narrow_gain16_shifted'),
                                        ('narrow_gain64_centered', 'narrow_gain64_shifted')]:
                    for arm in ARMS:
                        translated.append(dict(center=center, shifted=shifted, arm=arm,
                            stage_translation_differences={k: float((outputs[center, arm][k].double()-outputs[shifted, arm][k].double()).abs().max()) for k in outputs[center, arm]}))
                publish(f'translation-{device}-{str(dtype).split(".")[-1]}.json', translated)
                del outputs
        assert len(rows) == 81
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
        for path, digest in plan['sources'].items():
            assert sha(ROOT/path) == digest, path
        publish('result.json', dict(status='complete_new_literal_forward_sensitivity_diagnostic', rows=rows,
            cases=81, scalar_translation_controls_exact=True, caller_rng_and_inputs_preserved=True,
            numerical_errors_are_descriptive_not_acceptance_thresholds=True,
            zero_models_optimizers_checkpoints_corpus_quality_roles=True,
            failed_panel_root_cause_or_rescue=False, no_quality_or_adoption_claim=True))
    publish('STOP', dict(status='complete_generated_forward_diagnostic_only', no_replay_or_retry=True))
    print(json.dumps(dict(status='complete_new_literal_forward_sensitivity_diagnostic', cases=81)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan-sha', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    main(args.plan_sha)
