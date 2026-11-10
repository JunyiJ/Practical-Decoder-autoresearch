"""Fixed-gradient Nesterov/state qualification of composer-only FP32 Muon."""
import copy
import fcntl
import json
import os
import shutil
import time
import traceback
import torch
from research.composer_fp32_muon_v1 import ComposerFP32Muon, SHAPES, orthogonalize
from research.general_rsi_transfer_data_v1 import ROOT, read, put, sha
from research.general_rsi_training_v2 import write_checkpoint, read_checkpoint
from research.general_rsi_reuse_qualification_v1 import tree_hashes
from research.general_rsi_composer_reuse_worker_v1 import compare
from research import general_rsi_transfer_worker_v1 as checks
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

OUT = ROOT/'research/tokenizer-evidence/general-rsi-composer-fp32-fixture-001'


def inputs(cfg):
    weights, gradients = {}, {}
    pg = torch.Generator().manual_seed(cfg['parameter_seed'])
    gg = torch.Generator().manual_seed(cfg['gradient_seed'])
    for name, shape in SHAPES.items():
        weights[name] = torch.empty(shape).uniform_(-.05, .05, generator=pg)
        rank = torch.randn((shape[0], 1), generator=gg) @ torch.randn((1, shape[1]), generator=gg)
        rank *= .01/rank.norm()
        dense = torch.randn(shape, generator=gg)
        dense *= .01/dense.norm()
        sparse = torch.zeros(shape)
        ix = torch.arange(min(shape))
        sparse[ix, ix] = .01/(min(shape)**.5)
        gradients[name] = [torch.zeros(shape), rank, dense, -dense, sparse, .5*(rank+dense)]
    return weights, gradients


def make(weights):
    return ComposerFP32Muon([(n, torch.nn.Parameter(v.to('mps').clone())) for n, v in weights.items()])


def reference_step(parameters, buffers, gradients, step, lr):
    directions = {}
    for name, p in parameters.items():
        g = gradients[name][step].double()
        b = .95*buffers[name]+(1-.95)*g
        x = (1-.95)*g+.95*b
        transpose = x.shape[0] > x.shape[1]
        if transpose:
            x = x.T
        x = x/x.norm().clamp(min=1e-7)
        for _ in range(5):
            gram = x @ x.T
            x = 3.4445*x - 4.775*(gram@x) + 2.0315*((gram@gram)@x)
        direction = x.T if transpose else x
        parameters[name] = p-lr*.2*(max(p.shape)**.5)*direction
        buffers[name] = b
        directions[name] = direction
    return directions


def advance(opt, gradients, step, lr):
    checks.check()
    for (name, p), group in zip(opt.items, opt.param_groups, strict=True):
        p.grad = gradients[name][step].to('mps').clone()
        group['lr'] = lr
    opt.step()


def main():
    cfg = read(OUT/'plan.json')
    for path, h in cfg['sources'].items():
        assert sha(ROOT/path) == h
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '0'
    assert torch.__version__ == '2.14.1' and torch.backends.mps.is_available()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checks.check()
        assert shutil.disk_usage(ROOT).free > 10*1024**3+64*1024**2
        with CpuPrerequisiteGuard(OUT/'guard-01.json', ROOT, timeout_seconds=240,
                                 rss_limit_bytes=4*1024**3, reserve_bytes=10*1024**3):
            dest = OUT/'attempt-01'
            dest.mkdir()
            try:
                start = time.monotonic()
                weights, gradients = inputs(cfg)
                cpu_rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
                initial_inputs = tree_hashes(dict(weights=weights, gradients=gradients))
                opt = make(weights)
                params = {n: p.double().clone() for n, p in weights.items()}
                buffers = {n: torch.zeros_like(p) for n, p in params.items()}
                records = []
                for step, lr in enumerate(cfg['lrs']):
                    directions = reference_step(params, buffers, gradients, step, lr)
                    advance(opt, gradients, step, lr)
                    per_matrix = {}
                    for name, p in opt.items:
                        pc = compare(p.detach().cpu(), params[name], cfg['parameter_tolerance'])
                        bc = compare(opt.buffers[name].cpu(), buffers[name], cfg['momentum_tolerance'])
                        d = orthogonalize(p.grad.lerp(opt.buffers[name], .95)).cpu().double()
                        error = float((d-directions[name]).norm())
                        denominator = float(directions[name].norm())
                        relative = error/denominator if denominator else error
                        passed = pc['passed'] and bc['passed'] and relative <= cfg['direction_relative_tolerance']
                        per_matrix[name] = dict(parameter=pc, momentum=bc, direction_relative_error=relative, passed=passed)
                        if not passed:
                            receipt = write_checkpoint(dest/'failure-matrix.pt', dict(name=name, step=step+1,
                                parameter=p.detach().cpu(), reference_parameter=params[name],
                                momentum=opt.buffers[name].cpu(), reference_momentum=buffers[name]), cap=16*1024**2)
                            put(dest/'failure-matrix-receipt.json', receipt)
                            put(dest/'failed-comparison.json', dict(completed=records, current=per_matrix))
                            put(dest/'decision.json', dict(passed=False, reason='fixed_gradient_reference_gate', name=name, step=step+1))
                            return
                    records.append(dict(step=step+1, lr=lr, matrices=per_matrix))
                    put(dest/f'step-{step+1:02d}.json', records[-1])
                    if step == 2:
                        receipt = write_checkpoint(dest/'step-03.pt', opt.snapshot(), cap=16*1024**2)
                        put(dest/'step-03-receipt.json', receipt)
                    print('Reference step', step+1, 'passed', flush=True)
                expected = opt.snapshot()
                put(dest/'final-receipt.json', write_checkpoint(dest/'final.pt', expected, cap=16*1024**2))
                put(dest/'reference-receipt.json', write_checkpoint(dest/'reference.pt', dict(parameters=params, momentum=buffers), cap=16*1024**2))
                # Full independent rerun, then serialized restart at the midpoint.
                repeated = make(weights)
                for step, lr in enumerate(cfg['lrs']):
                    advance(repeated, gradients, step, lr)
                repeat_equal = tree_hashes(repeated.snapshot()) == tree_hashes(expected)
                restored = make(weights)
                saved = read_checkpoint(receipt)
                restored.restore(saved)
                immediate_equal = tree_hashes(restored.snapshot()) == tree_hashes(saved)
                for step in range(3, 6):
                    advance(restored, gradients, step, cfg['lrs'][step])
                restart_equal = tree_hashes(restored.snapshot()) == tree_hashes(expected)
                # One invalid last gradient must not partially update earlier parameters/state.
                bad = make(weights)
                for name, p in bad.items:
                    p.grad = gradients[name][1].to('mps').clone()
                bad.items[-1][1].grad[0, 0] = float('nan')
                before = tree_hashes(bad.snapshot())
                try:
                    bad.step()
                    rejected_gradient = False
                except AssertionError:
                    rejected_gradient = tree_hashes(bad.snapshot()) == before
                damaged = copy.deepcopy(saved)
                damaged['momentum'][list(SHAPES)[-1]][0, 0] = float('nan')
                fresh = make(weights)
                before = tree_hashes(fresh.snapshot())
                try:
                    fresh.restore(damaged)
                    rejected_restore = False
                except AssertionError:
                    rejected_restore = tree_hashes(fresh.snapshot()) == before
                inputs_unchanged = initial_inputs == tree_hashes(dict(weights=weights, gradients=gradients))
                rng_unchanged = torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
                gates = dict(repeat_bitwise=repeat_equal, immediate_restore_bitwise=immediate_equal,
                    restarted_bitwise=restart_equal, invalid_gradient_atomic=rejected_gradient,
                    invalid_restore_atomic=rejected_restore, inputs_unchanged=inputs_unchanged, rng_unchanged=rng_unchanged)
                put(dest/'decision.json', dict(passed=all(gates.values()), gates=gates,
                    max_direction_relative_error=max(v['direction_relative_error'] for r in records for v in r['matrices'].values()),
                    max_parameter_abs_delta=max(v['parameter']['max_abs_delta'] for r in records for v in r['matrices'].values()),
                    elapsed_seconds=time.monotonic()-start, test_opened=False, full_model_updates=0))
            except BaseException as error:
                put(dest/'failure.json', dict(error=repr(error), traceback=traceback.format_exc()))
                raise


if __name__ == '__main__':
    main()
