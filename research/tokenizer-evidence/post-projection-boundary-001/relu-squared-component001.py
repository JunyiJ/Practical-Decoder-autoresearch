"""One finite new literal FFN forward/analytic-backward gate; no trained state."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch
from research.relu_squared_component_v1 import forward, KINDS
from research.relu_squared_oracle_v1 import CASES, fixture, oracle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix('')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(raw) < 2*1024**2
    with (OUT/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def cpu64(v):
    return v.detach().cpu().to(torch.float64)


def compare(v, reference, atol, rtol):
    value, ref = cpu64(v), torch.tensor(reference, dtype=torch.float64)
    assert value.shape == ref.shape and bool(torch.isfinite(value).all())
    return dict(maximum_absolute_error=float((value-ref).abs().max()),
                all_elements_pass=bool(torch.allclose(value, ref, atol=atol, rtol=rtol)))


def invalids():
    names = []
    for kind in KINDS:
        c = fixture('mixed_d8', kind)
        base = [kind] + [torch.tensor(c[k], dtype=torch.float32) for k in ('x', 'up', 'down')]
        base.append(torch.tensor(c['gate'], dtype=torch.float32) if c['gate'] else None)
        def reject(name, changes, context=None):
            args = [v.detach().clone().requires_grad_(v.is_floating_point()) if type(v) is torch.Tensor else v
                    for v in base]
            for index, value in changes.items():
                args[index] = value.detach().clone().requires_grad_(value.is_floating_point()) if type(value) is torch.Tensor else value
            tensors = [v for v in args if type(v) is torch.Tensor]
            before = [v.detach().cpu().clone() for v in tensors]
            for v in tensors:
                if v.requires_grad:
                    v.grad = torch.full_like(v, .375)
            grads = [v.grad.detach().cpu().clone() if v.grad is not None else None for v in tensors]
            with context if context is not None else contextlib.nullcontext():
                try:
                    forward(*args)
                except ValueError:
                    pass
                else:
                    raise AssertionError('Invalid FFN accepted: '+name)
            assert all(torch.allclose(v.detach().cpu(), old, atol=0, rtol=0, equal_nan=True)
                       for v, old in zip(tensors, before))
            assert all((v.grad is None if old is None else torch.equal(v.grad.cpu(), old))
                       for v, old in zip(tensors, grads))
            names.append(kind+':'+name)
        reject('kind_unknown', {0: 'relu_squared_gate'})
        reject('kind_type', {0: True})
        for i, label in ((1, 'x'), (2, 'up'), (3, 'down')):
            reject(label+'_type', {i: []})
        x = base[1]
        for name, v in [('rank', x[None]), ('empty', x[:0]), ('rows17', x[:1].expand(17, -1)),
                        ('d1', x[:, :1]), ('odd3', x[:, :3]), ('d10', torch.zeros(3, 10))]:
            reject('x_'+name, {1: v})
        reject('up_shape', {2: base[2][:, :7]}); reject('down_shape', {3: base[3][:7]})
        reject('integer', {1: x.int()})
        reject('half', {i: v.half() for i, v in enumerate(base) if type(v) is torch.Tensor})
        for i, label in ((2, 'up'), (3, 'down')):
            reject('dtype_'+label, {i: base[i].double()})
            reject('device_'+label, {i: base[i].to('mps')})
        for i, label in ((1, 'x'), (2, 'up'), (3, 'down')):
            v = base[i].clone(); v.reshape(-1)[0] = float('nan') if i == 1 else float('inf')
            reject('nonfinite_'+label, {i: v})
        reject('cpu_autocast', {}, torch.autocast('cpu', dtype=torch.bfloat16))
        reject('mps_autocast', {i: v.to('mps') for i, v in enumerate(base) if type(v) is torch.Tensor},
               torch.autocast('mps', dtype=torch.float16))
        reject('overflow_up', {1: torch.full_like(x, 1e20), 2: torch.full_like(base[2], 1e20)})
        reject('overflow_down', {1: torch.ones_like(x), 2: torch.ones_like(base[2]),
                                3: torch.full_like(base[3], 1e38),
                                **({4: torch.ones_like(base[4])} if kind == 'native_swiglu' else {})})
        if kind == 'native_swiglu':
            reject('gate_missing', {4: None}); reject('gate_type', {4: []})
            reject('gate_shape', {4: base[4][:, :7]}); reject('gate_dtype', {4: base[4].double()})
            reject('gate_device', {4: base[4].to('mps')})
            v = base[4].clone(); v[0, 0] = float('-inf'); reject('gate_nonfinite', {4: v})
            reject('gate_overflow', {1: torch.full_like(x, 1e20), 4: torch.full_like(base[4], 1e20)})
            reject('hidden_overflow', {1: torch.ones_like(x), 2: torch.full_like(base[2], 1e20),
                                       4: torch.full_like(base[4], 1e20)})
        else:
            reject('foreign_gate', {4: torch.ones(12, 8)})
            if kind == 'ungated_relu_squared':
                reject('hidden_overflow', {1: torch.ones_like(x), 2: torch.full_like(base[2], 1e20)})
    assert len(names) == 89
    return names


def main(plan_sha):
    assert digest(OUT/'plan.json') == plan_sha
    plan = json.loads((OUT/'plan.json').read_text())
    for name, sha in plan['sources'].items():
        assert digest(ROOT/name) == sha, name
    assert all(os.environ.get(k) == v for k, v in plan['environment'].items())
    assert torch.get_num_threads() == 4 and torch.backends.mps.is_available()
    assert not (OUT/'attempt.json').exists() and not (OUT/'STOP').exists()
    lock = (ROOT/'research_state/tokenizer-worker-lock/.lock').open('r+')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    publish('attempt.json', dict(plan_sha256=plan_sha, pid=os.getpid(), literal_functional_only=True))
    try:
        with StrictMpsGuard(OUT/'mps-guard.json', ROOT):
            cpu_rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
            refs = {}
            with CpuPrerequisiteGuard(OUT/'cpu-oracle-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                for name in CASES:
                    for kind in KINDS:
                        c = fixture(name, kind)
                        for value in c.values():
                            if value is not None:
                                assert torch.tensor(value, dtype=torch.float32).double().tolist() == value
                        refs[name, kind] = oracle(c, kind)
            rows = []
            for device, dtype in [('cpu', torch.float64), ('cpu', torch.float32), ('mps', torch.float32)]:
                label = device+'-'+str(dtype).split('.')[-1]
                context = CpuPrerequisiteGuard(OUT/(label+'-guard.json'), ROOT, timeout_seconds=120,
                                              poll_seconds=.25) if device == 'cpu' else contextlib.nullcontext()
                with context:
                    tol = plan['tolerances']['cpu64' if dtype == torch.float64 else 'fp32']
                    for name in CASES:
                        for kind in KINDS:
                            c = fixture(name, kind)
                            keys = ['x', 'up', 'down'] + (['gate'] if kind == 'native_swiglu' else [])
                            values = [torch.tensor(c[k], device=device, dtype=dtype, requires_grad=True) for k in keys]
                            upstream = torch.tensor(c['upstream'], device=device, dtype=dtype)
                            before = [v.detach().cpu().clone() for v in (*values, upstream)]
                            for v in values:
                                v.grad = torch.full_like(v, -.625)
                            y = forward(kind, *values)
                            gradients = torch.autograd.grad((y*upstream).sum(), values)
                            checks = {'output': compare(y, refs[name, kind]['output'], tol['forward_atol'], tol['rtol'])}
                            checks.update({('g'+k): compare(v, refs[name, kind]['g'+k], tol['gradient_atol'], tol['rtol'])
                                           for k, v in zip(keys, gradients)})
                            count = sum(v.numel() for v in values[1:])
                            assert count == 3*len(c['x'][0])**2
                            row = dict(case=name, kind=kind, device=device, dtype=str(dtype), matrix_parameters=count,
                                       comparisons=checks)
                            rows.append(row); publish(f'case-{len(rows):02d}.json', row)
                            assert all(v['all_elements_pass'] for v in checks.values()), 'Independent scalar FFN gate failed'
                            assert all(torch.equal(v.detach().cpu(), old) for v, old in zip((*values, upstream), before))
                            assert all(torch.equal(v.grad, torch.full_like(v, -.625)) for v in values)
                            assert upstream.grad is None
            invalid = invalids(); publish('invalids.json', dict(cases=invalid, count=len(invalid)))
            assert len(rows) == 99
            assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
            for name, sha in plan['sources'].items():
                assert digest(ROOT/name) == sha, name
            publish('result.json', dict(status='accepted_tiny_three_equation_forward_analytic_backward', cases=99,
                invalids=len(invalid), rows=rows, source_bindings=len(plan['sources']),
                input_grad_slots_global_cpu_mps_rng_preserved=True,
                no_models_optimizers_checkpoints_corpus_rng_draws_quality_roles=True,
                actual_model_integration_or_quality_admission=False, adoption=False))
    except BaseException as exc:
        publish('STOP', dict(status='failed_closed', error=repr(exc), no_retry_or_replay=True))
        raise
    publish('STOP', dict(status='complete_new_tiny_ffn_gate', no_retry_or_replay=True))
    print(json.dumps(dict(status='accepted_tiny_three_equation_forward_analytic_backward', cases=99, invalids=len(invalid))))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--plan-sha', required=True)
    args = p.parse_args(); torch.set_num_threads(4); main(args.plan_sha)
