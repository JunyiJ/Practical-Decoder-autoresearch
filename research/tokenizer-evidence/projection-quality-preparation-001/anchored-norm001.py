"""One new functional forward/analytic-gradient gate; no trained-state access."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import torch
from torch.nn import functional as F
from research.anchored_layer_norm_v1 import layer_norm
from research.anchored_layer_norm_oracle_v1 import CASES, fixture, oracle
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).with_suffix('')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(raw) < 4*1024**2
    with (OUT/name).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def cpu64(value):
    # Always transfer first: no FP64 operation may be requested on MPS.
    return value.detach().cpu().to(torch.float64)


def compare(value, reference, atol, rtol):
    value = cpu64(value)
    ref = torch.tensor(reference, dtype=torch.float64)
    assert value.shape == ref.shape and bool(torch.isfinite(value).all())
    return dict(maximum_absolute_error=float((value-ref).abs().max()),
                all_elements_pass=bool(torch.allclose(value, ref, atol=atol, rtol=rtol)))


def invalids():
    x = torch.tensor(fixture('wide_signed')['x'], dtype=torch.float32, requires_grad=True)
    g = torch.ones(256, requires_grad=True)
    b = torch.zeros(256, requires_grad=True)
    cases = []
    def reject(name, args, context=None):
        tensors = [v for v in args if type(v) is torch.Tensor]
        before = [v.detach().cpu().clone() for v in tensors]
        with context if context is not None else contextlib.nullcontext():
            try:
                layer_norm(*args)
            except (ValueError, TypeError, RuntimeError):
                pass
            else:
                raise AssertionError('Invalid functional input accepted: '+name)
        assert all(torch.allclose(v.detach().cpu(), old, rtol=0, atol=0, equal_nan=True)
                   and v.grad is None for v, old in zip(tensors, before))
        cases.append(name)
    reject('x_type', [[], g, b]); reject('gamma_type', [x, [], b]); reject('beta_type', [x, g, []])
    for name, xx in [('rank', x[None].detach()), ('width255', x[:, :255].detach()),
                     ('empty_rows', x[:0].detach()), ('rows17', x[:1].detach().expand(17, -1))]:
        reject(name, [xx, g, b])
    for name, gg, bb in [('gamma_rank', g[None], b), ('beta_rank', g, b[None]),
                         ('gamma_length', g[:255], b), ('beta_length', g, b[:255])]:
        reject(name, [x, gg.detach(), bb.detach()])
    reject('integer_x', [x.detach().int(), g, b])
    reject('half_dtype', [x.detach().half(), g.detach().half(), b.detach().half()])
    reject('gamma_dtype', [x, g.detach().double(), b]); reject('beta_dtype', [x, g, b.detach().double()])
    reject('device_gamma', [x.detach().to('mps'), g, b.detach().to('mps')])
    reject('device_beta', [x, g, b.detach().to('mps')])
    for i in range(3):
        args = [v.detach().clone() for v in (x, g, b)]
        args[i].reshape(-1)[0] = float('nan') if i != 1 else float('inf')
        reject('nonfinite_'+str(i), args)
    for device in ('cpu', 'mps'):
        extreme = torch.tensor([[-3e38]+[3e38]*255], dtype=torch.float32, device=device)
        reject('anchor_overflow_'+device, [extreme, g.detach().to(device), b.detach().to(device)])
    reject('cpu_autocast', [x, g, b], torch.autocast('cpu', dtype=torch.bfloat16))
    reject('mps_autocast', [x.detach().to('mps'), g.detach().to('mps'), b.detach().to('mps')],
           torch.autocast('mps', dtype=torch.float16))
    assert len(cases) == 24
    return cases


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
    publish('attempt.json', dict(plan_sha256=plan_sha, pid=os.getpid(), no_models_or_checkpoint_reads=True))
    with StrictMpsGuard(OUT/'mps-guard.json', ROOT):
        cpu_rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
        refs = {}
        with CpuPrerequisiteGuard(OUT/'cpu-oracle-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
            for name in CASES:
                case = fixture(name)
                for key in ('x', 'gamma', 'beta', 'upstream'):
                    assert torch.tensor(case[key], dtype=torch.float32).double().tolist() == case[key]
                refs[name] = oracle(case)
            for low, high in [('narrow_low', 'narrow_high'), ('narrow_gain8_low', 'narrow_gain8_high')]:
                assert refs[low] == refs[high], 'Independent translation oracle differs'
        rows, comparisons = [], []
        for device, dtype in [('cpu', torch.float64), ('cpu', torch.float32), ('mps', torch.float32)]:
            label = device+'-'+str(dtype).split('.')[-1]
            context = CpuPrerequisiteGuard(OUT/(label+'-guard.json'), ROOT, timeout_seconds=120,
                                          poll_seconds=.25) if device == 'cpu' else contextlib.nullcontext()
            with context:
                tensors = {}
                tolerance = plan['tolerances']['cpu64' if dtype == torch.float64 else 'fp32']
                for name in CASES:
                    case = fixture(name)
                    x, g, b = [torch.tensor(case[k], device=device, dtype=dtype, requires_grad=True)
                               for k in ('x', 'gamma', 'beta')]
                    upstream = torch.tensor(case['upstream'], device=device, dtype=dtype)
                    before = [v.detach().cpu().clone() for v in (x, g, b, upstream)]
                    for kind in ('original', 'anchored'):
                        y = F.layer_norm(x, (256,), g, b, 1e-5) if kind == 'original' else layer_norm(x, g, b)
                        gradients = torch.autograd.grad((y*upstream).sum(), (x, g, b))
                        errors = {'output': compare(y, refs[name]['output'], tolerance['forward_atol'], tolerance['rtol'])}
                        errors.update({k: compare(v, refs[name][k], tolerance['gradient_atol'], tolerance['rtol'])
                                       for k, v in zip(('gx', 'ggamma', 'gbeta'), gradients)})
                        row = dict(case=name, kind=kind, device=device, dtype=str(dtype), comparisons=errors,
                                   anchored_gate_required=kind == 'anchored', original_descriptive_only=kind == 'original')
                        rows.append(row)
                        publish(f'case-{len(rows):02d}.json', row)
                        if kind == 'anchored':
                            assert all(v['all_elements_pass'] for v in errors.values()), 'Anchor independent forward/gradient gate failed'
                        tensors[name, kind] = [cpu64(v) for v in (y, *gradients)]
                    assert all(torch.equal(v.detach().cpu(), old) for v, old in zip((x, g, b, upstream), before))
                    assert x.grad is g.grad is b.grad is None
                for low, high in [('narrow_low', 'narrow_high'), ('narrow_gain8_low', 'narrow_gain8_high')]:
                    for kind in ('original', 'anchored'):
                        differences = [float((a-b).abs().max()) for a, b in zip(tensors[low, kind], tensors[high, kind])]
                        if kind == 'anchored':
                            assert differences == [0.]*4, 'Identical anchor inputs must retain exact pair forward/gradients'
                        comparisons.append(dict(backend=label, kind=kind, low=low, high=high,
                                                forward_gx_ggamma_gbeta_pair_max=differences))
                del tensors
        invalid = invalids()
        publish('invalids.json', dict(cases=invalid, count=len(invalid)))
        assert len(rows) == 66
        assert torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state())
        for name, sha in plan['sources'].items():
            assert digest(ROOT/name) == sha, name
        publish('result.json', dict(status='accepted_bounded_functional_anchor_forward_backward', cases=66,
            anchored_cases=33, original_descriptive_cases=33, invalids=24, rows=rows, translation_pairs=comparisons,
            global_cpu_mps_rng_inputs_grad_slots_preserved=True, source_bindings=len(plan['sources']),
            no_models_optimizers_checkpoints_corpus_quality_roles=True,
            actual_model_integration_or_failed_quality_root_cause=False, adoption=False))
    publish('STOP', dict(status='complete_new_functional_anchor_gate', no_retry_or_replay=True))
    print(json.dumps(dict(status='accepted_bounded_functional_anchor_forward_backward', cases=66, invalids=24)))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--plan-sha', required=True)
    args = p.parse_args()
    torch.set_num_threads(4)
    main(args.plan_sha)
