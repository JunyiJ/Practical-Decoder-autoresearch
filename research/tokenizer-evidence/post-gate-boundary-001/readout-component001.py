"""Finite literal CPU/MPS scalar forward/backward proof; no model or optimizer."""
import argparse
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import torch
from research.readout_mixture_v1.component import KINDS, log_probs, nll
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
R = E / 'readout-component001'
META = 2 * 1024**2


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= META,
            'Complete metadata cap exceeded')
    with (R/name).open('xb') as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())


def lse(values):
    m = max(values)
    return m + math.log(math.fsum(math.exp(x-m) for x in values))


def lsoft(values):
    c = lse(values)
    return [x-c for x in values]


def dot(x, y):
    return math.fsum(a*b for a, b in zip(x, y))


def oracle(kind, z, w, a, r, targets):
    # All arithmetic below is independent Python scalar calculus, not torch/autograd.
    n, d, v = len(z), len(z[0]), len(w)
    gz = [[0.]*d for _ in z]
    gw = [[0.]*d for _ in w]
    ga = [[[0.]*d for _ in range(d)] for _ in range(2)]
    gr = [[0.]*d for _ in range(2)]
    losses, distributions = [], []
    for i, x in enumerate(z):
        if kind == 'native':
            lp = lsoft([dot(row, x)/4 for row in w])
            losses.append(-lp[targets[i]])
            distributions.append(lp)
            err = [(math.exp(lp[j])-(j == targets[i]))/n for j in range(v)]
            for j in range(v):
                for q in range(d):
                    gw[j][q] += err[j]*x[q]/4
                    gz[i][q] += err[j]*w[j][q]/4
            continue
        h = [[math.tanh(dot(row, x)) for row in ak] for ak in a]
        gate = lsoft([dot(row, x) for row in r])
        pi = [math.exp(t) for t in gate]
        gh = [[0.]*d for _ in range(2)]
        if kind == 'context_mix2':
            bar = [math.fsum(pi[k]*h[k][q] for k in range(2)) for q in range(d)]
            lp = lsoft([dot(row, bar)/4 for row in w])
            losses.append(-lp[targets[i]])
            distributions.append(lp)
            err = [(math.exp(lp[j])-(j == targets[i]))/n for j in range(v)]
            dx = [math.fsum(err[j]*w[j][q]/4 for j in range(v)) for q in range(d)]
            gg = [pi[k]*dot(dx, [h[k][q]-bar[q] for q in range(d)]) for k in range(2)]
            for k in range(2):
                gh[k] = [pi[k]*t for t in dx]
            for j in range(v):
                for q in range(d):
                    gw[j][q] += err[j]*bar[q]/4
        else:
            lp = [lsoft([dot(row, hk)/4 for row in w]) for hk in h]
            total = lse([gate[k]+lp[k][targets[i]] for k in range(2)])
            responsibility = [math.exp(gate[k]+lp[k][targets[i]]-total) for k in range(2)]
            losses.append(-total)
            distributions.append([lse([gate[k]+lp[k][j] for k in range(2)]) for j in range(v)])
            gg = [(pi[k]-responsibility[k])/n for k in range(2)]
            for k in range(2):
                err = [responsibility[k]*(math.exp(lp[k][j])-(j == targets[i]))/n for j in range(v)]
                gh[k] = [math.fsum(err[j]*w[j][q]/4 for j in range(v)) for q in range(d)]
                for j in range(v):
                    for q in range(d):
                        gw[j][q] += err[j]*h[k][q]/4
        for k in range(2):
            for p in range(d):
                du = gh[k][p]*(1-h[k][p]**2)
                for q in range(d):
                    ga[k][p][q] += du*x[q]
                    gz[i][q] += du*a[k][p][q]
            for q in range(d):
                gr[k][q] += gg[k]*x[q]
                gz[i][q] += gg[k]*r[k][q]
    return dict(loss=math.fsum(losses)/n, nll=losses, logp=distributions, grads=[gz, gw, ga, gr])


def fixture(name):
    z = [[.2, -.4, .7], [-.6, .1, .3], [.4, .5, -.2]]
    w = [[.3, -.2, .1], [-.4, .5, .2], [.1, .2, -.6], [.7, -.1, .4], [-.2, -.3, .8]]
    a = [[[.8, .1, -.2], [.3, -.7, .2], [-.1, .4, .6]],
         [[-.4, .2, .7], [.6, .5, -.1], [.2, -.3, .8]]]
    r = [[.2, -.1, .4], [-.3, .5, -.2]]
    targets = [0, 3, 4]
    if name == 'zero_router':
        r = [[0.]*3 for _ in range(2)]
    elif name == 'sharp_router':
        r = [[t*200 for t in row] for row in r]
    elif name == 'saturated_context':
        a = [[[t*30 for t in row] for row in ak] for ak in a]
    elif name == 'sharp_vocabulary':
        w = [[t*80 for t in row] for row in w]
    elif name == 'identical_contexts':
        a[1] = copy.deepcopy(a[0])
    elif name == 'one_dimensional':
        z, w, a, r, targets = [[.7]], [[.3], [-.8]], [[[.4]], [[-.6]]], [[.2], [-.5]], [1]
    elif name != 'ordinary':
        raise ValueError('Unknown literal fixture')
    return z, w, a, r, targets


def compare(actual, expected, atol, rtol):
    x = actual.detach().cpu().double()
    y = torch.tensor(expected, dtype=torch.float64)
    require(x.shape == y.shape and torch.allclose(x, y, atol=atol, rtol=rtol),
            'Independent scalar forward/backward oracle differs')
    return float((x-y).abs().max())


def run_backend(device, dtype, tolerances):
    rows, discriminated = [], False
    for name in ('ordinary', 'zero_router', 'sharp_router', 'saturated_context',
                 'sharp_vocabulary', 'identical_contexts', 'one_dimensional'):
        vals = fixture(name)
        args = [torch.tensor(x, dtype=dtype, device=device, requires_grad=True) for x in vals[:4]]
        z, w, a, r = args
        y = torch.tensor(vals[4], dtype=torch.int64, device=device)
        before = [t.detach().cpu().clone() for t in args]
        scalar_inputs = [t.detach().cpu().double().tolist() for t in args]
        atol = tolerances['CPU64_forward_atol'] if dtype == torch.float64 else tolerances['FP32_forward_atol']
        gatol = tolerances['CPU64_gradient_atol'] if dtype == torch.float64 else tolerances['FP32_gradient_atol']
        rtol = 0. if dtype == torch.float64 else tolerances['FP32_rtol']
        arm_losses = {}
        for kind in KINDS:
            extra = {} if kind == 'native' else dict(a=a, r=r)
            expected = oracle(kind, *scalar_inputs, vals[4])
            loss = nll(kind, z, w, y, **extra, reduction='mean')
            gradients = torch.autograd.grad(loss, args, allow_unused=True)
            lp = log_probs(kind, z, w, **extra)
            full_loss = -lp.gather(1, y[:, None]).mean()
            full_gradients = torch.autograd.grad(full_loss, args, allow_unused=True)
            errors = [compare(loss, expected['loss'], atol, rtol), compare(lp, expected['logp'], atol, rtol),
                      compare(nll(kind, z, w, y, **extra), expected['nll'], atol, rtol)]
            grad_errors = []
            for actual, full, inp, wanted in zip(gradients, full_gradients, args, expected['grads']):
                if kind == 'native' and inp is not z and inp is not w:
                    require(actual is None and full is None, 'Native must not use extra tensors')
                    continue
                require(actual is not None and full is not None, 'All mixture tensors must receive a gradient')
                grad_errors += [compare(actual, wanted, gatol, rtol), compare(full, wanted, gatol, rtol)]
            compare(lp.exp().sum(-1), [1.]*len(y), atol, rtol)
            compare(full_loss, expected['loss'], atol, rtol)
            if kind != 'native':
                swapped = log_probs(kind, z, w, a.flip(0), r.flip(0))
                compare(swapped, expected['logp'], atol, rtol)
                if name == 'ordinary':
                    require(all(bool((t != 0).any()) for t in gradients), 'Literal asymmetric mixture must expose all gradients')
            arm_losses[kind] = float(loss.detach().cpu())
            rows.append(dict(device=device, dtype=str(dtype), case=name, kind=kind,
                             forward_max_error=max(errors), gradient_max_error=max(grad_errors),
                             normalized=True, target_full_gradient_agreement=True))
        if name == 'identical_contexts':
            require(abs(arm_losses['context_mix2']-arm_losses['probability_mix2']) <= atol,
                    'Identical components must agree')
        if name == 'ordinary':
            discriminated = abs(arm_losses['context_mix2']-arm_losses['probability_mix2']) > 1e-7
        require(all(torch.equal(t.detach().cpu(), v) and t.grad is None for t, v in zip(args, before)),
                'Inputs or caller gradient slots changed')
    require(discriminated, 'Probability mixing must be distinguished from feature mixing')
    return rows


def invalid_cases():
    z, w, a, r, targets = fixture('ordinary')
    base = [torch.tensor(x, dtype=torch.float32) for x in (z, w, a, r)]
    target = torch.tensor(targets, dtype=torch.int64)
    tests = []
    for index in range(4):
        args = [t.clone() for t in base]
        args[index].reshape(-1)[0] = float('nan')
        tests.append(('probability_mix2', args, target, 'none'))
        args = [t.clone() for t in base]
        args[index] = args[index].double()
        tests.append(('probability_mix2', args, target, 'none'))
    for index, replacement in [(0, torch.zeros(17, 3)), (0, torch.zeros(0, 3)),
                               (0, torch.zeros(3, 9)), (1, torch.zeros(32, 3)),
                               (2, torch.zeros(3, 3, 3)), (3, torch.zeros(2, 4))]:
        args = list(base)
        args[index] = replacement
        tests.append(('probability_mix2', args, target, 'none'))
    for y in (target.float(), torch.tensor([-1, 3, 4]), torch.tensor([5, 3, 4]), target[:2]):
        tests.append(('probability_mix2', base, y, 'none'))
    tests += [('foreign', base, target, 'none'), ('probability_mix2', base, target, True),
              ('native', base, target, 'none')]
    for kind, args, y, reduction in tests:
        try:
            nll(kind, args[0], args[1], y, args[2], args[3], reduction=reduction)
        except ValueError:
            pass
        else:
            raise AssertionError('Malformed input was accepted')
    return len(tests)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha', required=True)
    cli = parser.parse_args()
    require(digest(R/'plan.json') == cli.plan_sha, 'Immutable original plan digest differs')
    plan = json.loads((R/'plan.json').read_text())
    require(set(p.name for p in R.iterdir()) == {'plan.json'}, 'One pristine attempt only; never retry')
    for path, wanted in plan['sources'].items():
        require(digest(ROOT/path) == wanted, 'Pinned source differs: '+path)
    require(len(plan['sources']) == 1087, 'Exact1084predecessors plus3new sources required')
    require(digest(E/'readout-mixture-design001.json') == plan['design_sha256'], 'Admitted design differs')
    require(shutil.disk_usage(ROOT).free >= plan['required_free_bytes'], 'Complete prospective capacity not covered')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        torch.set_num_threads(4)
        require(all(os.environ.get(k) == '4' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')),
                'Four-thread environment required')
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=cli.plan_sha))
        try:
            with StrictMpsGuard(R/'mps-guard.json', ROOT), CpuPrerequisiteGuard(
                    R/'cpu-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                mps_rng = torch.mps.get_rng_state().clone()
                cpu_rng = torch.get_rng_state().clone()
                rows = []
                for device, dtype in [('cpu', torch.float64), ('cpu', torch.float32), ('mps', torch.float32)]:
                    rows += run_backend(device, dtype, plan['tolerances'])
                rejected = invalid_cases()
                torch.mps.synchronize()
                require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state()),
                        'Global CPU/MPS RNG changed')
                for path, wanted in plan['sources'].items():
                    require(digest(ROOT/path) == wanted, 'Pinned source changed during proof: '+path)
                publish('result.json', dict(status='complete_generated_readout_component_proof', rows=rows,
                    invalid_cases_rejected=rejected, sources_unchanged=True, inputs_unchanged=True,
                    global_cpu_mps_rng_unchanged=True, actual_model_updates=0, optimizer_calls=0,
                    corpus_or_checkpoint_loads=0, quality_seeds_used=False, tensor_files=0))
            publish('STOP', dict(status='complete_generated_readout_component_proof', no_replay=True))
            print(json.dumps(dict(status='complete', cases=len(rows), invalid_cases=rejected)), flush=True)
        except BaseException as exc:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', no_retry=True, exception=repr(exc)))
            raise


if __name__ == '__main__':
    main()
