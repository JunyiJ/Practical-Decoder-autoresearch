"""One generated full-model integration proof, never corpus/quality execution."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import torch
from research.readout_mixture_model_v1 import GPT, model_config, build_optimizer, load_weights
from research.body_allocation_model_v1.model import GPT as Native
from research.body_allocation_model_v1.contract import model_config as native_config, _BASE_CONFIG
from research.body_allocation_model_v1.optimizer import build_optimizer as native_optimizer
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
SEED = 600550
ARMS = ('native', 'context_mix2', 'probability_mix2')


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw) <= 8*1024**2, 'Metadata cap')
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def clean():
    gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()


def cpu(t):
    return t.detach().cpu().clone()


def hash_tensors(items):
    h = hashlib.sha256()
    for name, t in items:
        h.update(name.encode()); h.update(cpu(t).contiguous().numpy().tobytes())
    return h.hexdigest()


def compare(a, b, *, atol=4e-5, rtol=4e-4):
    a, b = a.detach().cpu(), b.detach().cpu()
    require(a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(), 'Comparison shape/finite')
    error = float((a-b).abs().max())
    require(torch.allclose(a, b, atol=atol, rtol=rtol), 'Fixed tolerance failed: '+str(error))
    return error


def ids(batch, length, device):
    # Literal arithmetic, no random fixtures or data/tokenizer access.
    x = (torch.arange(batch*length, dtype=torch.int64).reshape(batch, length)*37+19) % 50257
    return x.to(device), ((x*13+7) % 50257).to(device)


def grads(model):
    out = {}
    for n, p in model.named_parameters():
        require(p.grad is not None and torch.isfinite(p.grad).all(), 'Missing/nonfinite gradient: '+n)
        out[n] = cpu(p.grad)
    return out


def snapshot_optimizer(opt):
    children = []
    for child in opt.optimizers:
        rows = []
        for group in child.param_groups:
            p = group['params'][0]
            rows.append({k: cpu(v) if type(v) is torch.Tensor else copy.deepcopy(v)
                         for k, v in child.state[p].items()})
        children.append(rows)
    return children


def reject(call, model):
    before = hash_tensors(model.state_dict().items())
    try:
        call()
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError('Adversary accepted')
    require(hash_tensors(model.state_dict().items()) == before, 'Rejected consumer mutated model')


def run():
    baseline = None
    training = SimpleNamespace(**_BASE_CONFIG['training'])
    training.learning_rate = .00125
    initial_hash = None
    private_hash = None
    rows = []
    # First independent genuine-native reference; release before own MPS model.
    with CpuPrerequisiteGuard(R/'cpu-native-reference-guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(SEED)
            base = Native(native_config('incumbent'))
            draw_hash = hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest()
        initial_hash = hash_tensors(base.state_dict().items())
    base = base.to('mps')
    refopt = native_optimizer(base, SimpleNamespace(training=training))
    x, y = ids(2, 8, 'mps')
    native_losses = []
    for _ in range(2):
        refopt.zero_grad(); _, loss = base(x, y); native_losses.append(float(loss)); loss.backward(); refopt.step()
    baseline = dict(parameters={n: cpu(p) for n, p in base.named_parameters()}, optimizer=snapshot_optimizer(refopt))
    del base, refopt, loss; clean()

    for arm in ARMS:
        with CpuPrerequisiteGuard(R/('cpu-'+arm+'-guard.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
            m = GPT(model_config(arm), initializer_seed=SEED)
            require(m.native_cpu_draw_digest == draw_hash and hash_tensors(m.backbone.state_dict().items()) == initial_hash,
                    'Common88 construction/draws differ')
            names = [n for n, _ in m.named_parameters()]
            common = [n for n in names if n.startswith('backbone.')]
            require(len(common) == 88 and names[:88] == common, 'Common88 order differs')
            if arm != 'native':
                ph = hash_tensors(m.context_maps.named_parameters())
                if private_hash is None:
                    private_hash = ph
                require(ph == private_hash and torch.equal(m.router.weight, torch.zeros_like(m.router.weight)), 'Matched private initialization differs')
                require(not torch.equal(m.context_maps[0], m.context_maps[1])
                        and all(bool((p.abs() <= 1/16).all()) for p in m.context_maps), 'Independent standard context maps differ')
            # Generated small actual-width CPU loss/gradient reference.
            xcpu, ycpu = ids(2, 8, 'cpu')
            c_loss = m.loss(xcpu, ycpu); c_loss.backward(); c_grads = grads(m)
            c_value = float(c_loss.detach()); m.zero_grad(set_to_none=True); del c_loss
        m = m.to('mps')
        x, y = ids(2, 8, 'mps')
        loss = m.loss(x, y); loss.backward()
        ce_error = abs(float(loss.detach())-c_value)
        require(ce_error <= 3e-5, 'CPU/MPS CE failed')
        gradient_error = max(compare(p.grad, c_grads[n]) for n, p in m.named_parameters())
        m.zero_grad(set_to_none=True); del loss, c_grads
        # Own target-only path versus full normalized tagged distribution, including gradients.
        target_loss = m.loss(x, y); target_loss.backward(); target_grads = grads(m)
        target_value = float(target_loss.detach()); m.zero_grad(set_to_none=True); del target_loss
        dist = m(x); lp = dist.log_probs()
        require(float(torch.logsumexp(lp, -1).abs().max()) <= 3e-5, 'Distribution not normalized')
        full_loss = -lp.gather(-1, y.unsqueeze(-1)).mean(); full_loss.backward()
        gather_error = abs(float(full_loss.detach())-target_value)
        require(gather_error <= 3e-5, 'Target-only/full gather loss differs')
        gather_gradient_error = max(compare(p.grad, target_grads[n]) for n, p in m.named_parameters())
        m.zero_grad(set_to_none=True); del dist, lp, full_loss, target_grads
        opt = build_optimizer(m, training)
        group_names = [g['parameter_names'][0] for g in opt.param_groups]
        require(len(group_names) == len(names) and set(group_names) == set(names), 'Complete unique group coverage')
        require(all(g['lr'] == .00125*g['width_lr_scale'] and g['weight_decay'] == 0 for g in opt.param_groups), 'Group rates/decay differ')
        before_extra = {n: cpu(p) for n, p in m.named_parameters() if not n.startswith('backbone.')}
        losses = []
        for _ in range(2):
            opt.zero_grad(); loss = m.loss(x, y); losses.append(float(loss.detach())); loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()), 'Full learned gradient exposure')
            require(all(bool(p.grad.abs().max() > 0) for n, p in m.named_parameters()
                        if not n.startswith('backbone.')), 'Added parameter gradient is identically zero')
            opt.step()
        require(opt.completed_steps == 2 and all(v == 2 for v in opt.gradient_steps.values()), 'Full perparameter update exposure')
        if arm == 'native':
            require(losses == native_losses, 'Native same-MPS loss control differs')
            for n, p in m.named_parameters():
                compare(p, baseline['parameters'][n.removeprefix('backbone.')], atol=0, rtol=0)
            own_state = snapshot_optimizer(opt)
            for a, b in zip(own_state, baseline['optimizer']):
                for u, v in zip(a, b):
                    require(u.keys() == v.keys(), 'Native optimizer state keys differ')
                    for key in u:
                        if type(u[key]) is torch.Tensor:
                            compare(u[key], v[key], atol=0, rtol=0)
                        else:
                            require(u[key] == v[key], 'Native optimizer scalar differs')
            del baseline, own_state
        else:
            require(all(not torch.equal(p.cpu(), before_extra[n]) for n, p in m.named_parameters()
                        if n in before_extra), 'Added parameters did not update')
        original_step = opt.completed_steps
        foreign_state = dict(opt.state_dict(), readout_mixture_signature=dict(m.signature, arm='foreign'))
        reject(lambda: opt.load_state_dict(foreign_state), m)
        require(opt.completed_steps == original_step, 'Rejected optimizer changed exposure')
        del foreign_state
        opt.zero_grad(); del loss, opt, before_extra; clean()
        # Genuine full/cached/reset contexts; compare on CPU to bound simultaneous MPS arrays.
        cache_errors = []
        with torch.no_grad():
            for batch in (1, 8):
                xx, _ = ids(batch, 128, 'mps')
                full = cpu(m(xx).log_probs())
                cache = m.caches(batch)
                require(sum(t.numel()*t.element_size() for layer in cache.layers for t in layer.buffers()) == batch*196608,
                        'Native cache bytes differ')
                for repeat in range(2):
                    if repeat:
                        cache.reset()
                    p0 = cpu(m(xx[:, :96], cache=cache).log_probs())
                    p1 = cpu(m(xx[:, 96:], cache=cache, start_pos=96).log_probs())
                    cache_errors.append(compare(full, torch.cat((p0, p1), 1), atol=3e-5, rtol=3e-5))
                    del p0, p1
                del full, cache
            xx, _ = ids(1, 12, 'mps'); changed = xx.clone(); changed[:, 6:] = (changed[:, 6:]+123) % 50257
            v0 = cpu(m(xx).values[:, :6]); v1 = cpu(m(changed).values[:, :6])
            causality_error = compare(v0, v1, atol=0, rtol=0)
            del v0, v1
        # Exact weight identities and persisted/reconstructible buffers; no tensor file writes.
        saved = m.state_dict()
        require(not any('causal_mask' in n for n in saved), 'Nonpersistent causal mask entered checkpoint')
        require(all(torch.equal(b.attn.causal_mask.cpu(), torch.tril(torch.ones(128, 128, dtype=torch.bool)))
                    for b in m.backbone.transformer.h), 'Reconstructible causal mask differs')
        payload = dict(kind='native_readout_mixture_weights_v1', architecture=copy.deepcopy(m.signature), model_state=saved)
        bad = dict(payload, kind='legacy_logits'); reject(lambda: load_weights(m, bad), m)
        bad = dict(payload, architecture=dict(m.signature, arm='foreign')); reject(lambda: load_weights(m, bad), m)
        for name, value in [('missing', None), ('dtype', torch.zeros(256, dtype=torch.float64)),
                            ('shape', torch.zeros(1)), ('nonfinite', torch.full((256,), float('nan')))]:
            forged = saved.copy(); key = 'backbone.transformer.ln_f.bias'
            if name == 'missing':
                forged.pop(key)
            else:
                forged[key] = value
            reject(lambda: load_weights(m, dict(payload, model_state=forged)), m)
        forged = saved.copy(); forged['backbone.transformer.h.0.freq_cos'] = torch.zeros(128, 16)
        reject(lambda: load_weights(m, dict(payload, model_state=forged)), m)
        cache = m.caches(1)
        with torch.no_grad():
            foreign = copy.copy(cache); foreign.owner = lambda: None
            reject(lambda: m(xx, cache=foreign), m)
            load_weights(m, payload)
            reject(lambda: m(xx, cache=cache), m)
            cache.reset(); m(xx, cache=cache)
        del saved, payload, bad, forged, cache, foreign; clean()
        # New own-model input contract adversaries, all rejected before state mutation.
        small, target = ids(1, 4, 'mps')
        reject(lambda: m.loss(small, target.cpu()), m)
        reject(lambda: m.loss(small, target.to(torch.float32)), m)
        reject(lambda: m.loss(small, target[:, :2]), m)
        reject(lambda: m.loss(small, torch.full_like(target, 50257)), m)
        reject(lambda: m.loss(small, target, reduction='sum'), m)
        reject(lambda: m(small.to(torch.float32)), m)
        reject(lambda: m(small.cpu()), m)
        reject(lambda: m(small.expand(9, 4)), m)
        reject(lambda: m(small, start_pos=1), m)
        with torch.autocast('mps', dtype=torch.float16):
            reject(lambda: m(small), m)
        # Full training geometry stress, target-only output path, backward but no optimizer step.
        x, y = ids(8, 128, 'mps'); loss = m.loss(x, y); loss.backward()
        require(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),
                'Full B8x128 loss/backward not finite/complete')
        stress_loss = float(loss.detach()); m.zero_grad(set_to_none=True)
        rows.append(dict(arm=arm, parameters=m.signature['parameters'], tensors=len(names),
            muon=m.signature['muon_groups'], adamw=m.signature['adam_groups'], cpu_mps_ce_error=ce_error,
            cpu_mps_gradient_max_error=gradient_error, target_full_gather_error=gather_error,
            target_full_gradient_max_error=gather_gradient_error, cache_errors=cache_errors,
            causality_error=causality_error, synthetic_optimizer_steps=2, all_parameter_exposure=2,
            native_same_mps_control_exact=arm == 'native', B8x128_stress_loss=stress_loss,
            weight_cache_adversaries=9, optimizer_adversaries=1, input_adversaries=10, tensor_files=0))
        publish('arm-'+arm+'.json', rows[-1])
        print(json.dumps(dict(completed_arm=arm)), flush=True)
        del m, loss, x, y; clean()
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--plan-sha', required=True); arg = ap.parse_args()
    require(digest(R/'plan.json') == arg.plan_sha and set(p.name for p in R.iterdir()) == {'plan.json'}, 'One pristine pinned attempt')
    plan = json.loads((R/'plan.json').read_text())
    require(len(plan['sources']) == 1094, '1089 predecessors plus four model sources and harness')
    for p, h in plan['sources'].items():
        require(digest(ROOT/p) == h, 'Source differs: '+p)
    for ref in plan['initializer_census']:
        require(digest(ref['path']) == ref['sha256'], 'Closed initializer evidence differs')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(all(os.environ.get(k) == '4' for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')), 'Four thread flags')
        torch.set_num_threads(4)
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha, initializer_roles=plan['initializer_roles']))
        try:
            with StrictMpsGuard(R/'mps-guard.json', ROOT):
                cpu_rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
                rows = run()
                require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state()), 'Caller global RNG changed')
                for p, h in plan['sources'].items():
                    require(digest(ROOT/p) == h, 'Source changed: '+p)
                publish('result.json', dict(status='complete_generated_full_readout_model_proof', rows=rows,
                    source_global_cpu_mps_rng_unchanged=True, actual_generated_optimizer_steps=8,
                    corpus_scoring_checkpoint_holdout_quality_seed_calls=0, tensor_files=0,
                    cpu_mps_optimizer_transport_claim=False, quality_gain_claim=False))
            publish('STOP', dict(status='complete_generated_full_readout_model_proof', no_replay=True))
        except BaseException as exc:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', exception=repr(exc), no_retry=True))
            raise


if __name__ == '__main__':
    main()
