"""One NEW generated actual-model proof; no corpus, tensor files or quality work."""
import argparse
import copy
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import torch
from research.readout_projection_model_v1 import GPT, model_config, projection_seed, build_optimizer, load_weights
from research.readout_projection_model_v1.model import Distribution
from research.readout_projection_model_v1.contract import POLICY
from research.body_allocation_model_v1.model import GPT as Native
from research.body_allocation_model_v1.contract import model_config as native_config, _BASE_CONFIG
from research.body_allocation_model_v1.optimizer import build_optimizer as native_optimizer
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
ARMS = ('native', 'linear_projection1', 'tanh_projection1')


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir())+len(raw) < 8*1024**2, 'Complete proof metadata cap')
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def clean():
    gc.collect(); torch.mps.synchronize(); torch.mps.empty_cache()


def cpu(t):
    return t.detach().to(device='cpu', copy=True)


def tree_hash(value):
    h = hashlib.sha256()
    def visit(v):
        if type(v) is torch.Tensor:
            h.update(str((tuple(v.shape), v.dtype)).encode())
            h.update(cpu(v).contiguous().numpy().tobytes())
        elif isinstance(v, dict):
            for k, x in v.items():
                h.update(repr(k).encode()); visit(x)
        elif isinstance(v, (list, tuple)):
            h.update(type(v).__name__.encode())
            for x in v:
                visit(x)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def compare(a, b, *, atol=4e-5, rtol=4e-4):
    a, b = a.detach().cpu(), b.detach().cpu()
    require(a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all(), 'Finite matching shapes required')
    error = float((a-b).abs().max())
    require(torch.allclose(a, b, atol=atol, rtol=rtol), 'Fixed elementwise predicate failed: '+str(error))
    return error


def ids(batch, length, device):
    # NEW deterministic arithmetic fixture, no random/corpus/old fixture inputs.
    x = (torch.arange(batch*length, dtype=torch.int64).reshape(batch, length)*43+29) % 50257
    return x.to(device), ((x*17+11) % 50257).to(device)


def grads(model):
    out = {}
    for name, p in model.named_parameters():
        require(p.grad is not None and torch.isfinite(p.grad).all(), 'Complete finite gradient exposure: '+name)
        out[name] = cpu(p.grad)
    return out


def optimizer_snapshot(opt):
    return [[{k: cpu(v) if type(v) is torch.Tensor else copy.deepcopy(v)
              for k, v in child.state[group['params'][0]].items()}
             for group in child.param_groups] for child in opt.optimizers]


def cache_snapshot(cache):
    return None if cache is None else dict(epoch=cache.epoch,
        positions=[c.position for c in cache.layers],
        tensors=tree_hash([dict(c.named_buffers()) for c in cache.layers]))


def reject(name, fn, model, rejected, *, opt=None, cache=None):
    before = tree_hash(model.state_dict()); epoch = model.weight_epoch
    opt_before = None if opt is None else tree_hash(opt.state_dict())
    cache_before = cache_snapshot(cache)
    try:
        fn()
    except (ValueError, AssertionError):
        pass
    else:
        raise AssertionError('Invalid consumer accepted: '+name)
    require(tree_hash(model.state_dict()) == before and model.weight_epoch == epoch, 'Rejected consumer mutated model/epoch: '+name)
    require(opt is None or tree_hash(opt.state_dict()) == opt_before, 'Rejected consumer mutated optimizer: '+name)
    require(cache_snapshot(cache) == cache_before, 'Rejected consumer mutated cache: '+name)
    rejected.append(name)


def run(seed):
    training = SimpleNamespace(**_BASE_CONFIG['training']); training.learning_rate = .00125
    with CpuPrerequisiteGuard(R/'cpu-native-reference.json', ROOT, timeout_seconds=120, poll_seconds=.25):
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            reference = Native(native_config('incumbent'))
            native_draw = sha_bytes(torch.get_rng_state())
        initial_hash = tree_hash(reference.state_dict())
        initial_names = [n for n, _ in reference.named_parameters()]
    reference = reference.to('mps')
    opt = native_optimizer(reference, SimpleNamespace(training=training))
    x, y = ids(2, 8, 'mps'); native_losses = []
    for _ in range(2):
        opt.zero_grad(); _, loss = reference(x, y); native_losses.append(float(loss.detach())); loss.backward(); opt.step()
    baseline = dict(parameters={n: cpu(p) for n, p in reference.named_parameters()}, optimizer=optimizer_snapshot(opt))
    del reference, opt, loss, x, y; clean()
    private_hash = None; rows = []
    for arm in ARMS:
        rejected = []
        with CpuPrerequisiteGuard(R/('cpu-'+arm+'.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
            m = GPT(model_config(arm), initializer_seed=seed)
            require(m.backbone.cfg == native_config('incumbent') and m.native_cpu_draw_digest == native_draw
                    and tree_hash(m.backbone.state_dict()) == initial_hash, 'Genuine native initialization differs')
            names = [n for n, _ in m.named_parameters()]
            require([n.removeprefix('backbone.') for n in names[:88]] == initial_names, 'Common88 order differs')
            if arm != 'native':
                ph = tree_hash(m.projection.state_dict())
                private_hash = ph if private_hash is None else private_hash
                require(ph == private_hash and bool(((m.projection.weight >= -1/16) & (m.projection.weight <= 1/16)).all()),
                        'Shared private projection initialization differs')
            x, y = ids(2, 8, 'cpu')
            with torch.no_grad():
                c_logits = cpu(m(x).values)
            loss = m.loss(x, y); c_value = float(loss.detach()); loss.backward()
            c_grads = grads(m); m.zero_grad(set_to_none=True); del loss
        m = m.to('mps'); x, y = ids(2, 8, 'mps')
        with torch.no_grad():
            forward_error = compare(m(x).values, c_logits)
        del c_logits
        loss = m.loss(x, y); loss.backward(); ce_error = abs(float(loss.detach())-c_value)
        require(ce_error <= 3e-5, 'CPU/MPS CE failed')
        gradient_error = max(compare(p.grad, c_grads[n]) for n, p in m.named_parameters())
        m.zero_grad(set_to_none=True); del loss, c_grads
        target_loss = m.loss(x, y); target_loss.backward(); target_grads = grads(m)
        target_value = float(target_loss.detach()); m.zero_grad(set_to_none=True); del target_loss
        dist = m(x); lp = dist.log_probs()
        require(dist.policy == POLICY and dist.model_fingerprint == m.signature['model_fingerprint']
                and float(torch.logsumexp(lp, -1).abs().max()) <= 3e-5, 'Own normalized logits distribution differs')
        full_loss = -lp.gather(-1, y.unsqueeze(-1)).mean(); full_loss.backward()
        gather_error = abs(float(full_loss.detach())-target_value)
        require(gather_error <= 3e-5, 'CE/full normalized gather differs')
        gather_gradient_error = max(compare(p.grad, target_grads[n]) for n, p in m.named_parameters())
        m.zero_grad(set_to_none=True); del dist, lp, full_loss, target_grads
        opt = build_optimizer(m, training)
        group_names = [g['parameter_names'][0] for g in opt.param_groups]
        require(len(group_names) == len(names) and set(group_names) == set(names), 'Unique complete parameter groups required')
        require([n for n in group_names if n.startswith('backbone.')] ==
                [n for n in names if n.startswith('backbone.') and n.startswith('backbone.transformer.h.') and dict(m.named_parameters())[n].ndim == 2]+
                [n for n in names if n.startswith('backbone.') and not (n.startswith('backbone.transformer.h.') and dict(m.named_parameters())[n].ndim == 2)],
                'Common Muon/Adam relative group order differs')
        require(all(g['lr'] == .00125*g['width_lr_scale'] and g['weight_decay'] == 0. for g in opt.param_groups), 'Group LR/decay differs')
        require(all(g['width_lr_scale'] == (.5 if g['parameter_names'][0] == 'projection.weight' or
                    (g['parameter_names'][0].startswith('backbone.transformer.h.') and g['params'][0].ndim == 2) else 1.)
                    for g in opt.param_groups), 'Independent matrix/base LR scales differ')
        require(type(opt.optimizers[0]) is torch.optim.Muon and type(opt.optimizers[1]) is torch.optim.AdamW
                and all(g['betas'] == (.9, .999) and g['eps'] == 1e-6 for g in opt.optimizers[1].param_groups), 'Optimizer moment policy differs')
        before_A = None if arm == 'native' else cpu(m.projection.weight)
        stale_step_cache = m.caches(1); losses = []
        for _ in range(2):
            opt.zero_grad(); loss = m.loss(x, y); losses.append(float(loss.detach())); loss.backward()
            require(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()), 'Finite allparameter gradients required')
            if arm != 'native':
                require(bool(m.projection.weight.grad.abs().max() > 0), 'Projection gradient identically zero')
            opt.step()
        require(opt.completed_steps == 2 and all(v == 2 for v in opt.gradient_steps.values()), 'Complete perparameter step exposure required')
        if arm == 'native':
            require(losses == native_losses, 'Native same-MPS losses differ')
            for n, p in m.named_parameters():
                compare(p, baseline['parameters'][n.removeprefix('backbone.')], atol=0, rtol=0)
            own_state = optimizer_snapshot(opt)
            for a, b in zip(own_state, baseline['optimizer']):
                require(len(a) == len(b), 'Native optimizer state group counts differ')
                for u, v in zip(a, b):
                    require(u.keys() == v.keys(), 'Native optimizer state keys differ')
                    for key in u:
                        if type(u[key]) is torch.Tensor:
                            compare(u[key], v[key], atol=0, rtol=0)
                        else:
                            require(u[key] == v[key], 'Native optimizer state scalar differs')
            del baseline, own_state, a, b, u, v, p
        else:
            require(not torch.equal(cpu(m.projection.weight), before_A), 'Projection did not update')
        small, target = ids(1, 4, 'mps')
        with torch.no_grad():
            reject('stale_after_step', lambda: m(small, cache=stale_step_cache), m, rejected, cache=stale_step_cache)
        state = opt.state_dict()
        reject('foreign_optimizer_signature', lambda: opt.load_state_dict(dict(state, readout_projection_signature=dict(m.signature, arm='foreign'))), m, rejected, opt=opt)
        reject('optimizer_exposure_bool', lambda: opt.load_state_dict(dict(state, completed_steps=True)), m, rejected, opt=opt)
        bad = dict(state); bad['optimizers'] = [dict(v, param_groups=copy.deepcopy(v['param_groups'])) for v in state['optimizers']]
        bad['optimizers'][1]['param_groups'][0]['eps'] = 1e-8
        reject('optimizer_epsilon', lambda: opt.load_state_dict(bad), m, rejected, opt=opt)
        cache = m.caches(1); opt_before = tree_hash(opt.state_dict()); opt.load_state_dict(state)
        require(tree_hash(opt.state_dict()) == opt_before, 'Valid own optimizer load changed state values')
        with torch.no_grad():
            reject('stale_after_optimizer_load', lambda: m(small, cache=cache), m, rejected, cache=cache)
        opt.zero_grad(); del opt, state, bad, loss, before_A, stale_step_cache, cache; clean()
        # Every large reference/gradient/optimizer snapshot is released before full geometry.
        x, y = ids(8, 128, 'mps'); loss = m.loss(x, y); loss.backward()
        require(torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()), 'Full B8T128 backward exposure failed')
        stress_loss = float(loss.detach()); m.zero_grad(set_to_none=True); del loss, x, y; clean()
        cache_errors = []; cache_raw_errors = []
        with torch.no_grad():
            for batch in (1, 8):
                xx, _ = ids(batch, 128, 'mps'); distribution = m(xx)
                full = cpu(distribution.log_probs()); full_raw = cpu(distribution.values); del distribution
                cache = m.caches(batch)
                require(sum(t.numel()*t.element_size() for layer in cache.layers for t in layer.buffers()) == batch*196608, 'Cache byte layout differs')
                for repeat in range(2):
                    if repeat:
                        cache.reset()
                    distribution = m(xx[:, :96], cache=cache)
                    p0 = cpu(distribution.log_probs()); r0 = cpu(distribution.values); del distribution
                    distribution = m(xx[:, 96:], cache=cache, start_pos=96)
                    p1 = cpu(distribution.log_probs()); r1 = cpu(distribution.values); del distribution
                    cache_raw_errors.append(compare(full_raw, torch.cat((r0, r1), 1), atol=3e-5, rtol=3e-5))
                    cache_errors.append(compare(full, torch.cat((p0, p1), 1), atol=3e-5, rtol=3e-5)); del p0, p1, r0, r1
                del full, full_raw, cache, xx
            xx, _ = ids(1, 12, 'mps'); changed = xx.clone(); changed[:, 6:] = (changed[:, 6:]+137) % 50257
            v0 = cpu(m(xx).values[:, :6]); v1 = cpu(m(changed).values[:, :6]); causality_error = compare(v0, v1, atol=0, rtol=0)
            del v0, v1, xx, changed
        saved = m.state_dict()
        require(not any('causal_mask' in n for n in saved) and all(torch.equal(b.attn.causal_mask.cpu(), torch.tril(torch.ones(128, 128, dtype=torch.bool))) for b in m.backbone.transformer.h), 'Reconstructible mask persistence differs')
        payload = dict(kind='native_readout_projection_weights_v1', architecture=copy.deepcopy(m.signature), model_state=saved)
        reject('foreign_weight_kind', lambda: load_weights(m, dict(payload, kind='native_readout_mixture_weights_v1')), m, rejected)
        reject('foreign_weight_architecture', lambda: load_weights(m, dict(payload, architecture=dict(m.signature, arm='foreign'))), m, rejected)
        for name, value in [('missing', None), ('dtype', torch.zeros(256, dtype=torch.float64)), ('shape', torch.zeros(1)), ('nonfinite', torch.full((256,), float('nan')))]:
            forged = saved.copy(); key = 'backbone.transformer.ln_f.bias'
            if name == 'missing':
                forged.pop(key)
            else:
                forged[key] = value
            reject('weight_'+name, lambda: load_weights(m, dict(payload, model_state=forged)), m, rejected)
        forged = saved.copy(); forged['backbone.transformer.h.0.freq_cos'] = torch.zeros(128, 16)
        reject('weight_rope', lambda: load_weights(m, dict(payload, model_state=forged)), m, rejected)
        del forged
        cache = m.caches(1)
        with torch.no_grad():
            foreign = copy.copy(cache); foreign.owner = lambda: None
            reject('cache_foreign_owner', lambda: m(small, cache=foreign), m, rejected, cache=cache)
            foreign = copy.copy(cache); foreign.signature = dict(cache.signature, arm='foreign')
            reject('cache_foreign_signature', lambda: m(small, cache=foreign), m, rejected, cache=cache)
            foreign = copy.copy(cache); foreign.layers = [copy.copy(c) for c in cache.layers]
            for c in foreign.layers:
                c._buffers = dict(c._buffers)
            foreign.layers[1].c_cache = foreign.layers[0].c_cache
            reject('cache_aliased_storage', lambda: m(small, cache=foreign), m, rejected, cache=cache)
            reject('cache_wrong_position', lambda: m(small, cache=cache, start_pos=1), m, rejected, cache=cache)
            load_weights(m, payload)
            reject('stale_after_weights_load', lambda: m(small, cache=cache), m, rejected, cache=cache)
            cache.reset(); m(small, cache=cache); cache.reset()
        reject('cache_with_grad', lambda: m(small, cache=cache), m, rejected, cache=cache)
        del saved, payload, cache, foreign
        old_cfg = m.cfg; m.cfg = copy.deepcopy(m.cfg); m.cfg['readout_projection']['divisor'] = 8.
        reject('mutable_config', lambda: m(small), m, rejected); m.cfg = old_cfg
        for name, fn in [('target_device', lambda: m.loss(small, target.cpu())),
                         ('target_dtype', lambda: m.loss(small, target.float())),
                         ('target_shape', lambda: m.loss(small, target[:, :2])),
                         ('target_range', lambda: m.loss(small, torch.full_like(target, 50257))),
                         ('reduction', lambda: m.loss(small, target, reduction='sum')),
                         ('token_dtype', lambda: m(small.float())), ('token_device', lambda: m(small.cpu())),
                         ('B9', lambda: m(small.expand(9, 4))), ('T129', lambda: m(torch.zeros(1, 129, dtype=torch.int64, device='mps'))),
                         ('token_range', lambda: m(torch.full_like(small, 50257))), ('offset_without_cache', lambda: m(small, start_pos=1)),
                         ('foreign_distribution_policy', lambda: Distribution('foreign', m.signature['model_fingerprint'], 'native_logits', torch.zeros(1)).log_probs()),
                         ('mixture_distribution_kind', lambda: Distribution(POLICY, m.signature['model_fingerprint'], 'probability_mix_log_probs', torch.zeros(1)).log_probs())]:
            reject(name, fn, m, rejected)
        for ambient, dt in [('cpu', torch.bfloat16), ('mps', torch.float16)]:
            with torch.autocast(ambient, dtype=dt):
                reject('autocast_'+ambient, lambda: m(small), m, rejected)
        rows.append(dict(arm=arm, parameters=m.signature['parameters'], tensors=len(names), muon=m.signature['muon_groups'], adamw=m.signature['adam_groups'],
            cpu_mps_forward_max_error=forward_error, cpu_mps_ce_error=ce_error, cpu_mps_gradient_max_error=gradient_error, target_full_gather_error=gather_error,
            target_full_gradient_max_error=gather_gradient_error, cache_errors=cache_errors, cache_raw_errors=cache_raw_errors, causality_error=causality_error,
            synthetic_optimizer_steps=2, all_parameter_exposure=2, native_same_mps_control_exact=arm == 'native',
            B8T128_backward_loss=stress_loss, rejected_cases=rejected, all_elementwise_predicates_passed=True, tensor_files=0))
        publish('arm-'+arm+'.json', rows[-1]); print(json.dumps(dict(completed_arm=arm, rejected=len(rejected))), flush=True)
        del m, small, target; clean()
    return rows


def sha_bytes(t):
    return hashlib.sha256(bytes(t.tolist())).hexdigest()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan-sha', required=True); arg = parser.parse_args()
    require(sha(R/'plan.json') == arg.plan_sha and set(p.name for p in R.iterdir()) == {'plan.json'}, 'One pristine exact pinned plan required')
    plan = json.loads((R/'plan.json').read_text()); require(len(plan['sources']) == 1136, 'All1134 predecessors plus two new harnesses required')
    for p, h in plan['sources'].items():
        require(sha(ROOT/p) == h, 'Pinned source differs: '+p)
    for ref in plan['initializer_census']:
        require(sha(ref['path']) == ref['sha256'], 'Original initializer census differs')
    census = json.loads(Path(plan['initializer_census_result']).read_text())
    require(census['status'] == 'new_projection_initializer_roles_unused_before_construction' and census['prior_matches'] == [], 'Accepted new unused-role census required')
    seed = census['roles']['initializer']
    require(projection_seed(seed) == census['roles']['private_projection'], 'Audited private initializer role differs')
    require(shutil.disk_usage(ROOT).free >= 10*1024**3+8*1024**2, 'Proof output/reserve/extra2GiB required')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB); torch.set_num_threads(4)
        environment = dict(torch_version=str(torch.__version__), torch_git=torch.version.git_version, threads=torch.get_num_threads(),
                           flags={k:os.environ.get(k) for k in plan['environment']['flags']})
        require(environment == plan['environment'], 'Exact admitted runtime/flags required')
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha, initializer_roles=census['roles'], environment=environment))
        try:
            with StrictMpsGuard(R/'mps-guard.json', ROOT):
                cpu_rng, mps_rng = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
                rows = run(seed)
                require(torch.equal(cpu_rng, torch.get_rng_state()) and torch.equal(mps_rng, torch.mps.get_rng_state()), 'Caller global CPU/MPS RNG changed')
                for p, h in plan['sources'].items():
                    require(sha(ROOT/p) == h, 'Source changed: '+p)
                publish('result.json', dict(status='complete_generated_single_projection_actual_model_proof', rows=rows,
                    source_global_cpu_mps_rng_unchanged=True, actual_generated_optimizer_steps=8, full_B8T128_backwards=3,
                    corpus_scoring_checkpoint_holdout_quality_seed_calls=0, tensor_files=0,
                    cpu_mps_optimizer_transport_claim=False, resume_claim=False, quality_gain_claim=False))
            publish('STOP', dict(status='complete_generated_single_projection_actual_model_proof', no_replay=True, no_retry=True))
        except BaseException as exc:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', error=repr(exc), no_retry=True))
            raise


if __name__ == '__main__':
    main()
