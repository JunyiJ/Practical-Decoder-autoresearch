"""NEW TOP model-consumer proof SOURCE. No imports/calls until separately admitted.

This is not the closed copied-block/shared-W fixture or a state/cache proof.
The runtime plan and exclusive initializer census are deliberately NOT admitted.
"""
import base64
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import zlib

ROOT = Path(__file__).resolve().parents[3]
CASES = (
    ('unit_nearest_001', 1, 1, ((40001, 1117),)),
    ('repeated_tail_001', 3, 4, ((2107, 41, 2107, 913, 41, 49999, 2107),
                              (87, 50003, 87, 111, 50003, 111, 209))),
    ('changed_tail_001', 3, 4, ((2107, 41, 2107, 112, 41, 112, 41),
                             (87, 50003, 87, 208, 209, 208, 209))),
)
ARMS = (('native', 'repeated_tail_001'), ('copied_readout_next', 'repeated_tail_001'),
        ('future_proximity', 'unit_nearest_001'), ('future_proximity', 'repeated_tail_001'),
        ('future_proximity', 'changed_tail_001'))
BACKENDS = ('cpu', 'mps')
NEGATIVE_IDS = (
    'context_bool', 'context_zero', 'context_129', 'context_float', 'raw_list',
    'raw_float', 'raw_bool', 'raw_rank1', 'raw_empty_batch', 'raw_batch9',
    'raw_short', 'raw_long', 'raw_negative', 'raw_oov', 'raw_foreign_device',
    'spec_window_zero', 'cfg_dim_bool', 'child_mode', 'aux_nan', 'aux_inf',
    'aux_shape', 'aux_alias', 'aux_frozen', 'cpu_autocast',
    'general_forward', 'model_state', 'model_load', 'optimizer_state', 'optimizer_load',
    'lr_zero', 'lr_bool', 'lr_int', 'lr_nan', 'lr_too_large', 'zero_nonbool',
    'group_eps', 'group_lr_nan', 'group_lr_negative', 'group_width_schedule',
    'group_names', 'group_foreign_parameter', 'gradient_missing', 'gradient_nan',
    'gradient_inf',
)
CPU_ONLY_NEGATIVE_IDS = ('gradient_sparse_cpu',)
TOLERANCES = {'loss': (3e-5, 0.), 'logits': (4e-5, 4e-4),
              'logit_gradient': (4e-6, 4e-5), 'parameter_gradient': (4e-5, 4e-4),
              'hidden_gradient': (4e-5, 4e-4), 'head_update': (4e-5, 4e-4),
              'exact': (0., 0.)}
RESOURCE_CONTRACT = {
    'CPU_phase_seconds': 120, 'MPS_whole_seconds': 600, 'RSS_bytes': 4294967296,
    'MPS_bytes': 6442450944, 'reserve_bytes': 8589934592, 'extra_bytes': 2147483648,
    'output_bytes': 536870912, 'tensor_files': 0, 'threads': 4,
    'planned_RSS_bytes': 3758096384, 'planned_MPS_bytes': 3221225472,
    'forecast_seconds': 360, 'forecast_output_bytes': 201326592,
}
EXPECTED_REPORTS = {'arms': 10, 'transactions': 89, 'tail_pairs': 4, 'updates': 4}


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_safe(value):
    if type(value) is float and not math.isfinite(value): return {'nonfinite': repr(value)}
    if type(value) is dict: return {k: json_safe(v) for k, v in value.items()}
    if type(value) in (tuple, list): return [json_safe(v) for v in value]
    return value


def publish(run, name, payload):
    data = (json.dumps(json_safe(payload), sort_keys=True, allow_nan=False) + '\n').encode()
    used = sum(p.stat().st_size for p in run.iterdir() if p.is_file())
    if used + len(data) + 1048576 > RESOURCE_CONTRACT['output_bytes']:
        raise ValueError('Original complete-output cap exceeded; no retry/expansion')
    with (run / name).open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def independent_targets(raw, context, window):
    """Independent first-occurrence scan of NEW raw literals; no candidate helpers."""
    result = []
    for sequence in raw:
        for position in range(context):
            first = {}
            for distance in range(1, window + 1):
                token = sequence[position + distance]
                if token not in first:
                    first[token] = distance
            unnormalized = [(token, math.exp(1 - distance)) for token, distance in first.items()]
            total = math.fsum(mass for _, mass in unnormalized)
            # Exactly one FP32 representation of each Python mass; never renormalize.
            result.append(tuple((token, struct.unpack('<f', struct.pack('<f', mass / total))[0])
                                for token, mass in unnormalized))
    return tuple(result)


def independent_scalar(logits_cpu, targets):
    """CPU Python math.fsum/LSE, full V=50257 loss and analytic logit derivative."""
    rows = logits_cpu.reshape(-1, 50257).tolist()
    losses, gradients = [], []
    for values, weights in zip(rows, targets):
        maximum = max(values)
        exponentials = [math.exp(value - maximum) for value in values]
        denominator = math.fsum(exponentials)
        normalizer = maximum + math.log(denominator)
        mass = math.fsum(weight for _, weight in weights)
        by_id = dict(weights)
        losses.append(math.fsum(weight * (normalizer - values[token]) for token, weight in weights))
        gradients.extend((mass * exponential / denominator - by_id.get(token, 0.)) / len(rows)
                         for token, exponential in enumerate(exponentials))
    return math.fsum(losses) / len(rows), gradients


def cpu_values(value):
    # Transfer BEFORE FP64 conversion. No MPS FP64 construction, including invalids.
    return value.detach().to('cpu').contiguous()


def dense_evidence(value):
    cpu = cpu_values(value)
    raw = cpu.numpy().tobytes()
    return {'shape': list(cpu.shape), 'dtype': str(cpu.dtype), 'elements': cpu.numel(),
            'sha256': hashlib.sha256(raw).hexdigest(), 'encoding': 'zlib-base64-little-endian',
            'coordinates': base64.b64encode(zlib.compress(raw)).decode('ascii')}


def comparison(actual, expected, tolerance, *, dense=False):
    """Every coordinate uses the fixed expected-relative predicate, not a max gate.

    Parameters retain full boolean masks and both vector digests; at most64 failed
    coordinates additionally retain actual/expected/error. Logits/oracle/hidden
    coordinates retain complete dense vectors as compressed JSON evidence.
    """
    import torch
    a, e = cpu_values(actual), cpu_values(expected)
    if a.shape != e.shape:
        return {'passed': False, 'actual_shape': list(a.shape), 'expected_shape': list(e.shape)}
    ad, ed = a.to(torch.float64), e.to(torch.float64)
    atol, rtol = tolerance
    errors = (ad - ed).abs()
    mask = torch.isfinite(ad) & torch.isfinite(ed) & (errors <= atol + rtol * ed.abs())
    bad = (~mask).reshape(-1).nonzero().reshape(-1)
    mask_bytes = mask.numpy().tobytes()
    payload = {'shape': list(a.shape), 'coordinates': a.numel(), 'atol': atol, 'rtol': rtol,
               'passed': bool(mask.all()), 'failed_coordinates': bad.numel(),
               'mask_encoding': 'zlib-base64-bool8',
               'all_coordinate_mask': base64.b64encode(zlib.compress(mask_bytes)).decode('ascii'),
               'actual_sha256': hashlib.sha256(a.numpy().tobytes()).hexdigest(),
               'expected_sha256': hashlib.sha256(e.numpy().tobytes()).hexdigest(),
               'max_abs': float(errors.max()) if errors.numel() else 0.,
               'first_64_failures': [[i, float(ad.reshape(-1)[i]), float(ed.reshape(-1)[i]),
                                      float(errors.reshape(-1)[i])] for i in bad[:64].tolist()]}
    if dense:
        payload['actual'] = dense_evidence(a); payload['expected'] = dense_evidence(e)
    return payload


def digest_tensor(value):
    cpu = cpu_values(value)
    return (str(cpu.dtype), tuple(cpu.shape), hashlib.sha256(cpu.numpy().tobytes()).hexdigest())


def snapshot(model, optimizer, raw, extra=None):
    import torch
    parameters = tuple((name, digest_tensor(p), None if p.grad is None else
                        ('sparse', digest_tensor(p.grad.to_dense())) if p.grad.layout != torch.strided
                        else digest_tensor(p.grad)) for name, p in optimizer._items)
    own_inventory = tuple((name, id(p), p.untyped_storage().data_ptr(), tuple(p.shape), str(p.dtype), p.requires_grad)
                          for name, p in model.ordered_parameters())
    original_ids = {id(p) for _, p in optimizer._items}
    foreign_parameters = tuple((name, digest_tensor(p), None if p.grad is None else digest_tensor(p.grad))
                               for name, p in model.ordered_parameters() if id(p) not in original_ids)
    buffers = tuple((name, digest_tensor(p)) for name, p in model.native.named_buffers())
    groups = tuple(tuple((key, tuple(id(p) for p in value) if key == 'params' else repr(value))
                         for key, value in sorted(group.items())) for group in optimizer.param_groups)
    # All negative transactions precede the first update: state must be empty.
    empty_states = tuple(len(child.state) for child in optimizer.optimizers)
    from omegaconf import OmegaConf
    if type(extra) is torch.Tensor:
        extra = (str(extra.device), str(extra.dtype), tuple(extra.shape), None if extra.device.type == 'meta' else digest_tensor(extra))
    return {'parameters_and_populated_gradslots': parameters, 'own_inventory': own_inventory, 'foreign_parameters': foreign_parameters, 'buffers': buffers,
            'spec': repr(model.spec), 'cfg': OmegaConf.to_container(model.native.cfg, resolve=True),
            'modes': (model.training, model.native.training), 'extra_input_payload': extra, 'groups': groups,
            'optimizer_state_sizes': empty_states, 'raw': digest_tensor(raw),
            'CPU_RNG': digest_tensor(torch.get_rng_state()), 'MPS_RNG': digest_tensor(torch.mps.get_rng_state())}


def all_ok(value):
    if type(value) is dict:
        return value.get('passed', True) and all(all_ok(v) for v in value.values())
    if type(value) in (list, tuple):
        return all(all_ok(v) for v in value)
    return True


def graph_identity(primary, auxiliary):
    def nodes(value):
        pending, visited = [value.grad_fn], set()
        while pending:
            node = pending.pop()
            if node is None or node in visited: continue
            visited.add(node)
            pending.extend(edge[0] for edge in node.next_functions)
        return visited
    first, second = nodes(primary), nodes(auxiliary)
    norm_first = {node for node in first if 'LayerNormBackward' in type(node).__name__}
    norm_second = {node for node in second if 'LayerNormBackward' in type(node).__name__}
    return {'passed': len(norm_first) == len(norm_second) == len(norm_first & norm_second) == 1,
            'primary_finalnorm_nodes': len(norm_first), 'aux_finalnorm_nodes': len(norm_second),
            'shared_finalnorm_nodes': len(norm_first & norm_second), 'hooks_or_candidate_hidden_recompute': False}


def reference_hidden(reference, inputs):
    hidden = reference.transformer.wte(inputs)
    for block in reference.transformer.h:
        hidden, auxiliary = block(hidden)
        if auxiliary is not None: raise ValueError('Reference unrelated objective')
    return reference.transformer.ln_f(hidden)


def negative_transaction(run, ident, model, optimizer, raw, setup, action, restore, extra):
    setup()
    before = snapshot(model, optimizer, raw, extra)
    caught = None
    try:
        action()
    except ValueError as error:
        caught = str(error)
    finally:
        after = snapshot(model, optimizer, raw, extra)
        restore()
    evidence = {'id': ident, 'rejected_ValueError': caught is not None, 'error': caught,
                'before': before, 'after': after, 'passed': caught is not None and before == after}
    publish(run, ident + '.json', evidence)
    if not evidence['passed']: raise AssertionError(ident)


def negatives(run, model, optimizer, raw, context, device):
    import torch
    from research.top_proximity_model_v1.optimizer import build_optimizer
    noop = lambda: None
    cases = {}
    def add(name, action, setup=noop, restore=noop, extra=None): cases[name] = (setup, action, restore, extra)
    for name, value in (('context_bool', True), ('context_zero', 0), ('context_129', 129), ('context_float', 3.)):
        add(name, lambda value=value: model.forward_train(raw, value))
    raw_bad = {'raw_list': cpu_values(raw).tolist(), 'raw_float': raw.float(), 'raw_bool': raw.bool(),
               'raw_rank1': raw[0], 'raw_empty_batch': raw[:0], 'raw_batch9': raw[:1].repeat(9, 1),
               'raw_short': raw[:, :-1], 'raw_long': torch.cat((raw, raw[:, :1]), dim=1),
               'raw_negative': raw.clone(), 'raw_oov': raw.clone(),
               'raw_foreign_device': raw.cpu() if device == 'mps' else torch.empty(raw.shape, device='meta', dtype=torch.int64)}
    raw_bad['raw_negative'][0, 0] = -1; raw_bad['raw_oov'][0, 0] = 50257
    for name, value in raw_bad.items(): add(name, lambda value=value: model.forward_train(value, context), extra=value)
    old_spec_window = model.spec.window
    add('spec_window_zero', lambda: model.forward_train(raw, context),
        lambda: object.__setattr__(model.spec, 'window', 0), lambda: object.__setattr__(model.spec, 'window', old_spec_window))
    old_dim = model.native.cfg.dim
    add('cfg_dim_bool', lambda: model.forward_train(raw, context),
        lambda: setattr(model.native.cfg, 'dim', True), lambda: setattr(model.native.cfg, 'dim', old_dim))
    add('child_mode', lambda: model.forward_train(raw, context),
        lambda: setattr(model.native, 'training', False), lambda: setattr(model.native, 'training', True))
    aux = model.aux_weight
    old_scalar = float(aux[0, 0].detach().cpu())
    def scalar(value):
        with torch.no_grad(): aux[0, 0] = value
    for name, value in (('aux_nan', float('nan')), ('aux_inf', float('inf'))):
        add(name, lambda: model.forward_train(raw, context), lambda value=value: scalar(value), lambda: scalar(old_scalar))
    bad_aux = torch.nn.Parameter(aux[:1].detach().clone())
    add('aux_shape', lambda: model.forward_train(raw, context), lambda: setattr(model, 'aux_weight', bad_aux), lambda: setattr(model, 'aux_weight', aux))
    add('aux_alias', lambda: model.forward_train(raw, context), lambda: setattr(model, 'aux_weight', model.native.lm_head.weight), lambda: setattr(model, 'aux_weight', aux))
    add('aux_frozen', lambda: model.forward_train(raw, context), lambda: aux.requires_grad_(False), lambda: aux.requires_grad_(True))
    def autocast_action():
        with torch.autocast('cpu', dtype=torch.bfloat16): model.forward_train(raw, context)
    add('cpu_autocast', autocast_action)
    for name, action in (('general_forward', lambda: model(raw)), ('model_state', model.state_dict),
                         ('model_load', lambda: model.load_state_dict({})), ('optimizer_state', optimizer.state_dict),
                         ('optimizer_load', lambda: optimizer.load_state_dict({})), ('zero_nonbool', lambda: optimizer.zero_grad(1))):
        add(name, action)
    for name, value in (('lr_zero', 0.), ('lr_bool', True), ('lr_int', 1), ('lr_nan', float('nan')), ('lr_too_large', 1.01)):
        add(name, lambda value=value: build_optimizer(model, value))
    adam = optimizer.optimizers[1].param_groups[-1]
    def group_case(name, group, key, value):
        old = group[key]
        add(name, optimizer.step, lambda: group.__setitem__(key, value), lambda: group.__setitem__(key, old))
    group_case('group_eps', adam, 'eps', 1e-5)
    group_case('group_lr_nan', adam, 'lr', float('nan'))
    group_case('group_lr_negative', adam, 'lr', -.001)
    group_case('group_width_schedule', adam, 'lr', .002)
    group_case('group_names', adam, 'parameter_names', ['foreign'])
    group_case('group_foreign_parameter', adam, 'params', [model.native.lm_head.weight])
    parameter = model.native.transformer.ln_f.weight
    original_grad = parameter.grad
    add('gradient_missing', optimizer.step, lambda: setattr(parameter, 'grad', None), lambda: setattr(parameter, 'grad', original_grad))
    for name, value in (('gradient_nan', float('nan')), ('gradient_inf', float('inf'))):
        bad = original_grad.clone(); bad[0] = value
        add(name, optimizer.step, lambda bad=bad: setattr(parameter, 'grad', bad), lambda: setattr(parameter, 'grad', original_grad))
    if device == 'cpu':
        sparse = original_grad.to_sparse()
        add('gradient_sparse_cpu', optimizer.step, lambda: setattr(parameter, 'grad', sparse), lambda: setattr(parameter, 'grad', original_grad))
    expected = NEGATIVE_IDS + (CPU_ONLY_NEGATIVE_IDS if device == 'cpu' else ())
    if set(cases) != set(expected): raise AssertionError('Concrete negative payload coverage differs')
    for name in expected:
        setup, action, restore, extra = cases[name]
        negative_transaction(run, device + '-negative-' + name, model, optimizer, raw, setup, action, restore, extra)
    return len(expected)


def run_arm(run, device, variant, case, initializer):
    import torch
    from research.final_ln_scale_model_v1.runtime.models.gpt import GPT as Native
    from research.top_proximity_model_v1 import contract as C
    from research.top_proximity_model_v1.model import TopModel
    from research.top_proximity_model_v1.optimizer import build_optimizer
    name, context, window, literals = case
    ident = device + '-' + variant + '-' + name
    # CPU initializer is private to this NEW fixture; no global MPS seed/draw.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(initializer)
        reference = Native(C.native_config())
        reference_draws = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(initializer)
        model = TopModel(C.Spec(variant, window))
        own_draws = torch.get_rng_state().clone()
    init = {'RNG_draws_exact': torch.equal(reference_draws, own_draws), 'parameters': {},
            'native_count_exact': len(tuple(model.native.parameters())) == len(tuple(reference.parameters())) == 88}
    for (name1, own), (name2, ref) in zip(model.native.named_parameters(), reference.named_parameters()):
        init['parameters'][name1] = comparison(own, ref, TOLERANCES['exact'])
        if name1 != name2: raise AssertionError('Reference initialization order')
    if model.aux_weight is not None:
        init['aux'] = comparison(model.aux_weight, reference.lm_head.weight, TOLERANCES['exact'])
        init['separate_storage'] = model.aux_weight.untyped_storage().data_ptr() != model.native.lm_head.weight.untyped_storage().data_ptr()
    model.to(device); reference.to(device)
    optimizer = build_optimizer(model, .00125)
    for _, parameter in model.ordered_parameters(): parameter.grad = torch.full_like(parameter, .125)
    raw = torch.tensor(literals, dtype=torch.int64, device=device)
    raw_before = raw.clone()
    inputs = raw[:, :context].contiguous(); labels = raw[:, 1:context + 1].contiguous()
    before_math = snapshot(model, optimizer, raw)
    output = model.forward_train(raw, context)
    with torch.no_grad():
        true_primary, _ = reference(inputs)
    checks = {'initialization': init, 'genuine_primary': comparison(output.primary.values, true_primary, TOLERANCES['exact'], dense=True),
              'raw_preserved': torch.equal(raw, raw_before), 'identity': output.identity == model.identity,
              'primary_identity': output.primary.identity == model.identity,
              'training_kind': output.kind == 'top_model_training_loss_v1',
              'primary_kind': output.primary.kind == 'top_model_primary_next_logits_v1',
              'optimizer_groups': {'passed': len(optimizer.optimizers[0].param_groups) == 44 and len(optimizer.optimizers[1].param_groups) == (44 if variant == 'native' else 45),
                                   'Muon': len(optimizer.optimizers[0].param_groups), 'AdamW': len(optimizer.optimizers[1].param_groups)}}
    with torch.no_grad(): uncached = model.primary_logits(inputs)
    checks['uncached_primary'] = comparison(uncached.values, true_primary, TOLERANCES['exact'])
    primary_cpu = cpu_values(true_primary)
    primary_rows = tuple(((int(token), 1.),) for sequence in literals for token in sequence[1:context + 1])
    primary_loss, primary_derivative = independent_scalar(primary_cpu, primary_rows)
    checks['primary_loss'] = comparison(output.primary_loss, torch.tensor(primary_loss, dtype=torch.float64), TOLERANCES['loss'], dense=True)
    update_report = None; transaction_count = 0
    if variant != 'native':
        hidden = reference_hidden(reference, inputs)
        ref_aux_weight = reference.lm_head.weight.detach().clone().requires_grad_()
        ref_primary = reference.lm_head(hidden) / 4.
        ref_aux = torch.nn.functional.linear(hidden, ref_aux_weight) / 4.
        target_rows = primary_rows if variant == 'copied_readout_next' else independent_targets(literals, context, window)
        auxiliary_loss, auxiliary_derivative = independent_scalar(cpu_values(ref_aux), target_rows)
        checks['auxiliary_loss'] = comparison(output.auxiliary_loss, torch.tensor(auxiliary_loss, dtype=torch.float64), TOLERANCES['loss'], dense=True)
        checks['total_loss'] = comparison(output.total_loss, torch.tensor(.5 * primary_loss + .5 * auxiliary_loss, dtype=torch.float64), TOLERANCES['loss'], dense=True)
        checks['auxiliary_logits'] = comparison(output.auxiliary_logits, ref_aux, TOLERANCES['logits'], dense=True)
        checks['one_shared_final_norm'] = graph_identity(output.primary.values, output.auxiliary_logits)
        gp64 = torch.tensor(primary_derivative, dtype=torch.float64).reshape(ref_primary.shape)
        ga64 = torch.tensor(auxiliary_derivative, dtype=torch.float64).reshape(ref_aux.shape)
        gp = gp64.to(dtype=torch.float32, device=device); ga = ga64.to(dtype=torch.float32, device=device)
        actual_head_grads = torch.autograd.grad(output.total_loss, (output.primary.values, output.auxiliary_logits), retain_graph=True)
        checks['primary_logit_gradient'] = comparison(actual_head_grads[0], .5 * gp64, TOLERANCES['logit_gradient'], dense=True)
        checks['auxiliary_logit_gradient'] = comparison(actual_head_grads[1], .5 * ga64, TOLERANCES['logit_gradient'], dense=True)
        hidden_expected = (.5 * gp @ reference.lm_head.weight.detach() + .5 * ga @ ref_aux_weight.detach()) / 4.
        hidden_actual = torch.autograd.grad((ref_primary, ref_aux), hidden, (.5 * gp, .5 * ga), retain_graph=True)[0]
        checks['summed_hidden_gradient'] = comparison(hidden_actual, hidden_expected, TOLERANCES['hidden_gradient'], dense=True)
        own_items = model.ordered_parameters()
        ref_items = tuple(reference.named_parameters()) + (('aux_weight', ref_aux_weight),)
        expected_grads = torch.autograd.grad((ref_primary, ref_aux), tuple(p for _, p in ref_items), (.5 * gp, .5 * ga))
        own_grads = torch.autograd.grad(output.total_loss, tuple(p for _, p in own_items))
        checks['parameter_gradients'] = {name1: comparison(own, expected, TOLERANCES['parameter_gradient'])
                                        for (name1, _), own, expected in zip(own_items, own_grads, expected_grads)}
        checks['gradient_coverage'] = len(own_grads) == len(expected_grads) == 89 and tuple(name for name, _ in own_items) == tuple(name for name, _ in ref_items)
        checks['aux_nonzero_gradient'] = bool(own_grads[-1].abs().sum() > 0)
        checks['represented_targets'] = target_rows
        checks['caller_preservation'] = {'before': before_math, 'after': snapshot(model, optimizer, raw)}
        checks['passed'] = all_ok(checks) and all(value for value in checks.values() if type(value) is bool) and all(value for value in init.values() if type(value) is bool) and checks['caller_preservation']['before'] == checks['caller_preservation']['after']
        # All original scalar/coordinate/mask evidence is durable BEFORE predicates.
        publish(run, ident + '-math.json', checks)
        if not all_ok(checks) or not all(v for v in init.values() if type(v) is bool) or not checks['gradient_coverage'] or not checks['aux_nonzero_gradient']:
            raise AssertionError(ident + ' mathematical predicates')
        if name == 'repeated_tail_001':
            # Drop both graph lifetimes before preflight snapshots, clipping or update.
            for (_, parameter), grad in zip(own_items, own_grads): parameter.grad = grad.detach()
            del expected_grads, own_grads, hidden, hidden_actual, hidden_expected, ref_primary, ref_aux, ref_aux_weight, actual_head_grads
            if variant == 'future_proximity':
                transaction_count = negatives(run, model, optimizer, raw, context, device)
            before = model.aux_weight.detach().clone()
            grad_norm = torch.nn.utils.clip_grad_norm_([p for _, p in own_items], 1., error_if_nonfinite=True)
            clipped = model.aux_weight.grad.detach().clone()
            # Independent first-Adam-step equation; no native-update equivalence claim.
            expected = before * (1. - .00006) - .00125 * clipped / (clipped.abs() + 1e-6)
            optimizer.step()
            update_report = {'head_update': comparison(model.aux_weight, expected, TOLERANCES['head_update']),
                             'aux_changed': not torch.equal(model.aux_weight, before),
                             'grad_norm_before_clip': float(grad_norm.detach().cpu()),
                             'Muon_groups': len(optimizer.optimizers[0].param_groups),
                             'AdamW_groups': len(optimizer.optimizers[1].param_groups),
                             'gradient_exposure': [name for name, p in own_items if p.grad is not None]}
            publish(run, ident + '-update.json', update_report)
            if not all_ok(update_report) or not update_report['aux_changed'] or len(update_report['gradient_exposure']) != 89:
                raise AssertionError(ident + ' finite update')
    else:
        checks['total_loss'] = comparison(output.total_loss, torch.tensor(primary_loss, dtype=torch.float64), TOLERANCES['loss'], dense=True)
        checks['no_aux'] = output.auxiliary_logits is None and output.auxiliary_loss is None
        checks['caller_preservation'] = {'before': before_math, 'after': snapshot(model, optimizer, raw)}
        checks['passed'] = all_ok(checks) and all(value for value in checks.values() if type(value) is bool) and init['RNG_draws_exact'] and init['native_count_exact'] and checks['caller_preservation']['before'] == checks['caller_preservation']['after']
        publish(run, ident + '-math.json', checks)
        if not all_ok(checks) or not checks['no_aux'] or not init['RNG_draws_exact']:
            raise AssertionError(ident + ' native predicates')
    # Return only raw pre-update logits for declared tail-only pair comparisons.
    pair = (cpu_values(output.primary.values), None if output.auxiliary_logits is None else cpu_values(output.auxiliary_logits))
    del model, reference, output
    gc.collect(); torch.mps.empty_cache()
    return pair, transaction_count, update_report is not None


def numerical_body(run, plan):
    import torch
    torch.set_num_threads(4)
    if torch.get_num_threads() != 4 or sys.byteorder != 'little': raise ValueError('Pinned runtime required')
    caller_cpu, caller_mps = torch.get_rng_state().clone(), torch.mps.get_rng_state().clone()
    counts = {'arms': 0, 'transactions': 0, 'tail_pairs': 0, 'updates': 0}
    try:
        for device in BACKENDS:
            pairs = {}
            for variant, case_name in ARMS:
                case = next(value for value in CASES if value[0] == case_name)
                from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
                with CpuPrerequisiteGuard(run / (device + '-' + variant + '-' + case_name + '-phase-guard.json'), ROOT, timeout_seconds=120, poll_seconds=.25):
                    pair, transactions, updated = run_arm(run, device, variant, case, plan['initializer'])
                gc.collect(); torch.mps.empty_cache()
                counts['arms'] += 1; counts['transactions'] += transactions; counts['updates'] += int(updated)
                if variant == 'future_proximity' and case_name != 'unit_nearest_001': pairs[case_name] = pair
            for index, label in enumerate(('primary', 'auxiliary')):
                evidence = comparison(pairs['repeated_tail_001'][index], pairs['changed_tail_001'][index], TOLERANCES['exact'], dense=True)
                publish(run, device + '-tail-only-' + label + '.json', evidence)
                counts['tail_pairs'] += 1
                if not evidence['passed']: raise AssertionError('Tail entered model body')
        preservation = {'CPU_RNG': torch.equal(caller_cpu, torch.get_rng_state()),
                        'MPS_RNG': torch.equal(caller_mps, torch.mps.get_rng_state()),
                        'counts': counts, 'passed': counts == EXPECTED_REPORTS}
        publish(run, 'preservation-and-counts.json', preservation)
        if not preservation['CPU_RNG'] or not preservation['MPS_RNG'] or not preservation['passed']:
            raise AssertionError('Caller RNG or exact report counts')
    finally:
        torch.set_rng_state(caller_cpu); torch.mps.set_rng_state(caller_mps)
    publish(run, 'result.json', {'status': 'complete_NEW_TOP_model_math_finite_scope_only', 'counts': counts,
                               'no_state_cache_scoring_quality_qualification': True})


def main():
    # A source-only plan cannot enter this code. NEW role census and full runtime
    # pin admission must be independently reviewed before changing any plan.
    if len(sys.argv) != 3: raise ValueError('Exact NEW runtime plan path and SHA required')
    path = Path(sys.argv[1]).resolve(); run = path.parent
    if sha_file(path) != sys.argv[2]: raise ValueError('Plan hash differs')
    plan = json.loads(path.read_text())
    if plan.get('execution_admitted') is not True: raise ValueError('SOURCE ONLY; runtime not admitted')
    if plan['resources'] != RESOURCE_CONTRACT or plan['expected_reports'] != EXPECTED_REPORTS: raise ValueError('Runtime contract differs')
    census = json.loads((ROOT / plan['initializer_census']).read_text())
    if sha_file(ROOT / plan['initializer_census']) != plan['initializer_census_sha256']: raise ValueError('Census pin differs')
    if not (census['accepted'] and census['roles'] == {'initializer': plan['initializer']} and census['prior_matches'] == 0):
        raise ValueError('NEW exclusive unused initializer census required')
    if type(plan['initializer']) is not int or plan['initializer'] <= 0: raise ValueError('Exact new role required')
    if any((run / name).exists() for name in ('attempt.json', 'STOP', 'result.json')): raise ValueError('Closed command; no replay/retry')
    for source, digest in plan['sources'].items():
        if sha_file(ROOT / source) != digest: raise ValueError('Source pins differ')
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        if os.environ.get(name) != '4': raise ValueError('Four-thread environment required')
    for name in ('PYTORCH_ENABLE_MPS_FALLBACK', 'PYTORCH_MPS_FAST_MATH'):
        if os.environ.get(name) != '0': raise ValueError('Strict backend required')
    if shutil.disk_usage(ROOT).free < 11274289152: raise ValueError('Full output/reserve/headroom entry requirement')
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish(run, 'attempt.json', {'pid': os.getpid(), 'plan_sha256': sys.argv[2], 'no_retry': True, 'no_replay': True})
        try:
            from research.block_norm_v1.mps_guard import StrictMpsGuard
            with StrictMpsGuard(run / 'mps-guard.json', ROOT): numerical_body(run, plan)
        except BaseException as error:
            publish(run, 'STOP', {'status': 'closed_failed_NEW_TOP_model_math', 'error': repr(error), 'no_retry': True, 'no_replay': True})
            raise
        else:
            publish(run, 'STOP', {'status': 'complete_NEW_TOP_model_math_finite_scope_only', 'no_retry': True, 'no_replay': True})


if __name__ == '__main__': main()
