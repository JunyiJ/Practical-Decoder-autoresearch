"""Numerical checkpoint prerequisite SOURCE; not admitted for execution yet.

Own model/optimizer/RNG state only. This is NOT a complete training checkpoint:
samplers, data identity, schedule, clipping history and H200 remain later work.
Outer entry owns source/runtime admission, worker lock, resource guards and roles.
"""
import copy
import hashlib
import math
import os
from pathlib import Path
import torch
from research.byte_input_model_v5 import ByteInputModel, ByteInputOptimizer
from research.byte_input_model_v3.contract import require, same_typed

KIND = 'byte_input_numerical_checkpoint_v2'
ROOT = Path(__file__).resolve().parents[1]
FILE_CAP = 512 * 1024**2


def binding(value):
    require(type(value) is dict and set(value) == {'source_sha256', 'runtime_sha256'},
            'Exact admitted source/runtime binding required')
    require(all(type(v) is str and len(v) == 64 and all(c in '0123456789abcdef' for c in v)
                for v in value.values()), 'SHA256 binding fields required')


def live(model, optimizer):
    require(type(model) is ByteInputModel and type(optimizer) is ByteInputOptimizer
            and optimizer.model is model, 'Own model and optimizer owner required')
    optimizer.validate()
    require(all(p.grad is None for _, p in model.ordered_parameters()),
            'Checkpoint only at an explicitly cleared-gradient update boundary')
    require(0 <= optimizer._steps <= 1200, 'Bounded completed update count')
    return dict(model.ordered_parameters())


def clone(value):
    return value.detach().to(device='cpu').contiguous().clone()


def capture(model, optimizer, admitted_binding):
    binding(admitted_binding)
    parameters = live(model, optimizer)
    by_id = {id(p): n for n, p in parameters.items()}
    children = []
    for child in optimizer._optimizers:
        groups = [dict(name=by_id[id(g['params'][0])],
                       options=copy.deepcopy({k: v for k, v in g.items() if k != 'params'}))
                  for g in child.param_groups]
        state = {by_id[id(p)]: {k: clone(v) for k, v in fields.items()}
                 for p, fields in child.state.items()}
        children.append(dict(kind=type(child).__name__, groups=groups, state=state))
    payload = dict(kind=KIND, identity=copy.deepcopy(model.identity),
                   binding=copy.deepcopy(admitted_binding), backend=str(next(iter(parameters.values())).device),
                   training=model.training, completed_steps=optimizer._steps,
                   parameters={n: clone(p) for n, p in parameters.items()},
                   immutable_buffer_hashes=copy.deepcopy(model._own_expected_buffer_hashes),
                   children=children, cpu_rng=torch.random.get_rng_state().clone(),
                   mps_rng=torch.mps.get_rng_state().clone())
    validate(model, optimizer, payload, admitted_binding)
    return payload


def _tensor(value, shape, seen, *, dtype=torch.float32, nonnegative=False):
    require(type(value) is torch.Tensor and value.device.type == 'cpu'
            and value.layout == torch.strided and value.dtype == dtype
            and tuple(value.shape) == tuple(shape) and value.is_contiguous()
            and not value.requires_grad and bool(torch.isfinite(value).all()),
            'Exact finite detached contiguous CPU payload tensor')
    pointer = value.untyped_storage().data_ptr()
    require(pointer not in seen, 'Payload tensors must not alias each other or live CPU state')
    seen.add(pointer)
    if nonnegative:
        require(bool((value >= 0).all()), 'Nonnegative saved Adam second moment')


def validate(model, optimizer, payload, admitted_binding):
    binding(admitted_binding)
    parameters = live(model, optimizer)
    fields = {'kind', 'identity', 'binding', 'backend', 'training', 'completed_steps',
              'parameters', 'immutable_buffer_hashes', 'children', 'cpu_rng', 'mps_rng'}
    require(type(payload) is dict and set(payload) == fields and payload['kind'] == KIND
            and same_typed(payload['identity'], model.identity)
            and same_typed(payload['binding'], admitted_binding)
            and type(payload['backend']) is str
            and payload['backend'] == str(next(iter(parameters.values())).device)
            and type(payload['training']) is bool
            and type(payload['completed_steps']) is int and 0 <= payload['completed_steps'] <= 1200,
            'Exact numerical checkpoint identity, backend and completed step')
    require(same_typed(payload['immutable_buffer_hashes'], model._own_expected_buffer_hashes),
            'Reconstructed codebook and native constant buffers must match exactly')
    require(type(payload['parameters']) is dict and list(payload['parameters']) == list(parameters),
            'Complete ordered own parameter inventory')
    live_tensors = list(parameters.values()) + list(model.buffers())
    live_tensors += [v for child in optimizer._optimizers for state in child.state.values() for v in state.values()]
    seen = {v.untyped_storage().data_ptr() for v in live_tensors if v.device.type == 'cpu'}
    for name, parameter in parameters.items():
        _tensor(payload['parameters'][name], parameter.shape, seen)
    step = payload['completed_steps']
    children = payload['children']
    require(type(children) is list and len(children) == 2, 'Exact two optimizer children')
    base_lr = None
    for index, (saved, child) in enumerate(zip(children, optimizer._optimizers)):
        require(type(saved) is dict and set(saved) == {'kind', 'groups', 'state'}
                and saved['kind'] == type(child).__name__ and type(saved['groups']) is list
                and len(saved['groups']) == len(child.param_groups) and type(saved['state']) is dict,
                'Exact optimizer child schema and group count')
        expected_names = []
        for group, actual in zip(saved['groups'], child.param_groups):
            name = next(n for n, p in parameters.items() if p is actual['params'][0])
            expected_names.append(name)
            require(type(group) is dict and set(group) == {'name', 'options'} and group['name'] == name
                    and type(group['options']) is dict, 'Ordered canonical optimizer group')
            options = group['options']
            expected = {k: v for k, v in actual.items() if k not in ('params', 'lr')}
            require(set(options) == set(expected) | {'lr'} and
                    same_typed({k: v for k, v in options.items() if k != 'lr'}, expected),
                    'Unchanged optimizer numerical policy')
            rate = options['lr']
            require(type(rate) is float and math.isfinite(rate) and rate >= 0., 'Finite scheduled LR')
            scale = .5 if index == 0 else 1.
            if base_lr is None: base_lr = rate / scale
            require(rate / scale == base_lr, 'One shared scheduled base LR')
        require(list(saved['state']) == (expected_names if step else []),
                'Complete ordered active state, or no state before first update')
        for name, state in saved['state'].items():
            wanted = {'momentum_buffer'} if index == 0 else {'step', 'exp_avg', 'exp_avg_sq'}
            require(type(state) is dict and set(state) == wanted, 'Exact optimizer state fields')
            for key, value in state.items():
                _tensor(value, () if key == 'step' else parameters[name].shape, seen,
                        nonnegative=key == 'exp_avg_sq')
                if key == 'step': require(value.item() == step, 'Every Adam step must agree')
    current_cpu = torch.random.get_rng_state()
    current_mps = torch.mps.get_rng_state().clone()
    _tensor(payload['cpu_rng'], current_cpu.shape, seen, dtype=torch.uint8)
    _tensor(payload['mps_rng'], current_mps.shape, seen, dtype=torch.uint8)
    try:
        torch.Generator(device='cpu').set_state(payload['cpu_rng'])
        torch.mps.set_rng_state(payload['mps_rng'])
    except RuntimeError as exc:
        raise ValueError('Checkpoint RNG encoding is invalid') from exc
    finally:
        torch.mps.set_rng_state(current_mps)
    require(torch.equal(torch.random.get_rng_state(), current_cpu)
            and torch.equal(torch.mps.get_rng_state(), current_mps), 'RNG validation must preserve caller streams')
    return step


def restore_pristine(model, optimizer, payload, admitted_binding):
    """Load an unexposed fresh destination; all payload checks precede mutation.

    Invalid input preserves the destination. Resource/driver failures during a
    valid commit require discarding that fresh destination, never publishing it.
    Success intentionally restores checkpoint CPU/MPS RNG, not caller RNG.
    """
    require(type(optimizer) is ByteInputOptimizer and optimizer._steps == 0
            and all(not child.state for child in optimizer._optimizers), 'Pristine destination optimizer only')
    step = validate(model, optimizer, payload, admitted_binding)
    parameters = dict(model.ordered_parameters())
    device = next(iter(parameters.values())).device
    # Allocate complete destination state before any commit. Own model copies
    # and optimizer state are separate allocations, including CPU restore.
    staged_parameters = {n: value.to(device=device).clone() for n, value in payload['parameters'].items()}
    staged_children = []
    for saved in payload['children']:
        staged_children.append({name: {key: value.to(device='cpu' if key == 'step' else device).clone()
                                        for key, value in state.items()}
                                for name, state in saved['state'].items()})
    with torch.no_grad():
        for name, parameter in parameters.items(): parameter.copy_(staged_parameters[name])
    for child, saved, states in zip(optimizer._optimizers, payload['children'], staged_children):
        for group, record in zip(child.param_groups, saved['groups']): group['lr'] = record['options']['lr']
        for name, state in states.items(): child.state[parameters[name]] = state
    optimizer._steps = step
    model.train(payload['training'])
    optimizer.validate()
    torch.mps.set_rng_state(payload['mps_rng'])
    torch.random.set_rng_state(payload['cpu_rng'])
    return step


def _path(value):
    p = Path(value)
    require(p.is_absolute() and p == p.resolve() and p.is_relative_to(ROOT/'research_runs')
            and p.parent.is_dir() and not any(x.is_symlink() for x in (p, *p.parents)),
            'Canonical own research-run path required')
    return p


def _sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        while block := stream.read(1048576): h.update(block)
    return h.hexdigest()


def write_new(path, payload):
    """Caller must validate the complete numerical payload immediately before I/O."""
    require(type(payload) is dict and payload.get('kind') == KIND, 'Own numerical payload only')
    path = _path(path); temporary = path.with_suffix(path.suffix + '.tmp')
    require(not path.exists() and not temporary.exists(), 'Exclusive new artifact only')
    class Bounded:
        def __init__(self, stream): self.stream = stream
        def write(self, value):
            require(self.stream.tell() + len(value) <= FILE_CAP, 'Complete serialization cap')
            return self.stream.write(value)
        def flush(self): return self.stream.flush()
        def tell(self): return self.stream.tell()
    with temporary.open('xb') as stream:
        torch.save(payload, Bounded(stream)); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path)  # Exclusive publication; cannot overwrite a retained artifact.
    temporary.unlink()       # Only this successful operation's own temporary link.
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return dict(kind=KIND, path=str(path), size=path.stat().st_size, sha256=_sha(path))


def read_bound(receipt):
    require(type(receipt) is dict and set(receipt) == {'kind', 'path', 'size', 'sha256'}
            and receipt['kind'] == KIND and type(receipt['path']) is str
            and type(receipt['size']) is int and 0 < receipt['size'] <= FILE_CAP
            and type(receipt['sha256']) is str and len(receipt['sha256']) == 64
            and all(c in '0123456789abcdef' for c in receipt['sha256']), 'Exact bounded archive receipt')
    path = _path(receipt['path'])
    require(path.is_file() and path.stat().st_nlink == 1 and path.stat().st_size == receipt['size']
            and _sha(path) == receipt['sha256'], 'Exact retained archive before read')
    value = torch.load(path, map_location='cpu', weights_only=True)
    require(path.stat().st_size == receipt['size'] and _sha(path) == receipt['sha256'], 'Archive changed during read')
    return value  # Still requires validate(...), even after a successful hash-bound read.
