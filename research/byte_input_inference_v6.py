"""Averaged-weight export and uncached validation consumer for model-v3.

No training dispatch or scientific selection. The caller owns source admission,
the shared worker lock and resource guards. Expected identity/parent come from
the admitted training plan, never from an untrusted export's self-description.
"""
import copy
import math
import os
import torch
from torch.nn import functional as F
from research import byte_input_training_v7 as training
from research import byte_input_state_v7 as numerical
from research.byte_input_model_v7 import ByteInputModel, Spec
from research.byte_input_model_v3.contract import require, same_typed
from research.final_ln_scale_worker_v1.validation import target_contract, validate_report

KIND = 'byte_input_h200_inference_v1'
FILE_CAP = 128 * 1024**2


def receipt(value, kind, cap):
    require(type(value) is dict and set(value) == {'kind', 'path', 'size', 'sha256'}
            and value['kind'] == kind and type(value['size']) is int
            and 0 < value['size'] <= cap and type(value['path']) is str
            and type(value['sha256']) is str and len(value['sha256']) == 64
            and all(c in '0123456789abcdef' for c in value['sha256']), 'Exact own artifact receipt')
    return numerical._path(value['path'])


def identity(value):
    require(type(value) is dict and set(value) == {'kind', 'recipe', 'roles', 'binding', 'data', 'model'}
            and value['kind'] == training.KIND, 'Exact expected training identity')
    training.roles(value['roles']); numerical.binding(value['binding'])
    config = value['recipe']
    require(type(config) is dict and same_typed(config, training.recipe(config.get('max_iters'), config.get('learning_rate'))),
            'Original training recipe required')
    return config['max_iters'], config['weight_averaging']


def validate(model, payload, expected_identity, expected_parent):
    require(type(model) is ByteInputModel, 'Own model destination')
    model.validate()
    horizon, averaging = identity(expected_identity)
    receipt(expected_parent, training.KIND, numerical.FILE_CAP)
    require(type(payload) is dict and set(payload) == {'kind', 'identity', 'parent', 'completed_steps',
            'averaging', 'parameters', 'immutable_buffer_hashes'} and payload['kind'] == KIND
            and same_typed(payload['identity'], expected_identity)
            and same_typed(payload['parent'], expected_parent)
            and same_typed(expected_identity['model'], model.identity)
            and type(payload['completed_steps']) is int and payload['completed_steps'] == horizon,
            'Only complete horizon H200 export with exact parent/identity')
    expected = dict(half_life=200, spec=averaging,
                    snapshots=1+(horizon-averaging['start_step'])//averaging['interval'])
    require(same_typed(payload['averaging'], expected), 'Exact original H200 metadata')
    require(same_typed(payload['immutable_buffer_hashes'], model._own_expected_buffer_hashes),
            'Reconstructed codebook/native buffers differ')
    parameters = dict(model.ordered_parameters())
    require(type(payload['parameters']) is dict and list(payload['parameters']) == list(parameters),
            'Complete ordered averaged weights')
    seen = {t.untyped_storage().data_ptr() for t in [*parameters.values(), *model.buffers()] if t.device.type == 'cpu'}
    for name, parameter in parameters.items():
        numerical._tensor(payload['parameters'][name], parameter.shape, seen)
    return parameters


def export(context, parent_receipt):
    """Read and validate the complete RAW parent; do not alter the live context."""
    require(type(context) is training.Context, 'Own validating training context')
    parent = training.read_bound(parent_receipt)
    step = training.validate(context, parent)
    require(step == context.recipe['max_iters'], 'Only completed fixed horizon may be exported')
    average = parent['averages']
    value = dict(kind=KIND, identity=copy.deepcopy(context._identity), parent=copy.deepcopy(parent_receipt),
        completed_steps=step, averaging=dict(half_life=200, spec=copy.deepcopy(average['spec']), snapshots=average['snapshots']),
        parameters={n: numerical.clone(t) for n,t in average['averages']['200'].items()},
        immutable_buffer_hashes=copy.deepcopy(parent['numerical']['immutable_buffer_hashes']))
    validate(context.model, value, context._identity, parent_receipt)
    return value


def write_new(path, value):
    """Caller must validate immediately before this bounded exclusive write."""
    require(type(value) is dict and value.get('kind') == KIND, 'Own inference payload')
    path = numerical._path(path); temporary = path.with_suffix(path.suffix+'.tmp')
    require(not path.exists() and not temporary.exists(), 'Exclusive new export')
    class Bounded:
        def __init__(self, stream): self.stream = stream
        def write(self, data):
            require(self.stream.tell()+len(data) <= FILE_CAP, 'Inference serialization cap')
            return self.stream.write(data)
        def flush(self): return self.stream.flush()
        def tell(self): return self.stream.tell()
    with temporary.open('xb') as stream:
        torch.save(value, Bounded(stream)); stream.flush(); os.fsync(stream.fileno())
    os.link(temporary, path); temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)
    return dict(kind=KIND, path=str(path), size=path.stat().st_size, sha256=numerical._sha(path))


def read_bound(record):
    path = receipt(record, KIND, FILE_CAP)
    require(path.is_file() and path.stat().st_nlink == 1 and path.stat().st_size == record['size']
            and numerical._sha(path) == record['sha256'], 'Exact inference artifact before read')
    result = torch.load(path, map_location='cpu', weights_only=True)
    require(path.stat().st_size == record['size'] and numerical._sha(path) == record['sha256'],
            'Inference artifact unchanged during read')
    return result


def load(record, expected_identity, expected_parent, codebook, backend):
    """Reconstruct historical identity only to load all weights; preserve RNG.

    Constructor draws are not a new training initializer role or a seed search.
    The returned eval model remains uncached; no optimizer is loaded or exposed.
    """
    identity(expected_identity)
    require(backend in ('cpu', 'mps'), 'Declared CPU/MPS consumer')
    value = read_bound(record)
    signature = expected_identity['model']
    require(type(signature) is tuple and len(signature) == 7, 'Complete model signature')
    spec = Spec(signature[1], signature[2], signature[3])
    old_mps = torch.mps.get_rng_state().clone()
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(expected_identity['roles']['native_initializer'])
            model = ByteInputModel(spec, codebook)
    finally:
        torch.mps.set_rng_state(old_mps)
    parameters = validate(model, value, expected_identity, expected_parent)
    # No caller-owned model is mutated; allocation/copy failures discard this one.
    with torch.no_grad():
        for name, parameter in parameters.items(): parameter.copy_(value['parameters'][name])
    model.to(backend).eval(); model.validate()
    return model


def _logits(model, inputs):
    """Same accepted model-v3 forward, within a validated read-only scoring pass."""
    model._ids(inputs, next(model.parameters()).device)
    if model.spec.variant == 'native':
        result, _ = model.native(inputs.contiguous())
    else:
        result = model.readout(model._composed_hidden(inputs.contiguous())) / 4.
    require(result.dtype == torch.float32 and tuple(result.shape) == (*inputs.shape, 50257)
            and bool(torch.isfinite(result).all()), 'Finite ordinary primary logits')
    return result


def score(model, view, expected_identity, artifact_receipt):
    """Original full validation windows and FP32 CE/FP64 aggregation.

    Validate expensive immutable model/codebook state at both ends of this
    synchronous, read-only pass. Per-window ID and output checks remain active.
    Expose per-target losses for independent coverage and aggregation review.
    """
    identity(expected_identity); receipt(artifact_receipt, KIND, FILE_CAP)
    training.check_view(view, expected_identity['roles'])
    require(type(model) is ByteInputModel and not model.training
            and same_typed(model.identity, expected_identity['model'])
            and same_typed(view.data_identity, expected_identity['data']), 'Bound eval model and original data')
    device = model.validate()
    target = target_contract(view); ids = view.val_data; losses = []
    with torch.no_grad():
        for start in range(0, len(ids)-1, 128):
            count = min(128, len(ids)-1-start)
            inputs = ids[start:start+count].unsqueeze(0).to(device)
            labels = ids[start+1:start+count+1].to(device)
            logits = _logits(model, inputs)
            values = F.cross_entropy(logits[0], labels, reduction='none').cpu().to(torch.float64)
            require(bool(torch.isfinite(values).all()) and bool((values >= 0).all()), 'Finite per-target CE')
            losses.append(values)
    loss = torch.cat(losses); require(loss.numel() == target['targets'], 'Complete target coverage')
    total = float(loss.sum())
    report = dict(binding=copy.deepcopy(artifact_receipt), target=target,
        window_loss_sums=[float(v.sum()) for v in losses],
        block_loss_sums=[float(loss[i:i+2048].sum()) for i in range(0, len(loss), 2048)],
        total_loss=total, ce=total/target['targets'], bits_per_byte=total/math.log(2)/target['target_bytes'],
        loss_dtype='float32_cross_entropy_float64_aggregation')
    validate_report(report, artifact_receipt, view)
    model.validate(); training.check_view(view, expected_identity['roles'])
    return dict(report=report, per_target_losses=loss.tolist())
