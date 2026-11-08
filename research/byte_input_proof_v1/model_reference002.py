"""Independent composition/integration equations; functional proof not admitted.

Native blocks are the unchanged trusted consumer. This checks composition and
integration, not a new independent proof of the adopted attention equations.
No candidate methods, historical failed proof, or historical oracle is called.
"""
import torch
from torch.nn import functional as F


def compose(parameters, codes, lengths):
    # Derive windows through explicit slices, independently of candidate unfold.
    # Use the declared fused FP32 linear primitive: algebraically equivalent
    # Conv1d/separate-bias arithmetic can choose different ReLU branches near
    # zero, which is not an implementation-gradient correctness comparison.
    position = torch.arange(codes.shape[1], device=codes.device)[None, :]
    valid = position < lengths[:, None]
    index = torch.where(codes < 257, codes, codes - 1)
    embedded = parameters['byte_weight'][index]
    embedded = torch.where(valid[:, :, None], embedded, torch.zeros_like(embedded))
    features = []
    for j, channels in enumerate((25, 50, 75, 100, 125, 150)):
        width = j + 1
        weights = parameters['filters.' + str(j)]
        bias = parameters['filter_biases.' + str(j)]
        count = embedded.shape[1] - width + 1
        windows = torch.stack([embedded[:, offset:offset+count] for offset in range(width)], dim=-1)
        windows = windows.reshape(codes.shape[0], count, 15*width)
        activation = torch.tanh(F.linear(windows, weights, bias))
        allowed = torch.arange(count, device=codes.device)[None, :] <= lengths[:, None] - width
        maximum = torch.where(allowed[:, :, None], activation,
                              torch.full_like(activation, -2.)).amax(1)
        features.append(torch.where((lengths >= width)[:, None], maximum, torch.zeros_like(maximum)))
    y = torch.cat(features, dim=1)
    transformed = torch.relu(F.linear(y, parameters['highway_weight'], parameters['highway_bias']))
    gate = torch.sigmoid(F.linear(y, parameters['transform_weight'], parameters['transform_bias']))
    combined = gate * transformed + (1 - gate) * y
    return F.linear(combined, parameters['projection_weight'])


def primary(reference_native, composer_parameters, codes, lengths, inputs, labels):
    if composer_parameters is None:
        hidden = reference_native.transformer.wte(inputs)
    else:
        rows = inputs.reshape(-1)
        hidden = compose(composer_parameters, codes[rows], lengths[rows]).reshape(*inputs.shape, 256)
    for layer in reference_native.transformer.h:
        hidden, auxiliary = layer(hidden, None, 0)
        assert auxiliary is None
    hidden = reference_native.transformer.ln_f(hidden)
    logits = (hidden @ reference_native.lm_head.weight.T) * 0.25
    # Explicit logsumexp/gather checks ordinary CE independently of F.cross_entropy.
    normalizer = torch.logsumexp(logits, dim=-1)
    chosen = logits.gather(-1, labels[:, :, None]).squeeze(-1)
    return logits, (normalizer - chosen).mean()


def parameter_inventory(native, composer_parameters):
    result = dict(native.named_parameters())
    if composer_parameters is not None:
        del result['transformer.wte.weight']
        result.update(('composer.' + name, value) for name, value in composer_parameters.items())
    return result


def optimizer(parameters, base_lr):
    # Explicit expected common matrices, independent of candidate grouping helper.
    muon_names = set()
    for layer in range(4):
        prefix = 'transformer.h.' + str(layer) + '.'
        muon_names.update(name for name, p in parameters.items()
                          if name.startswith(prefix) and len(p.shape) == 2)
    muon_names.update('composer.filters.' + str(j) for j in range(6))
    muon_names.update(('composer.highway_weight', 'composer.transform_weight',
                       'composer.projection_weight'))
    muon, adam = [], []
    for name, parameter in parameters.items():
        use_muon = name in muon_names
        group = dict(params=[parameter], lr=base_lr * (0.5 if use_muon else 1.), weight_decay=0.)
        (muon if use_muon else adam).append(group)
    return (torch.optim.Muon(muon, adjust_lr_fn='match_rms_adamw'),
            torch.optim.AdamW(adam, betas=(0.9, 0.999), eps=1e-6))


def update(parameters, children):
    # Same qualified library kernels; this tests wrapper ordering/grouping,
    # complete state and independent decay, not the library optimizer equations.
    with torch.no_grad():
        for parameter in parameters.values():
            parameter.mul_(0.99994)
    for child in children:
        child.step()
