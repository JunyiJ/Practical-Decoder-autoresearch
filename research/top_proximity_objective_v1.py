"""Own supplied-logit proximity objective; SOURCE ONLY, not a model/scorer.

Targets are immutable flattened B*T rows of (ID, Python-float mass) pairs.
Masses are cast once to the logits' dtype/device and are NOT renormalized.
No gradients are assigned, cleared or backwarded by this function.
"""
import math
import torch
from torch.nn import functional as F


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def sparse_proximity_ce(logits, targets):
    """Mean weighted negative log probability over every B*T input position.

    Python targets must have positive finite masses summing to1 within1e-14.
    For the actual represented weights w, the exact-real derivative is
    (sum(w)*softmax(logits)-w)/(B*T). FP32 casting can change sum(w), and
    sufficiently small positive Python masses can round to zero. No hidden
    normalization, dtype promotion, target-gradient path or objective mixing.
    Supplied logits do not establish causal model or future-target provenance.
    """
    _require(type(logits) is torch.Tensor, 'Plain logits tensor required')
    _require(logits.layout == torch.strided and logits.device.type in ('cpu', 'mps')
             and logits.ndim == 3 and logits.dtype in (torch.float32, torch.float64),
             'Dense CPU64/CPU32/MPS32 logits required')
    _require(logits.device.type != 'mps' or logits.dtype == torch.float32,
             'MPS FP32 required')
    _require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'),
             'Autocast forbidden')
    batch, time, vocabulary = logits.shape
    _require(1 <= batch <= 8 and 1 <= time <= 128 and 2 <= vocabulary <= 50257,
             'Bounded positive B/T/V dimensions required')
    _require(bool(torch.isfinite(logits).all()), 'Finite supplied logits required')
    positions = batch * time
    _require(type(targets) is tuple and len(targets) == positions,
             'One immutable sparse target row per B*T input required')
    coordinates, masses = [], []
    for index, row in enumerate(targets):
        _require(type(row) is tuple and 1 <= len(row) <= min(128, vocabulary),
                 'Nonempty bounded immutable target row required')
        ids, row_masses = set(), []
        for pair in row:
            _require(type(pair) is tuple and len(pair) == 2, 'Exact immutable ID/mass pair required')
            token, mass = pair
            _require(type(token) is int and 0 <= token < vocabulary and token not in ids,
                     'Distinct ordinary in-vocabulary target IDs required')
            _require(type(mass) is float and math.isfinite(mass) and 0. < mass <= 1.,
                     'Positive finite Python-float target mass required')
            ids.add(token)
            row_masses.append(mass)
            coordinates.append((index, token))
            masses.append(mass)
        _require(abs(math.fsum(row_masses) - 1.) <= 1e-14,
                 'Python target row mass must sum to1 within1e-14')
    # All input/target validation precedes staging or differentiable operations.
    indices = torch.tensor(coordinates, dtype=torch.int64, device=logits.device)
    represented = torch.tensor(masses, dtype=logits.dtype, device=logits.device)
    log_probabilities = F.log_softmax(logits, dim=-1).reshape(positions, vocabulary)
    selected = log_probabilities[indices[:, 0], indices[:, 1]]
    loss = -(selected * represented).sum() / positions
    _require(loss.ndim == 0 and loss.dtype == logits.dtype and bool(torch.isfinite(loss))
             and bool(loss >= 0), 'Finite nonnegative same-dtype scalar loss required')
    return loss
