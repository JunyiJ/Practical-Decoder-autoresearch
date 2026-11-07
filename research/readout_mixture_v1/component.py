"""Tiny explicit native/MoC/MoS equations, deliberately excluding model tensors."""
import torch
from torch.nn import functional as F

KINDS = ('native', 'context_mix2', 'probability_mix2')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def validate(kind, z, w, a, r, targets=None, reduction='none'):
    require(type(kind) is str and kind in KINDS, 'Explicit distribution kind required')
    require(type(z) is torch.Tensor and type(w) is torch.Tensor, 'Plain generated tensors required')
    require(z.ndim == w.ndim == 2 and 0 < z.shape[0] <= 16 and 0 < z.shape[1] <= 8,
            'Tiny N<=16,D<=8 required; actual model tensors forbidden')
    n, d = z.shape
    require(1 < w.shape[0] <= 31 and w.shape[1] == d, 'Tiny V<=31 matching width required')
    tensors = [z, w]
    if kind == 'native':
        require(a is None and r is None, 'Native arm has no extra context/router tensors')
    else:
        require(type(a) is torch.Tensor and type(r) is torch.Tensor and a.shape == (2, d, d)
                and r.shape == (2, d), 'Exactly two context maps and router rows required')
        tensors += [a, r]
    require(z.device.type in ('cpu', 'mps') and (z.dtype == torch.float32
            or (z.device.type == 'cpu' and z.dtype == torch.float64)), 'Declared CPU64/CPU32/MPS32 only')
    require(not torch.is_autocast_enabled(z.device.type), 'Autocast forbidden')
    for t in tensors:
        require(t.device == z.device and t.dtype == z.dtype and t.layout == torch.strided
                and t.numel() <= 4096 and bool(torch.isfinite(t).all()), 'Matching finite bounded tensors required')
    require(type(reduction) is str and reduction in ('none', 'mean'), 'Explicit reduction required')
    if targets is not None:
        require(type(targets) is torch.Tensor and targets.shape == (n,) and targets.dtype == torch.int64
                and targets.device == z.device and bool(((targets >= 0) & (targets < w.shape[0])).all()),
                'Matching int64 native targets required')


def contexts(z, a, r):
    return [torch.tanh(F.linear(z, a[k])) for k in range(2)], F.log_softmax(F.linear(z, r), dim=-1)


def finite(value):
    require(bool(torch.isfinite(value).all()), 'Finite computed distribution/loss required')
    return value


def log_probs(kind, z, w, a=None, r=None):
    validate(kind, z, w, a, r)
    if kind == 'native':
        return finite(F.log_softmax(F.linear(z, w) / 4, dim=-1))
    h, lp = contexts(z, a, r)
    if kind == 'context_mix2':
        hbar = lp[:, 0:1].exp() * h[0] + lp[:, 1:2].exp() * h[1]
        return finite(F.log_softmax(F.linear(hbar, w) / 4, dim=-1))
    parts = [lp[:, k:k+1] + F.log_softmax(F.linear(h[k], w) / 4, dim=-1) for k in range(2)]
    return finite(torch.logaddexp(parts[0], parts[1]))


def nll(kind, z, w, targets, a=None, r=None, reduction='none'):
    validate(kind, z, w, a, r, targets, reduction)
    require(targets is not None, 'Targets required for NLL')
    if kind == 'native':
        return finite(F.cross_entropy(F.linear(z, w) / 4, targets, reduction=reduction))
    h, lp = contexts(z, a, r)
    if kind == 'context_mix2':
        hbar = lp[:, 0:1].exp() * h[0] + lp[:, 1:2].exp() * h[1]
        return finite(F.cross_entropy(F.linear(hbar, w) / 4, targets, reduction=reduction))
    # Only [N,2] target terms; never construct full mixture [N,V] during training.
    terms = torch.stack([lp[:, k] - F.cross_entropy(F.linear(h[k], w) / 4, targets,
                                                   reduction='none') for k in range(2)], dim=1)
    result = -torch.logsumexp(terms, dim=1)
    return finite(result if reduction == 'none' else result.mean())
