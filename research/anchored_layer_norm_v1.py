"""SOURCE ONLY: bounded equivalent-real-arithmetic LayerNorm formulation.

No frozen model imports this candidate. Forward/backward runtime is untested.
"""
import torch
from torch.nn import functional as F


def layer_norm(x, gamma, beta):
    if any(type(v) is not torch.Tensor for v in (x, gamma, beta)):
        raise ValueError('Explicit functional tensors required')
    if x.ndim != 2 or not 1 <= x.shape[0] <= 16 or x.shape[1] != 256:
        raise ValueError('New isolated N<=16,D256 scope only')
    if gamma.shape != (256,) or beta.shape != (256,):
        raise ValueError('Exact affine vectors required')
    if x.device.type not in ('cpu', 'mps') or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('CPU64/CPU32/MPS32 only')
    if x.device.type == 'mps' and x.dtype != torch.float32:
        raise ValueError('MPS float32 only')
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('mps'):
        raise ValueError('Autocast forbidden')
    if any(v.device != x.device or v.dtype != x.dtype or v.numel() > 65536
           or not bool(torch.isfinite(v).all()) for v in (x, gamma, beta)):
        raise ValueError('Bounded finite same-device/dtype literals required')
    centered = x-x[:, :1]
    if not bool(torch.isfinite(centered).all()):
        raise ValueError('Anchor subtraction overflow; no fallback or clamping')
    return F.layer_norm(centered, (256,), gamma, beta, 1e-5)
