"""Bounded literal native-vocabulary CE probe; no model/scorer integration."""
import torch
from torch.nn import functional as F

VOCAB = 50257
MAX_ROWS = 6


def losses(logits, targets):
    """Genuine per-row CE with transactional input preflight; no state or RNG."""
    if type(logits) is not torch.Tensor or type(targets) is not torch.Tensor:
        raise ValueError('Plain tensors required')
    if (logits.layout != torch.strided or targets.layout != torch.strided
            or logits.device.type not in ('cpu', 'mps') or targets.device != logits.device):
        raise ValueError('Dense same-device CPU/MPS tensors required')
    if (logits.dtype not in (torch.float32, torch.float64)
            or (logits.device.type == 'mps' and logits.dtype != torch.float32)
            or targets.dtype != torch.int64):
        raise ValueError('CPU FP64/FP32 or MPS FP32 logits and int64 targets required')
    if (logits.ndim != 2 or not 1 <= logits.shape[0] <= MAX_ROWS
            or logits.shape[1] != VOCAB or targets.shape != (logits.shape[0],)
            or not logits.is_contiguous() or not targets.is_contiguous()):
        raise ValueError('Bounded contiguous native-vocabulary rows required')
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('mps'):
        raise ValueError('Autocast forbidden')
    if (not bool(torch.isfinite(logits).all())
            or not bool(((targets >= 0) & (targets < VOCAB)).all())):
        raise ValueError('Finite logits and in-range targets required')
    result = F.cross_entropy(logits, targets, reduction='none')
    if (result.dtype != logits.dtype or result.shape != targets.shape
            or not bool(torch.isfinite(result).all()) or not bool((result >= 0).all())):
        raise ValueError('Finite nonnegative genuine CE required')
    return result
