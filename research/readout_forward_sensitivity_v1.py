"""Bounded functional diagnostic; no model, optimizer or checkpoint interface."""
import torch
from torch.nn import functional as F

ARMS = ('native', 'linear_projection1', 'tanh_projection1')


def evaluate(x, w, a, gamma, beta, targets, arm, *, normalize):
    if type(arm) is not str or arm not in ARMS or type(normalize) is not bool:
        raise ValueError('Explicit diagnostic arm and normalization control required')
    values = (x, w, a, gamma, beta)
    if any(type(v) is not torch.Tensor for v in values):
        raise ValueError('Literal tensors required')
    if x.ndim != 2 or not 1 <= x.shape[0] <= 16 or x.shape[1] != 256:
        raise ValueError('Diagnostic N<=16,D256 only')
    if w.ndim != 2 or not 2 <= w.shape[0] <= 31 or w.shape[1] != 256:
        raise ValueError('Diagnostic vocabulary2..31 only')
    if a.shape != (256, 256) or gamma.shape != (256,) or beta.shape != (256,):
        raise ValueError('Exact diagnostic matrix/affine shapes required')
    if x.device.type not in ('cpu', 'mps') or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('CPU64/CPU32/MPS32 only')
    if x.device.type == 'mps' and x.dtype != torch.float32:
        raise ValueError('MPS FP32 only')
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('mps'):
        raise ValueError('Autocast forbidden')
    if any(v.numel() > 65536 or v.device != x.device or v.dtype != x.dtype
           or v.requires_grad or not bool(torch.isfinite(v).all()) for v in values):
        raise ValueError('Bounded finite same-device/dtype no-gradient literals required')
    if (type(targets) is not torch.Tensor or targets.shape != (x.shape[0],)
            or targets.dtype != torch.int64 or targets.device != x.device
            or not bool(((targets >= 0) & (targets < w.shape[0])).all())):
        raise ValueError('Exact literal diagnostic targets required')
    with torch.no_grad():
        z = F.layer_norm(x, (256,), gamma, beta, 1e-5) if normalize else x
        u = z if arm == 'native' else F.linear(z, a)
        h = torch.tanh(u) if arm == 'tanh_projection1' else u
        logits = F.linear(h, w)/4.
        losses = F.cross_entropy(logits, targets, reduction='none')
    return dict(features=z, projected=u, transformed=h, logits=logits, nll=losses)
