"""Isolated tiny functional FFNs; no model/state/training admission."""
import torch
from torch.nn import functional as F

KINDS = ('native_swiglu', 'ungated_silu', 'ungated_relu_squared')


def forward(kind, x, up, down, gate=None):
    if type(kind) is not str or kind not in KINDS:
        raise ValueError('Declared FFN equation required')
    values = (x, up, down) + ((gate,) if kind == 'native_swiglu' else ())
    if kind != 'native_swiglu' and gate is not None:
        raise ValueError('Ungated equation has no independent gate')
    if any(type(v) is not torch.Tensor for v in values):
        raise ValueError('Explicit functional tensors required')
    if x.ndim != 2 or not 1 <= x.shape[0] <= 16 or x.shape[1] not in (2, 4, 6, 8):
        raise ValueError('Tiny N<=16, even D2..8 only')
    d = x.shape[1]
    h = d if kind == 'native_swiglu' else 3*d//2
    if up.shape != (h, d) or down.shape != (d, h):
        raise ValueError('Exact matched matrix allowance required')
    if kind == 'native_swiglu' and gate.shape != (h, d):
        raise ValueError('Exact native gate shape required')
    if x.device.type not in ('cpu', 'mps') or x.dtype not in (torch.float32, torch.float64):
        raise ValueError('CPU64/CPU32/MPS32 only')
    if x.device.type == 'mps' and x.dtype != torch.float32:
        raise ValueError('MPS32 only')
    if torch.is_autocast_enabled('cpu') or torch.is_autocast_enabled('mps'):
        raise ValueError('Autocast forbidden')
    if any(v.device != x.device or v.dtype != x.dtype or not bool(torch.isfinite(v).all())
           for v in values):
        raise ValueError('Finite same-device/dtype literals required')
    u = F.linear(x, up)
    if not bool(torch.isfinite(u).all()):
        raise ValueError('Up projection overflow')
    if kind == 'native_swiglu':
        g = F.linear(x, gate)
        if not bool(torch.isfinite(g).all()):
            raise ValueError('Gate projection overflow')
        hidden = F.silu(g)*u
    elif kind == 'ungated_silu':
        hidden = F.silu(u)
    else:
        hidden = F.relu(u).square()
    if not bool(torch.isfinite(hidden).all()):
        raise ValueError('Hidden overflow; no fallback/clamping')
    output = F.linear(hidden, down)
    if not bool(torch.isfinite(output).all()):
        raise ValueError('Down projection overflow')
    return output
