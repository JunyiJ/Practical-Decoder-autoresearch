"""Tiny native/linear/tanh readout functions, intentionally excluding model state."""
import torch
from torch.nn import functional as F

KINDS=('native','linear_projection1','tanh_projection1')

def require(value,message):
    if not value:raise ValueError(message)

def _inputs(kind,z,w,a):
    require(type(kind) is str and kind in KINDS,'Explicit single-projection kind required')
    require(type(z) is torch.Tensor and type(w) is torch.Tensor,'Literal tensor inputs required')
    require(z.ndim==w.ndim==2,'Two-dimensional features/head required')
    n,d=z.shape;v,dw=w.shape
    require(0<n<=16 and 0<d<=8 and 1<v<=31 and dw==d,'Tiny generated shapes only')
    if kind=='native':require(a is None,'Native has no projection')
    else:require(type(a) is torch.Tensor and a.ndim==2 and tuple(a.shape)==(d,d),'One exact square projection required')
    values=[z,w]+([] if a is None else [a])
    require(z.device.type in ('cpu','mps') and z.dtype in (torch.float32,torch.float64),'CPU/MPS FP32 or CPU FP64 required')
    require(z.device.type!='mps' or z.dtype==torch.float32,'MPS FP32 only')
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'),'Autocast forbidden')
    require(all(t.numel()<=4096 and t.dtype==z.dtype and t.device==z.device for t in values),'Tiny same-device/same-dtype inputs required')
    require(all(bool(torch.isfinite(t).all()) for t in values),'Finite inputs required')
    return n,d,v

def logits(kind,z,w,a=None):
    _inputs(kind,z,w,a)
    h=z if kind=='native' else F.linear(z,a)
    if kind=='tanh_projection1':h=torch.tanh(h)
    return F.linear(h,w)/4.

def log_probs(kind,z,w,a=None):
    return F.log_softmax(logits(kind,z,w,a),dim=-1)

def nll(kind,z,w,a,targets,*,reduction='mean'):
    n,d,v=_inputs(kind,z,w,a)
    require(type(targets) is torch.Tensor and targets.dtype==torch.int64 and targets.device==z.device
            and tuple(targets.shape)==(n,),'Exact device-local target vector required')
    require(type(reduction) is str and reduction in ('mean','none'),'Explicit supported reduction required')
    require(bool(((targets>=0)&(targets<v)).all()),'Targets outside tiny vocabulary')
    return F.cross_entropy(logits(kind,z,w,a),targets,reduction=reduction)
