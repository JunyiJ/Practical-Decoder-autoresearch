"""Bounded byte lookup with repeatable first-order MPS gradient accumulation.

Forward and CPU backward keep native embedding semantics. MPS sums every fixed
row in ordered 2048-position chunks, without repeated-index atomic scatter.
This primitive does not change tokenizer codes, parameters or initialization.
"""
import torch
from torch.nn import functional as F
from torch.autograd.function import once_differentiable

KIND = 'rsi_byte_embedding_grouped_mps_v1'
CHUNK = 2048


def require(value, reason):
    if not value: raise ValueError(reason)


class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, rows):
        ctx.save_for_backward(rows)
        return F.embedding(rows, weight)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        (rows,) = ctx.saved_tensors
        require(upstream.dtype == torch.float32 and upstream.device == rows.device
                and tuple(upstream.shape) == (*rows.shape, 15)
                and bool(torch.isfinite(upstream).all()), 'Finite exact byte lookup upstream required')
        indices = rows.reshape(-1)
        gradients = upstream.reshape(-1, 15)
        result = torch.zeros((259, 15), dtype=torch.float32, device=rows.device)
        vocabulary = torch.arange(259, device=rows.device)
        for start in range(0, indices.numel(), CHUNK):
            membership = (indices[start:start+CHUNK, None] == vocabulary[None, :]).float()
            result.add_(membership.T @ gradients[start:start+CHUNK])
        return result, None


def embedding(weight, rows):
    require(isinstance(weight, torch.Tensor) and weight.layout == torch.strided
            and weight.dtype == torch.float32 and tuple(weight.shape) == (259, 15)
            and weight.is_contiguous() and weight.device.type in ('cpu', 'mps')
            and bool(torch.isfinite(weight).all()), 'Finite dense byte lookup weight required')
    require(type(rows) is torch.Tensor and rows.layout == torch.strided
            and rows.dtype == torch.int64 and rows.device == weight.device
            and rows.ndim in (1, 2) and 1 <= rows.numel() <= 1024*130
            and bool(((rows >= 0) & (rows < 259)).all()), 'Bounded same-device byte row IDs required')
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'),
            'Autocast forbidden')
    return F.embedding(rows, weight) if weight.device.type == 'cpu' else _Grouped.apply(weight, rows)
