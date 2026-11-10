"""Standalone whole-token reuse prototype. No production model integration."""
import torch
from torch.autograd.function import once_differentiable


class ExpandUnique(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, inverse):
        assert values.dtype == torch.float32 and inverse.dtype == torch.int64
        assert values.ndim == 2 and inverse.ndim == 1 and inverse.numel() <= 1024
        ctx.save_for_backward(inverse)
        ctx.rows = values.shape[0]
        return values.index_select(0, inverse)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        inverse, = ctx.saved_tensors
        membership = (inverse[:, None] == torch.arange(ctx.rows, device=inverse.device)[None, :]).float()
        # Fixed FP32 matrix reduction, avoiding repeated-index atomic scatter.
        return membership.T @ upstream, None


def compose(composer, code_ids, code_lengths, inputs, reuse):
    ids = inputs.reshape(-1)
    if reuse:
        ids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
    values = composer(code_ids.index_select(0, ids), code_lengths.index_select(0, ids))
    return ExpandUnique.apply(values, inverse) if reuse else values
