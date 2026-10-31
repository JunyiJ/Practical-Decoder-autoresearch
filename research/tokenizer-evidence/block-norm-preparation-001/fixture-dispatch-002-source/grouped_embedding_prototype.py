"""Standalone all-MPS first-order prototype; not imported by the training runtime.

Native embedding forward; backward groups unique token IDs, reduces each group's
FP32 upstream vectors by matrix multiplication, then writes distinct output rows.
Only the tested dense, untied, no-padding/no-frequency-scaling case is supported.
"""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _GroupedEmbedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weight,indices):
        if weight.dtype!=torch.float32 or weight.ndim!=2 or tuple(weight.shape)!=(50257,256):
            raise ValueError('Prototype requires FP32 vocabulary50257 by width256')
        if indices.dtype!=torch.int64 or indices.ndim!=2 or not 1<=indices.numel()<=1024:
            raise ValueError('Prototype requires1..1024 int64 token IDs in two dimensions')
        if weight.device.type!='mps' or indices.device!=weight.device:
            raise ValueError('Prototype requires one MPS device, without fallback')
        if torch.is_autocast_enabled('mps'):
            raise ValueError('Prototype forbids autocast')
        if bool((indices<0).any()) or bool((indices>=weight.shape[0]).any()):
            raise ValueError('Prototype token ID outside vocabulary')
        ctx.save_for_backward(indices);ctx.weight_shape=tuple(weight.shape)
        return F.embedding(indices,weight)

    @staticmethod
    @once_differentiable
    def backward(ctx,upstream):
        (indices,)=ctx.saved_tensors
        if upstream.dtype!=torch.float32 or upstream.device!=indices.device:
            raise ValueError('Prototype backward requires FP32 MPS gradients')
        tokens=indices.reshape(-1)
        unique=torch.unique(tokens,sorted=True)
        membership=(unique[:,None]==tokens[None,:]).to(dtype=torch.float32)
        summed=membership @ upstream.reshape(-1,ctx.weight_shape[1])
        gradient=torch.zeros(ctx.weight_shape,dtype=torch.float32,device=upstream.device)
        gradient.index_copy_(0,unique,summed)
        if any(t.device!=upstream.device for t in (unique,membership,summed,gradient)):
            raise RuntimeError('Prototype reduction left the MPS device')
        return gradient,None


def grouped_embedding(weight,indices):
    return _GroupedEmbedding.apply(weight,indices)
