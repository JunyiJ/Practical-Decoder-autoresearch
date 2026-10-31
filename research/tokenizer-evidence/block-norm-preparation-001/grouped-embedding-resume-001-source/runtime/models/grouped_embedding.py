"""Versioned all-MPS first-order embedding backward, copied from the tested prototype.

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


EMBEDDING_FIELD = 'embedding_backward'
EMBEDDING_SPEC = dict(version=1, policy='tokenizer_grouped_mps_embedding_backward_v1',
    forward='native_dense_embedding', mps_backward='unique_membership_matmul_index_copy',
    cpu_backward='native_reference', dtype='float32', vocab_size=50257, width=256,
    max_gradient_tokens=1024, mps_autograd_order=1, padding_idx=None,
    scale_grad_by_freq=False, sparse=False)


def contract(value):
    from collections.abc import Mapping
    if (not isinstance(value, Mapping) or set(value)!=set(EMBEDDING_SPEC)
            or any(type(value[k]) is not type(v) or value[k]!=v for k,v in EMBEDDING_SPEC.items())):
        raise ValueError('Exact grouped embedding backward specification required')
    return dict(EMBEDDING_SPEC)


class GroupedMpsEmbedding(torch.nn.Embedding):
    def __init__(self,num_embeddings,embedding_dim):
        if (num_embeddings,embedding_dim)!=(50257,256):
            raise ValueError('Grouped embedding shape differs')
        super().__init__(num_embeddings,embedding_dim)
        self.embedding_backward_spec=dict(EMBEDDING_SPEC)

    def validate(self):
        contract(self.embedding_backward_spec)
        if (self.padding_idx is not None or self.max_norm is not None or self.norm_type!=2.
                or self.scale_grad_by_freq is not False or self.sparse is not False
                or tuple(self.weight.shape)!=(50257,256) or self.weight.dtype!=torch.float32
                or not self.weight.requires_grad or self.weight.device.type not in ('cpu','mps')
                or list(self.buffers())):
            raise ValueError('Constructed grouped embedding semantics differ')

    def forward(self,indices):
        self.validate()
        if indices.dtype!=torch.int64 or indices.ndim!=2 or indices.numel()==0:
            raise ValueError('Grouped embedding requires nonempty rank2 int64 IDs')
        if indices.device!=self.weight.device or torch.is_autocast_enabled(self.weight.device.type):
            raise ValueError('Grouped embedding device or autocast differs')
        if torch.is_grad_enabled():
            if indices.numel()>EMBEDDING_SPEC['max_gradient_tokens']:
                raise ValueError('Grouped embedding gradient token budget exceeded')
            if self.weight.device.type=='mps':
                return grouped_embedding(self.weight,indices)
        return super().forward(indices)
