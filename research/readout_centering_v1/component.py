"""Matched-epsilon affine normalization with only centering varied."""
import torch
from torch import nn
from torch.nn import functional as F

KINDS = ('layer_affine', 'rms_affine_bias')


class ReadoutNormalization(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        if type(width) is not int or width <= 0 or kind not in KINDS:
            raise ValueError('Positive width and explicit readout kind required')
        self.width = width
        self.kind = kind
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(width))
        self.bias = nn.Parameter(torch.zeros(width))

    def forward(self, x):
        if x.shape[-1] != self.width:
            raise ValueError('Readout width differs')
        if self.kind == 'layer_affine':
            return F.layer_norm(x, (self.width,), self.weight, self.bias, self.eps)
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight + self.bias
