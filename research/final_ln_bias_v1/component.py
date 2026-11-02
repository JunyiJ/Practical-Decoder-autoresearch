"""Final LayerNorm with fixed centering/scale and explicit bias policy."""
import torch
from torch import nn
from torch.nn import functional as F

KINDS = ('learned', 'fixed_zero')


class FinalLayerNorm(nn.Module):
    def __init__(self, width, kind):
        super().__init__()
        if type(width) is not int or width <= 0 or kind not in KINDS:
            raise ValueError('Positive width and explicit bias policy required')
        self.width, self.kind, self.eps = width, kind, 1e-5
        self.weight = nn.Parameter(torch.ones(width))
        self.register_parameter('bias', nn.Parameter(torch.zeros(width)) if kind == 'learned' else None)

    def forward(self, x):
        if x.shape[-1] != self.width:
            raise ValueError('Final LayerNorm width differs')
        return F.layer_norm(x, (self.width,), self.weight, self.bias, self.eps)
