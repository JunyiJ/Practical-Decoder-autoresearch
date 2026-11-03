"""Centered final LayerNorm with learned bias and explicit scale policy."""
import torch
from torch import nn
from torch.nn import functional as F
KINDS=('learned','fixed_unit')
class FinalLayerNorm(nn.Module):
 def __init__(self,width,kind):
  super().__init__()
  if type(width) is not int or width<=0 or kind not in KINDS:raise ValueError('Positive width and explicit scale policy required')
  self.width,self.kind,self.eps=width,kind,1e-5
  self.register_parameter('weight',nn.Parameter(torch.ones(width)) if kind=='learned' else None)
  self.bias=nn.Parameter(torch.zeros(width))
 def forward(self,x):
  if x.shape[-1]!=self.width:raise ValueError('Final LayerNorm width differs')
  return F.layer_norm(x,(self.width,),self.weight,self.bias,self.eps)
