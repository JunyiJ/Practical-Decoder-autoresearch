"""Identity-initialized, bias-free per-channel causal three-tap mixing."""
import torch
from torch import nn

class CausalThreeTap(nn.Module):
    def __init__(self,width):
        super().__init__()
        if type(width) is not int or width<=0:raise ValueError('Positive channel width required')
        self.width=width
        self.current=nn.Parameter(torch.ones(width))
        self.lag1=nn.Parameter(torch.zeros(width))
        self.lag2=nn.Parameter(torch.zeros(width))

    def forward_chunk(self,x,history=None):
        if x.ndim!=3 or x.shape[1]<1 or x.shape[2]!=self.width:raise ValueError('Nonempty B,T,D input required')
        if x.device!=self.current.device or x.dtype!=self.current.dtype:raise ValueError('Input/model dtype and device must agree')
        if history is None:history=x.new_zeros((x.shape[0],2,self.width))
        if history.shape!=(x.shape[0],2,self.width) or history.dtype!=x.dtype or history.device!=x.device:raise ValueError('Exact two-token per-channel history required')
        joined=torch.cat((history,x),dim=1);n=x.shape[1]
        y=joined[:,2:2+n]*self.current+joined[:,1:1+n]*self.lag1+joined[:,:n]*self.lag2
        return y,joined[:,-2:].clone()

    def forward(self,x):
        return self.forward_chunk(x)[0]
