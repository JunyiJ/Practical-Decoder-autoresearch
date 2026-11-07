"""Query-local elementwise sigmoid after weighted values,before output projection."""
import torch
from torch import nn
from torch.nn import functional as F

KINDS=('none','static_half','sigmoid_elementwise_zero')

def require(ok,message):
    if not ok:raise ValueError(message)

class ElementwiseOutputGate(nn.Module):
    def __init__(self,kind,width,heads,head_dim,*,device='cpu',dtype=torch.float32):
        super().__init__()
        require(type(kind) is str and kind in KINDS,'Explicit output-gate kind required')
        require(all(type(v) is int and v>0 for v in (width,heads,head_dim)) and width==heads*head_dim and width<=256 and heads<=8 and head_dim<=32,'Explicit bounded head/channel geometry required')
        require(device in ('cpu','mps') and (dtype==torch.float32 or (device=='cpu' and dtype==torch.float64)),'CPUFP32/FP64 oracle or nativeMPSFP32 required')
        self.kind=kind;self.width=width;self.heads=heads;self.head_dim=head_dim;self._signature=(kind,width,heads,head_dim)
        # No random initialization: learned and static-half outputs match exactly at step0.
        if kind=='sigmoid_elementwise_zero':self.weight=nn.Parameter(torch.zeros(width,width,device=device,dtype=dtype))
        else:self.register_parameter('weight',None)

    def forward(self,x,weighted_values):
        require((self.kind,self.width,self.heads,self.head_dim)==self._signature,'Mutable gate geometry/kind forbidden')
        y=weighted_values
        require(type(x) is torch.Tensor and type(y) is torch.Tensor and x.ndim==3 and y.ndim==4,'X[B,T,C] andY[B,H,T,D] required')
        require(x.layout==y.layout==torch.strided and min(x.shape)>0 and min(y.shape)>0,'Nonempty strided gate inputs required')
        b,t,c=x.shape
        require(b<=8 and t<=128 and c==self.width and y.shape==(b,self.heads,t,self.head_dim),'Exact bounded gate geometry required')
        require(x.dtype==y.dtype and x.device==y.device and x.device.type in ('cpu','mps') and (x.dtype==torch.float32 or (x.device.type=='cpu' and x.dtype==torch.float64)),'Matching declared device/dtype required')
        require(not torch.is_autocast_enabled(x.device.type),'Autocast forbidden')
        require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),'Finite gate inputs required before forward/gradient mutation')
        if self.kind=='sigmoid_elementwise_zero':
            w=self.weight
            require(type(w) is nn.Parameter and w.shape==(c,c) and w.device==x.device and w.dtype==x.dtype and w.requires_grad and bool(torch.isfinite(w).all()),'Exact finite learned elementwise gate matrix required')
            scores=torch.sigmoid(F.linear(x,w)).view(b,t,self.heads,self.head_dim).transpose(1,2)
            return y*scores
        require(self.weight is None,'Ungated/static arm must add no parameter')
        return y if self.kind=='none' else y*.5
