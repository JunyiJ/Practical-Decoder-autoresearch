"""Two fixed serial branch gains; no model-policy override or learned parameters."""
from torch import nn
KINDS=('unit','fixed_half')
GAINS={'unit':1.,'fixed_half':.5}
class ResidualScaleComposition(nn.Module):
    def __init__(self,attention,ffn,norm1,norm2,kind):
        super().__init__()
        if type(kind) is not str or kind not in KINDS:
            raise ValueError('Explicit unit or fixed_half required')
        if norm1 is norm2 or {id(p) for p in norm1.parameters()} & {id(p) for p in norm2.parameters()}:
            raise ValueError('Distinct normalization modules and parameters required')
        self.attention,self.ffn,self.norm1,self.norm2=attention,ffn,norm1,norm2
        self.kind=kind
    def forward(self,x):
        if self.kind=='unit':
            u=x+self.attention(self.norm1(x))
            return u+self.ffn(self.norm2(u))
        u=x+.5*self.attention(self.norm1(x))
        return u+.5*self.ffn(self.norm2(u))
