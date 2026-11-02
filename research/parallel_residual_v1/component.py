"""Two separate-normalizer residual equations; no model-policy override."""
from torch import nn
KINDS=('serial','parallel_separate_norms')

class ResidualComposition(nn.Module):
    def __init__(self,attention,ffn,norm1,norm2,kind):
        super().__init__()
        if type(kind) is not str or kind not in KINDS:
            raise ValueError('Explicit serial or parallel_separate_norms required')
        if norm1 is norm2 or {id(p) for p in norm1.parameters()} & {id(p) for p in norm2.parameters()}:
            raise ValueError('Distinct normalization modules and parameters required')
        self.attention,self.ffn,self.norm1,self.norm2=attention,ffn,norm1,norm2
        self.kind=kind

    def forward(self,x):
        u=x+self.attention(self.norm1(x))
        branch_input=x if self.kind=='parallel_separate_norms' else u
        return u+self.ffn(self.norm2(branch_input))
