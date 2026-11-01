import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, hidden_norm='none', gate_activation='silu'):
        super().__init__()
        if type(gate_activation) is not str or gate_activation not in ('silu', 'gelu_exact', 'relu'):
            raise ValueError('Unknown gated FFN activation')
        self.gate_activation = gate_activation
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.linear_gate = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.linear_up = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.linear_down = nn.Linear(self.hidden_dim, self.dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        from .ffn_hidden_norm import make_ffn_hidden_norm
        self.hidden_norm = make_ffn_hidden_norm(hidden_dim, hidden_norm)

    def forward(self, x):
        gate = self.linear_gate(x)
        if self.gate_activation == 'silu':
            gate = F.silu(gate)
        elif self.gate_activation == 'gelu_exact':
            gate = F.gelu(gate, approximate='none')
        else:
            gate = F.relu(gate)
        x = gate * self.linear_up(x)
        if self.hidden_norm is not None:
            x = self.hidden_norm(x)
        x = self.dropout(self.linear_down(x))
        return x


class UngatedSiLU(nn.Module):
    """Bias-free D256 -> H1008 -> D256, ordinary Linear initialization."""
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        if type(dim) is not int or dim != 256 or type(hidden_dim) is not int or hidden_dim != 1008 or type(dropout) is not float or dropout != 0.:
            raise ValueError('UngatedSiLU requires D256/H1008/dropout0')
        self.dim, self.hidden_dim = dim, hidden_dim
        self.linear_up = nn.Linear(dim, hidden_dim, bias=False)
        self.activation = nn.SiLU()
        self.linear_down = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.linear_down(self.activation(self.linear_up(x))))
