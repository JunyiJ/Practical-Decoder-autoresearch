"""Explicit top-two/four SwiGLU with deterministic lower-index ties, no dropping."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class Expert(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class RoutedSwiGLU(nn.Module):
    def __init__(self, dim, hidden, router_scale=1.0):
        super().__init__()
        if type(dim) is not int or type(hidden) is not int or min(dim, hidden) < 1:
            raise ValueError('Positive integer component dimensions required')
        if not math.isfinite(router_scale) or router_scale <= 0:
            raise ValueError('Positive finite router scale required')
        self.dim, self.hidden, self.router_scale = dim, hidden, float(router_scale)
        self.router = nn.Linear(dim, 4, bias=False)
        self.experts = nn.ModuleList([Expert(dim, hidden) for _ in range(4)])
        self.register_buffer('assignment_counts', torch.zeros(4, dtype=torch.int64))
        self.register_buffer('routed_tokens', torch.zeros((), dtype=torch.int64))

    def forward(self, x, *, record_routes=False):
        if x.ndim != 3 or x.shape[-1] != self.dim or x.numel() == 0:
            raise ValueError('Nonempty B,T,D component input required')
        flat = x.reshape(-1, self.dim)
        logits = self.router(flat) / self.router_scale
        probs = F.softmax(logits, dim=-1)
        chosen = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :2]
        selected_probs = probs.gather(1, chosen)
        weights = selected_probs / selected_probs.sum(dim=1, keepdim=True)
        output = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            tokens, positions = torch.where(chosen == i)
            if tokens.numel():
                values = expert(flat[tokens]) * weights[tokens, positions, None]
                output = output.index_add(0, tokens, values)
        counts = torch.bincount(chosen.reshape(-1), minlength=4)
        fractions = counts.to(probs.dtype) / (2 * flat.shape[0])
        aux = 4 * (fractions * probs.mean(dim=0)).sum()
        if record_routes:
            with torch.no_grad():
                self.assignment_counts.add_(counts)
                self.routed_tokens.add_(flat.shape[0])
        return output.reshape_as(x), aux, counts
