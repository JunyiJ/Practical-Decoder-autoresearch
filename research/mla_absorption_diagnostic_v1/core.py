"""Affine MLA core controls at head32/latent64, with explicit rotary keys.

Inputs already contain down-projected queries, compressed KV and rotated
queries/keys. Both cores include Uq and the output projection. Neither includes
down-projection, rotary evaluation, normalization, cache allocation/update or MLP.
"""
import math
import torch
from research.body_allocation_model_v1.contract import require

def scores(q, k, rq, rk, mask):
    # Concatenated oracle preserves the actual original arithmetic/order.
    qq = torch.cat((q, rq), -1)
    kk = torch.cat((k, rk.unsqueeze(1).expand(-1, 8, -1, -1)), -1)
    return torch.softmax((((qq @ kk.transpose(-2, -1)) / math.sqrt(64)) * .5)
                         .masked_fill(~mask, float('-inf')), -1)

def original(a, cq, c, rq, rk, mask):
    b, n, _ = cq.shape; total = c.shape[1]
    q = a.uq(cq).view(b, n, 8, 32).transpose(1, 2)
    k = a.uk(c).view(b, total, 8, 32).transpose(1, 2)
    v = a.uv(c).view(b, total, 8, 32).transpose(1, 2)
    att = scores(q, k, rq, rk, mask)
    return a.out((att @ v).transpose(1, 2).contiguous().view(b, n, 256))

class Absorbed:
    def __init__(self, a):
        require(not torch.is_grad_enabled(), 'Derived inference matrices only')
        self.a = a; self.stamp = self.weight_stamp()
        self.uk = a.uk.weight.view(8, 32, 64).detach()
        self.bk = a.uk.bias.view(8, 32).detach()
        # O_h V_h maps each head's latent weighted mean directly to output.
        out = a.out.weight.view(256, 8, 32)
        self.value_out = torch.einsum('ohd,hdc->hco', out, a.uv.weight.view(8, 32, 64)).reshape(512, 256)
        self.value_bias_out = torch.einsum('ohd,hd->ho', out, a.uv.bias.view(8, 32))

    def weight_stamp(self):
        return tuple((n, p._version, p.untyped_storage().data_ptr(), p.device) for n, p in self.a.named_parameters())

    def __call__(self, cq, c, rq, rk, mask):
        require(not torch.is_grad_enabled() and self.weight_stamp() == self.stamp,
                'Unchanged fixed attention weights and no-grad required')
        b, n, _ = cq.shape
        q = self.a.uq(cq).view(b, n, 8, 32).transpose(1, 2)
        qc = torch.einsum('bhqd,hdc->bhqc', q, self.uk)
        content = qc @ c.unsqueeze(1).transpose(-2, -1)
        # The affine key term is retained, although constant across keys.
        content = content + (q * self.bk[None, :, None, :]).sum(-1, keepdim=True)
        rotary = rq @ rk.unsqueeze(1).transpose(-2, -1)
        att = torch.softmax((((content + rotary) / math.sqrt(64)) * .5)
                            .masked_fill(~mask, float('-inf')), -1)
        weighted_c = att @ c.unsqueeze(1)
        y = weighted_c.transpose(1, 2).contiguous().view(b, n, 512) @ self.value_out
        # Preserve finite-precision row mass; do not silently assume exactly one.
        y = y + torch.einsum('bhq,ho->bqo', att.sum(-1), self.value_bias_out)
        return y + self.a.out.bias

def matmul_macs(batch, queries, keys):
    """Multiply-accumulate counts for cores above; excludes adds/softmax/copies."""
    b, q, k = batch, queries, keys
    common_uq = b*q*64*256
    original = common_uq + 2*b*k*64*256 + b*8*q*k*64 + b*8*q*k*32 + b*q*256*256
    absorbed = common_uq + b*8*q*32*64 + b*8*q*k*(64+32) + b*8*q*k*64 + b*q*512*256
    return dict(original=original, absorbed=absorbed, ratio=absorbed/original,
                static_value_out_precompute=8*256*32*64,
                scope='Attention core only, not transformer/full-model FLOPs or measured latency.')
