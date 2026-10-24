"""Explicit optimizer/width contracts for fixed-depth dense-MHA transfer studies.

Muon uses torch's RMS-matched update at the base width. Under muP its
hidden-matrix LR scales by width**(-1/2), cancelling RMS matching's sqrt(width).
The resulting spectral update scale is constant for fixed matrix aspect ratios.
Embeddings, readout and vectors use AdamW. Adam hidden matrices scale by 1/width;
readout uses an explicit 1/width forward multiplier and constant optimizer LR.
"""
import math
import torch


def optimizer_options(training):
    result = {
        'optimizer': str(getattr(training, 'optimizer', 'adamw')),
        'decay_mode': str(getattr(training, 'decay_mode', 'conventional')),
    }
    if result['optimizer'] not in {'adamw', 'muon'}:
        raise ValueError('optimizer must be adamw or muon')
    if result['decay_mode'] not in {'conventional', 'independent'}:
        raise ValueError('decay_mode must be conventional or independent')
    return result


class HybridOptimizer:
    """Checkpointable composition; independent decay applied exactly once.

    Only parameters with a gradient decay, matching PyTorch AdamW semantics.
    The independent coefficient is the per-step shrinkage, not AdamW's lambda.
    """
    def __init__(self, optimizers, decay):
        self.optimizers = optimizers
        self.decay = decay
        self.param_groups = [g for opt in optimizers for g in opt.param_groups]

    def zero_grad(self, set_to_none=True):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        if self.decay:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is not None:
                        p.mul_(1 - self.decay)
        for opt in self.optimizers:
            opt.step()

    def state_dict(self):
        return {'version': 1, 'independent_decay': self.decay,
                'optimizers': [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state):
        if (state.get('version') != 1 or state['independent_decay'] != self.decay
                or len(state['optimizers']) != len(self.optimizers)):
            raise ValueError('Optimizer composition or independent decay changed')
        for opt, saved in zip(self.optimizers, state['optimizers']):
            opt.load_state_dict(saved)


def build_optimizer(model, training):
    opts = optimizer_options(training)
    lr, wd = float(training.learning_rate), float(training.weight_decay)
    if not math.isfinite(lr) or lr < 0 or not math.isfinite(wd) or wd < 0:
        raise ValueError('Learning rate and decay must be finite and nonnegative')
    independent = opts['decay_mode'] == 'independent'
    if independent and wd >= 1:
        raise ValueError('Independent per-step decay must be < 1')
    mup = getattr(model.cfg, 'parameterization', 'sp') == 'mup'
    muon = opts['optimizer'] == 'muon'
    if mup or muon:
        cfg = model.cfg
        policy = getattr(cfg, 'ffn_width_policy', 'dense_4x')
        dense = (policy == 'dense_4x' and getattr(cfg, 'ffn', 'gelu') in {'gelu', 'silu'}
                 and getattr(cfg, 'hidden_dim', 4 * cfg.dim) == 4 * cfg.dim)
        gated = (policy == 'swiglu_21_8' and getattr(cfg, 'ffn', 'gelu') == 'swiglu'
                 and cfg.dim in {64,128,256} and getattr(cfg, 'base_width',64) == 64
                 and ((cfg.n_layers == 2 and not hasattr(cfg, 'depth_width_policy'))
                      or (cfg.n_layers == 4 and getattr(cfg, 'depth_width_policy', None) == 'depth4_unscaled'
                          and cfg.attn_type == 'mha' and cfg.n_heads == cfg.dim // 32
                          and getattr(cfg, 'attention_width_policy', 'mha_head32') == 'mha_head32'
                          and not hasattr(cfg, 'n_kv_heads') and not hasattr(cfg, 'n_groups')))
                 and ((getattr(cfg, 'attention_width_policy', 'mha_head32') == 'mha_head32' and cfg.n_heads == cfg.dim // 32)
                      or (getattr(cfg, 'attention_width_policy', None) == 'mha_head16' and cfg.n_heads == cfg.dim // 16)
                      or (getattr(cfg, 'attention_width_policy', None) == 'gqa_half_head32' and cfg.n_heads == cfg.dim // 32))
                 and not hasattr(cfg, 'hidden_dim') and getattr(cfg, 'ffn_bias',False) is False)
        attention_policy = getattr(cfg, 'attention_width_policy', 'mha_head32')
        attention_ok = (cfg.attn_type == 'mha' and attention_policy in {'mha_head32', 'mha_head16'}) or (
            gated and cfg.attn_type == 'gqa' and attention_policy == 'gqa_half_head32'
            and not hasattr(cfg, 'n_kv_heads') and not hasattr(cfg, 'n_groups'))
        if not attention_ok or cfg.mlp_type != 'dense' or not (dense or gated):
            raise ValueError('Optimizer width study supports dense GELU/SiLU 4x MHA or explicit SwiGLU MHA/GQA width contract only')
    # Preserve the historical optimizer and checkpoint format in default mode.
    if not mup and not muon and not independent:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    mult = model.width_mult if mup else 1.0
    adam, matrices = [], []
    for name, param in model.named_parameters():
        hidden = name.startswith('transformer.h.') and param.ndim == 2
        scale = (1 / math.sqrt(mult) if muon else 1 / mult) if hidden else 1.0
        group = {'params': [param], 'lr': lr * scale, 'weight_decay': 0.0 if independent else wd,
                 'parameter_names': [name], 'width_lr_scale': scale}
        (matrices if muon and hidden else adam).append(group)
    optimizers = []
    if matrices:
        optimizers.append(torch.optim.Muon(matrices, lr=lr, weight_decay=0 if independent else wd,
                                          adjust_lr_fn='match_rms_adamw'))
    if adam:
        optimizers.append(torch.optim.AdamW(adam, lr=lr, weight_decay=0 if independent else wd))
    return HybridOptimizer(optimizers, wd if independent else 0.0)
