import math
import torch
import torch.nn as nn

from .blocks import TransformerBlock
from .rope import resolve_position_encoding

class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        from .gated_activation import gated_activation_signature
        self.gated_activation_signature = gated_activation_signature(cfg)
        from .attention_scale import attention_scale_signature
        self.attention_scale_signature = attention_scale_signature(cfg)
        from .residual_topology import residual_topology_signature
        self.residual_topology_signature = residual_topology_signature(cfg)
        from .weight_tying import weight_tying_signature, readout_scale
        self.weight_tying_signature = weight_tying_signature(cfg)
        self.table_readout_scale = readout_scale(cfg)
        from .output_bias import output_bias_signature, build_output_bias
        self.output_bias_signature = output_bias_signature(cfg)
        from .embedding_scale import resolve_input_scale
        self.input_embedding_scale = resolve_input_scale(cfg)
        from .readout_norm import readout_norm_signature, build_readout_norm
        self.readout_norm_signature = readout_norm_signature(cfg)
        from ..attention.kda_scaling import has_kda_policy, validate_kda_contract
        if has_kda_policy(cfg):
            validate_kda_contract(cfg)
        from .ffn_hidden_norm import validate_ffn_hidden_norm_contract
        validate_ffn_hidden_norm_contract(cfg)
        from ..attention.output_gate import validate_output_gate_contract
        validate_output_gate_contract(cfg)
        from ..attention.qk_norm import resolve_qk_norm, validate_qkrms_contract
        qk_kind = resolve_qk_norm(cfg)
        if (qk_kind != 'none'
                or getattr(cfg, 'depth_width_policy', None) == 'depth4_qkrms_unscaled'
                or getattr(cfg, 'attention_width_policy', None) == 'mha_head32_qkrms'):
            validate_qkrms_contract(cfg)
        if getattr(cfg, 'depth_width_policy', None) == 'depth6_dense_unscaled':
            from .depth_scaling import validate_depth6_dense_contract
            validate_depth6_dense_contract(cfg)
        moe_cfg = getattr(cfg, "moe", None)
        self.aux_loss_weight = getattr(moe_cfg, "aux_loss_weight", getattr(cfg, "aux_loss_weight", 1.0))
        self.position_encoding = resolve_position_encoding(cfg)
        self.enable_rope = self.position_encoding == "rope"
        embeddings = {"wte": nn.Embedding(cfg.vocab_size, cfg.dim)}
        if self.position_encoding == "learned":
            embeddings["wpe"] = nn.Embedding(cfg.block_size, cfg.dim)
        self.transformer = nn.ModuleDict({
            **embeddings,
            "h": nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)]),
            "ln_f": build_readout_norm(cfg)
        })
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        parameterization = getattr(cfg, 'parameterization', 'sp')
        if parameterization not in {'sp', 'mup'}:
            raise ValueError('parameterization must be sp or mup')
        base_width = getattr(cfg, 'base_width', 64)
        if isinstance(base_width, bool) or not isinstance(base_width, int) or base_width < 1:
            raise ValueError('base_width must be a positive integer')
        self.width_mult = cfg.dim / base_width if parameterization == 'mup' else 1.0
        # Default hidden fan-in initialization already has the required scaling.
        # Readout initialization has width-independent std in these coordinates;
        # the forward multiplier makes its effective std scale as 1/width.
        if parameterization == 'mup':
            with torch.no_grad():
                self.lm_head.weight.mul_(math.sqrt(self.width_mult))
                for module in self.transformer.h.modules():
                    if isinstance(module, nn.Linear) and module.bias is not None:
                        module.bias.mul_(math.sqrt(self.width_mult))

        # Allocate only after all shared weights; tensor construction consumes no RNG.
        self.output_bias = build_output_bias(cfg)
        # Consume all legacy initialization draws before cloning/removing the head.
        # The shared table has one canonical parameter/state/shadow entry.
        tying = None if self.weight_tying_signature is None else self.weight_tying_signature[0]
        if tying == 'matched_untied':
            with torch.no_grad():
                self.lm_head.weight.copy_(self.transformer.wte.weight)
        elif tying == 'scaled_tied':
            self.lm_head = None

    def forward(
        self,
        idx,
        targets=None,
        cache=None,
        start_pos=0,
        include_aux_loss=True,
        return_loss_breakdown=False,
    ):
        b, t = idx.size()
        if start_pos + t > self.cfg.block_size:
            raise ValueError(
                f"Sequence length {t} with start_pos {start_pos} exceeds block_size {self.cfg.block_size}"
            )
        token_emb = self.transformer.wte(idx)
        # Keep legacy and explicit-unit forward paths exactly unchanged.
        if self.input_embedding_scale != 1.:
            token_emb = token_emb * self.input_embedding_scale
        if self.position_encoding == "learned":
            pos = torch.arange(start_pos, start_pos + t, device=idx.device)
            pos_emb = self.transformer.wpe(pos)
            x = token_emb + pos_emb
        else:
            x = token_emb
        aux_losses = []
        if cache is not None and not isinstance(cache, (list, tuple)):
            raise TypeError("cache must be a list/tuple of per-layer KVCache objects")
        if cache is not None and len(cache) != len(self.transformer.h):
            raise ValueError("cache length must match number of transformer blocks")
        for layer_idx, block in enumerate(self.transformer.h):
            layer_cache = None if cache is None else cache[layer_idx]
            x, block_aux_loss = block(x, layer_cache, start_pos)
            if block_aux_loss is not None:
                aux_losses.append(block_aux_loss)
        x = self.transformer.ln_f(x)
        tying = None if self.weight_tying_signature is None else self.weight_tying_signature[0]
        if tying == 'scaled_tied':
            logits = nn.functional.linear(x, self.transformer.wte.weight) * self.table_readout_scale / self.width_mult
        elif tying == 'matched_untied':
            logits = self.lm_head(x) * self.table_readout_scale / self.width_mult
        else:
            logits = self.lm_head(x) / self.width_mult
        if self.output_bias is not None:
            logits = logits + self.output_bias

        loss = None
        ce_loss = None
        aux_loss = None
        if targets is not None:
            ce_loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss = ce_loss
            if aux_losses:
                aux_loss = sum(aux_losses) / len(aux_losses)
                if include_aux_loss:
                    loss = loss + self.aux_loss_weight * aux_loss

        if return_loss_breakdown:
            return logits, loss, {
                "ce_loss": ce_loss,
                "aux_loss": aux_loss,
                "total_loss": loss,
            }
        return logits, loss
