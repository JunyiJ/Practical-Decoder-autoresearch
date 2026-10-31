import torch
import torch.nn as nn
from .mla import latent_dimensions
from .gqa import kv_head_count

class KVCache(nn.Module):
    def __init__(self, cfg, device, batch_size=None, dtype=None, n_kv_heads=None, layer_index=None):
        super().__init__()
        from ..models.tokenizer_hybrid import hybrid_contract, layer_family
        self.hybrid_signature = hybrid_contract(cfg.model)
        self.layer_index = layer_index
        from ..models.block_norm import contract as norm_contract
        self.block_norm_signature = norm_contract(cfg.model)
        if self.block_norm_signature is not None:
            self.position = 0
        self.attn_type = layer_family(cfg.model, layer_index)
        if self.hybrid_signature is not None:
            if n_kv_heads is not None:
                raise ValueError('Hybrid cache rejects KV head overrides')
            self.position = 0
        self.is_mla = self.attn_type in {"mla", "gated_mla"}
        if self.attn_type == "gqa":
            expected = kv_head_count(cfg.model)
            if n_kv_heads is not None and n_kv_heads != expected:
                raise ValueError("Cache KV heads must match GQA attention")
            n_kv_heads = expected
        if n_kv_heads is None:
            n_kv_heads = getattr(cfg.model, "n_kv_heads", cfg.model.n_heads)
            if getattr(cfg.model, "attn_type", None) == "mha":
                n_kv_heads = cfg.model.n_heads
        if batch_size is None:
            batch_size = getattr(cfg.training, "batch_size", 1)
        if self.attn_type == "kda":
            heads = cfg.model.n_heads
            head_dim = cfg.model.dim // heads
            self.max_seq_len = cfg.model.block_size
            self.position = 0
            state_dtype = torch.float64 if dtype == torch.float64 else torch.float32
            self.register_buffer("state", torch.zeros(
                batch_size, heads, head_dim, head_dim, device=device, dtype=state_dtype))
            for name in ("q", "k", "v"):
                self.register_buffer(name + "_history", torch.zeros(
                    batch_size, cfg.model.dim, 3, device=device, dtype=dtype))
            self.k_cache = None
            self.v_cache = None
            return
        if self.is_mla:
            _, dimc_kv = latent_dimensions(cfg.model)
            head_dim_r = getattr(cfg.model, "head_dim_r", cfg.model.dim // cfg.model.n_heads)
            c_cache_shape = (batch_size, cfg.model.block_size, dimc_kv)
            r_cache_shape = (batch_size, cfg.model.block_size, head_dim_r)
            self.register_buffer("c_cache", torch.zeros(c_cache_shape, device=device, dtype=dtype))
            self.register_buffer("r_cache", torch.zeros(r_cache_shape, device=device, dtype=dtype))
            self.k_cache = None
            self.v_cache = None
            return
        # Cache shape: (Batch, Num_Heads, Max_Seq_Len, Head_Dim)
        head_dim = cfg.model.dim // cfg.model.n_heads
        cache_shape = (
            batch_size,
            n_kv_heads,
            cfg.model.block_size,
            head_dim,
        )
        self.register_buffer("k_cache", torch.zeros(cache_shape, device=device, dtype=dtype))
        self.register_buffer("v_cache", torch.zeros(cache_shape, device=device, dtype=dtype))

    def update(self, k_new, v_new, start_pos):
        """
        k_new, v_new: (Batch, Num_Heads, Seq_Len_New, Head_Dim)
        """
        if self.block_norm_signature is not None:
            if type(start_pos) is not int or start_pos != self.position:
                raise ValueError('Block norm cache requires contiguous integer positions')
            if (k_new.ndim != 3 or v_new.ndim != 3 or k_new.shape[:2] != v_new.shape[:2]
                    or k_new.shape[0] != self.c_cache.shape[0]
                    or k_new.shape[2] != 64 or v_new.shape[2] != 32
                    or k_new.dtype != self.c_cache.dtype or v_new.dtype != self.r_cache.dtype
                    or k_new.device != self.c_cache.device or v_new.device != self.r_cache.device):
                raise ValueError('Block norm cache update shape/dtype/device differs')
        if self.attn_type == "kda":
            raise ValueError("KDA updates recurrent state and convolution history directly")
        if self.is_mla:
            c_new = k_new
            r_new = v_new
            if self.hybrid_signature is not None and start_pos != self.position:
                raise ValueError('Hybrid cache requires contiguous positions')
            end_pos = start_pos + c_new.size(1)
            if end_pos > self.c_cache.size(1):
                raise ValueError("Cache size exceeded; increase block_size or reset cache.")
            self.c_cache[:, start_pos:end_pos, :] = c_new
            self.r_cache[:, start_pos:end_pos, :] = r_new
            if self.hybrid_signature is not None or self.block_norm_signature is not None:
                self.position = end_pos
            return self.c_cache[:, :end_pos, :], self.r_cache[:, :end_pos, :]
        end_pos = start_pos + k_new.size(2)
        if end_pos > self.k_cache.size(2):
            raise ValueError("Cache size exceeded; increase block_size or reset cache.")
        self.k_cache[:, :, start_pos : end_pos, :] = k_new
        self.v_cache[:, :, start_pos: end_pos, :] = v_new
        return self.k_cache[:, :, :end_pos, :], self.v_cache[:, :, :end_pos, :]


    def reset(self):
        if self.attn_type == "kda":
            self.state.zero_()
            for name in ("q", "k", "v"):
                getattr(self, name + "_history").zero_()
            self.position = 0
            return
        if self.is_mla:
            self.c_cache.zero_()
            self.r_cache.zero_()
            if self.hybrid_signature is not None or self.block_norm_signature is not None:
                self.position = 0
            return
        self.k_cache.zero_()
        self.v_cache.zero_()


def build_caches(cfg, device, batch_size=None, dtype=None):
    """Build cache state from the same validated layer layout as the model."""
    return [KVCache(cfg, device, batch_size=batch_size, dtype=dtype, layer_index=i)
            for i in range(cfg.model.n_layers)]
