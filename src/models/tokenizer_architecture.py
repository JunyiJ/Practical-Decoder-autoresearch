"""Explicit GPT-2 MHA/GQA backbone, separate from historical MHA-only studies."""
from collections.abc import Mapping

POLICY = 'tokenizer_mha_gqa_v1'
MLA_POLICY = 'tokenizer_mla_quarter_rope_v1'
SCORE_POLICY = 'tokenizer_mha_gqa_score_v1'
BIAS_POLICY = 'tokenizer_attention_no_output_bias'
FIXED = {'tokenizer_architecture_policy': POLICY, 'n_layers': 4, 'mlp_type': 'dense',
             'ffn': 'swiglu', 'ffn_bias': False, 'ffn_width_policy': 'swiglu_21_8',
             'parameterization': 'mup', 'base_width': 64, 'norm_type': 'rmsnorm', 'rope': 1,
             'dropout': 0., 'attn_dropout': 0., 'block_size': 128, 'vocab_size': 50257,
             'attention_logit_scale': .5, 'attention_logit_scale_width_policy': SCORE_POLICY,
             'output_bias_kind': 'none', 'output_bias_width_policy': BIAS_POLICY}
VARIABLE = {'dim', 'n_heads', 'attn_type', 'attention_width_policy',
            'depth_width_policy', 'output_bias_prior'}


def _get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def architecture_contract(cfg):
    from .tokenizer_hybrid import hybrid_contract
    hybrid = hybrid_contract(cfg)
    if hybrid is not None:
        return hybrid
    from .tokenizer_tables import table_contract
    tables = table_contract(cfg)
    if tables is not None:
        return tables
    if _get(cfg, 'tokenizer_architecture_policy') is None:
        return None
    policy = _get(cfg, 'tokenizer_architecture_policy')
    if policy not in (POLICY, MLA_POLICY):
        raise ValueError('Unknown tokenizer architecture policy')
    keys = set(cfg.keys()) if hasattr(cfg, 'keys') else set(vars(cfg))
    if keys != set(FIXED) | VARIABLE:
        raise ValueError('Tokenizer architecture requires its exact explicit field set')
    for key, value in FIXED.items():
        if key == 'tokenizer_architecture_policy':
            value = policy
        actual = _get(cfg, key)
        if type(actual) is not type(value) or actual != value:
            raise ValueError(f'Tokenizer architecture changed fixed field {key}')
    width, heads, family = (_get(cfg, k) for k in ('dim', 'n_heads', 'attn_type'))
    if type(width) is not int or width not in (64, 256) or type(heads) is not int or heads != width // 32:
        raise ValueError('Tokenizer architecture requires widths64/256 and head32')
    if policy == MLA_POLICY:
        if family != 'mla':
            raise ValueError('Tokenizer MLA policy requires MLA')
        attention, depth = 'mla_quarter_head32', 'depth4_unscaled'
    elif family not in ('mha', 'gqa'):
        raise ValueError('Tokenizer architecture supports only explicit MHA/GQA')
    else:
        attention, depth = ('mha_head32', 'depth4_unscaled') if family == 'mha' else ('gqa_half_head32', 'depth4_gqa_unscaled')
    if _get(cfg, 'attention_width_policy') != attention or _get(cfg, 'depth_width_policy') != depth:
        raise ValueError('Tokenizer architecture geometry policy mismatch')
    # Output-bias validation separately checks the complete tokenizer/data binding.
    prior = _get(cfg, 'output_bias_prior')
    return {'policy': policy, 'family': family, 'width': width, 'heads': heads,
                'score_scale': .5, 'prior_sha256': _get(prior, 'prior_sha256')}


def validate_architecture_model(model):
    contract = architecture_contract(model.cfg)
    if getattr(model, 'tokenizer_architecture_signature', None) != contract:
        raise ValueError('Constructed tokenizer architecture differs from mutable config')
    if contract is None:
        return
    if contract['family'] == 'hybrid':
        from .tokenizer_hybrid import validate_hybrid_model
        validate_hybrid_model(model, contract)
        return
    from torch import nn

    from ..attention.gqa import GQA
    from ..attention.mha import MHA
    from ..attention.mla import MLA
    expected = {'mha': MHA, 'gqa': GQA, 'mla': MLA}[contract['family']]
    if len(model.transformer.h) != 4 or hasattr(model.transformer, 'wpe'):
        raise ValueError('Tokenizer architecture layer/position table mismatch')
    if (type(model.transformer.ln_f) is not nn.LayerNorm or model.transformer.ln_f.eps != 1e-5
            or model.transformer.ln_f.bias is None
            or (model.lm_head is None) != (contract.get('table_kind') == 'scaled_tied')
            or model.output_bias is not None or model.input_embedding_scale != 1.):
        raise ValueError('Tokenizer architecture readout/input semantics mismatch')
    for block in model.transformer.h:
        if contract['family'] == 'mla' and (
                block.attn.dimc_q != contract['width']//4
                or block.attn.dimc_kv != contract['width']//4
                or block.attn.head_dim != 32 or block.attn.head_dim_r != 32
                or block.attn.n_heads != contract['heads']):
            raise ValueError('Tokenizer MLA latent/head geometry mismatch')
        if (type(block.attn) is not expected or type(block.ln1) is not nn.RMSNorm
                or type(block.ln2) is not nn.RMSNorm or not block.enable_rope
                or block.residual_topology_kind != 'serial'
                or block.mlp.gate_activation != 'silu' or block.mlp.hidden_norm is not None
                or block.attn.attention_logit_scale != .5):
            raise ValueError('Tokenizer architecture block semantics mismatch')


def validate_architecture_load(model, checkpoint):
    saved = _get(checkpoint.get('cfg', {}), 'model', {})
    if architecture_contract(model.cfg) != architecture_contract(saved):
        raise ValueError('Checkpoint tokenizer architecture contract differs')
    validate_architecture_model(model)
