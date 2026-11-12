"""Context128/256 contract, isolated from historical fixed128 contracts."""
from collections.abc import Mapping
import torch
from torch import nn
from research.block_norm_v1.spec import Spec, decode, encode, make, checked_forward, validate_eight, require

POLICY = 'tokenizer_mla_block_norm_v1'
GROUPED_POLICY = 'tokenizer_mla_block_norm_grouped_embedding_v2'
FFN_POLICY = 'tokenizer_mla_ffn_structure_v1'
MLA_SCORE_POLICY = 'tokenizer_mla_score_transfer_v1'
FFN_NORM_POLICY = 'tokenizer_mla_swiglu_product_rms_v1'
LOCAL_MIX_POLICY = 'tokenizer_mla_ffn_input_causal_mix_v1'
BRANCH_POLICY = 'tokenizer_mla_fixed_complete_branch_gain_v1'
READOUT_POLICY = 'tokenizer_mla_final_ln_scale_v1'
POLICIES = (READOUT_POLICY, BRANCH_POLICY, POLICY, GROUPED_POLICY, FFN_POLICY, MLA_SCORE_POLICY, FFN_NORM_POLICY, LOCAL_MIX_POLICY)
GROUPED_POLICIES = (READOUT_POLICY, BRANCH_POLICY, GROUPED_POLICY, FFN_POLICY, MLA_SCORE_POLICY, FFN_NORM_POLICY, LOCAL_MIX_POLICY)
FIELD = 'block_normalization'
PLACEMENTS = tuple(f'transformer.h.{i}.ln{j}' for i in range(4) for j in (1, 2))


def get(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)


def contract(cfg):
    policy = get(cfg, 'tokenizer_architecture_policy')
    require(policy == READOUT_POLICY, 'Native learned readout policy only')
    length = get(cfg, 'block_size')
    require(type(length) is int and length in (128, 256), 'Context128 or256 required')
    from .grouped_embedding import EMBEDDING_FIELD, contract as embedding_contract
    if policy not in POLICIES:
        require(get(cfg, FIELD) is None, 'Block normalization requires its named architecture policy')
        require(get(cfg, EMBEDDING_FIELD) is None, 'Grouped embedding requires its named architecture policy')
        return None
    from .tokenizer_architecture import FIXED, VARIABLE
    keys = set(cfg.keys()) if hasattr(cfg, 'keys') else set(vars(cfg))
    extra = {EMBEDDING_FIELD} if policy in GROUPED_POLICIES else set()
    if policy in (BRANCH_POLICY, READOUT_POLICY):
        extra |= {'residual_gain_kind', 'residual_gain_width_policy'}
    if policy == READOUT_POLICY:
        extra |= {'readout_norm_kind', 'readout_norm_width_policy', 'readout_norm_epsilon', 'final_ln_bias_kind', 'final_ln_bias_policy', 'final_ln_scale_kind', 'final_ln_scale_policy'}
    if policy == LOCAL_MIX_POLICY:
        extra |= {'ffn_input_mixing'}
    if policy == FFN_NORM_POLICY:
        extra |= {'ffn_hidden_norm', 'ffn_hidden_norm_policy'}
    require(keys == set(FIXED) | VARIABLE | {FIELD} | extra, 'Block normalization requires exact explicit fields')
    spec = decode(get(cfg, FIELD))
    embedding = embedding_contract(get(cfg, EMBEDDING_FIELD)) if extra else None
    fixed = dict(FIXED, block_size=length, tokenizer_architecture_policy=policy,
                 norm_type='layernorm' if spec.bias else 'rmsnorm', dim=256, n_heads=8,
                 attn_type='mla', attention_width_policy='mla_quarter_head32',
                 depth_width_policy='depth4_unscaled')
    structure = None
    if policy == FFN_POLICY:
        require(spec.kind == 'rms_incumbent', 'FFN comparison requires incumbent RMSNorm')
        choice = (get(cfg, 'ffn'), get(cfg, 'ffn_width_policy'))
        require(choice in (('swiglu', 'swiglu_21_8'), ('silu', 'ungated_silu_63_16')),
                'FFN comparison requires one exact matched-budget arm')
        fixed.update(ffn=choice[0], ffn_width_policy=choice[1])
        structure = 'swiglu' if choice[0] == 'swiglu' else 'ungated_silu'
    score = .5
    if policy == MLA_SCORE_POLICY:
        require(spec.kind == 'rms_incumbent', 'Score comparison requires incumbent RMSNorm')
        score = get(cfg, 'attention_logit_scale')
        require(type(score) is float and score in (.5, 1.),
                'Score comparison requires exactly float0.5 or float1.0')
        fixed['attention_logit_scale'] = score
        structure = 'swiglu'
    hidden_norm = None
    if policy == FFN_NORM_POLICY:
        require(spec.kind == 'rms_incumbent', 'Product norm requires incumbent block RMSNorm')
        kind = get(cfg, 'ffn_hidden_norm')
        require(type(kind) is str and kind in ('none', 'rms'), 'Exact none/RMS product norm arm required')
        require(get(cfg, 'ffn_hidden_norm_policy') == 'mla_swiglu_product_rms_672_v1', 'Product norm policy differs')
        hidden_norm = dict(kind=kind, epsilon=1e-5, affine=kind == 'rms', bias=False,
                           width=672, placement='post_product_pre_down', gain_init=1.)
        structure = 'swiglu'
    mixing = None
    if policy == LOCAL_MIX_POLICY:
        require(spec.kind == 'rms_incumbent', 'Local mixing requires incumbent RMSNorm')
        kind = get(cfg, 'ffn_input_mixing')
        require(type(kind) is str and kind in ('none', 'causal3'), 'Exact none/causal3 arm required')
        mixing = dict(kind=kind, width=256, taps=3, placement='post_ffn_norm_pre_swiglu',
                      bias=False, initialization=[1., 0., 0.], history_tokens=2)
    gain = None
    if policy in (BRANCH_POLICY, READOUT_POLICY):
        require(spec.kind == 'rms_incumbent', 'Fixed branch scaling requires incumbent RMSNorm')
        kind = get(cfg, 'residual_gain_kind')
        require(type(kind) is str and kind in ('unit', 'fixed_half'), 'Exact unit/fixed_half arm required')
        require(get(cfg, 'residual_gain_width_policy') == 'mla4_fixed_complete_branch_gain_v1',
                'Fixed complete-branch gain policy differs')
        gain = dict(kind=kind, value=1. if kind == 'unit' else .5,
                    placement='complete_post_projection_both_branches', learned=False)
    final_norm = dict(kind='layer_affine', epsilon=1e-5)
    if policy == READOUT_POLICY:
        require(gain['kind'] == 'unit', 'Readout requires fixed unit branches')
        kind = get(cfg, 'readout_norm_kind')
        require(type(kind) is str and kind == 'layer_affine', 'Exact readout kind required')
        require(get(cfg, 'readout_norm_width_policy') == 'mla4_final_ln_scale_v1'
                and type(get(cfg, 'readout_norm_epsilon')) is float
                and get(cfg, 'readout_norm_epsilon') == 1e-5, 'Exact readout policy and epsilon required')
        bias_kind = get(cfg, 'final_ln_bias_kind')
        require(type(bias_kind) is str and bias_kind == 'learned'
                and get(cfg, 'final_ln_bias_policy') == 'mla4_final_ln_bias_v1', 'Exact final bias policy required')
        scale_kind=get(cfg,'final_ln_scale_kind')
        require(type(scale_kind) is str and scale_kind in ('learned','fixed_unit') and get(cfg,'final_ln_scale_policy')=='mla4_final_ln_scale_v1','Exact final scale policy required')
        final_norm = dict(kind=kind,epsilon=1e-5,affine=True,bias=True,bias_kind='learned',scale_kind=scale_kind,scale=scale_kind=='learned',width=256,placement='final_readout_only',weight_init=1.,bias_init=0.)

    for key, expected in fixed.items():
        value = get(cfg, key)
        require(type(value) is type(expected) and value == expected,
                f'Block normalization changed fixed field {key}')
    return dict(**(dict(branch_gain=gain) if gain is not None else {}),
                **(dict(ffn_input_mixing=mixing) if mixing is not None else {}),
                **(dict(ffn_product_normalization=hidden_norm) if hidden_norm is not None else {}),
                context_length=length, context_policy='general_rsi_context_length_v1',
                policy=policy, family='mla', width=256, heads=8, score_scale=score,
                **({EMBEDDING_FIELD: embedding} if embedding is not None else {}),
                prior_sha256=get(get(cfg, 'output_bias_prior'), 'prior_sha256'),
                normalization=encode(spec), placements=list(PLACEMENTS),
                final_normalization=final_norm,
                gate_kind='silu', hidden_width=1008 if structure == 'ungated_silu' else 672,
                **(dict(ffn_structure=structure) if structure is not None else {}),
                residual='serial_unit', tables='untied')


def resolve(cfg):
    value = contract(cfg)
    return None if value is None else decode(value['normalization'])


def validate_model(model, value):
    spec = decode(value['normalization'])
    require(model.block_norm_spec == spec, 'Constructed block norm specification differs')
    normalizers = [model.get_submodule(name) for name in PLACEMENTS]
    validate_eight(spec, normalizers)
    require(len({id(m.weight) for m in normalizers}) == 8, 'Separate normalizer parameters required')
    parameters = list(model.parameters())
    require(all(p.dtype == torch.float32 for p in parameters), 'Block norm model requires FP32 parameters')
    require(len({p.device for p in parameters}) == 1, 'Block norm model devices differ')
    require(sum(p.numel() for p in parameters) == 28492416 - (256 if value['final_normalization'].get('scale_kind') == 'fixed_unit' else 0) + (2048 if spec.bias else 0) + (2688 if value.get('ffn_product_normalization', {}).get('kind') == 'rms' else 0) + (3072 if value.get('ffn_input_mixing', {}).get('kind') == 'causal3' else 0),
            'Block norm model parameter inventory differs')
    from .grouped_embedding import GroupedMpsEmbedding, EMBEDDING_FIELD
    expected_embedding = GroupedMpsEmbedding if value['policy'] in GROUPED_POLICIES else nn.Embedding
    require(type(model.transformer.wte) is expected_embedding, 'Constructed embedding backward policy differs')
    if expected_embedding is GroupedMpsEmbedding:
        model.transformer.wte.validate()
        require(_same_typed(model.transformer.wte.embedding_backward_spec, value[EMBEDDING_FIELD]),
                'Constructed embedding backward specification differs')
    final = model.transformer.ln_f
    if value['policy'] == READOUT_POLICY:
        from .readout_norm import validate_final
        validate_final(model)
    else:
        require(type(final) is nn.LayerNorm and final.normalized_shape == (256,)
                and final.eps == 1e-5 and final.elementwise_affine is True
                and final.weight is not None and final.bias is not None
                and final.weight.requires_grad and final.bias.requires_grad,
                'Final affine LayerNorm differs')
    require(model.lm_head is not None and model.lm_head.bias is None
            and model.lm_head.weight.shape == (50257, 256)
            and model.transformer.wte.weight.shape == (50257, 256)
            and model.lm_head.weight.untyped_storage().data_ptr()
            != model.transformer.wte.weight.untyped_storage().data_ptr(), 'Untied tables differ')
    from .mlp import SwiGLU, UngatedSiLU
    for i, block in enumerate(model.transformer.h):
        require(block.block_norm_spec == spec and block.layer_index == i
                and block.norm_type == ('layernorm' if spec.bias else 'rmsnorm'), 'Block norm placement differs')
        require(block.residual_gain_kind == value.get('branch_gain', {}).get('kind', 'unit')
                and not any(hasattr(block, n) for n in ('attention_residual_gain', 'ffn_residual_gain')),
                'Constructed fixed branch gain differs')
        if 'ffn_input_mixing' in value:
            from research.local_mix_v1.component import CausalThreeTap
            mixer = block.ffn_input_mixer
            if value['ffn_input_mixing']['kind'] == 'none':
                require(mixer is None, 'Control cannot contain a mixer')
            else:
                require(type(mixer) is CausalThreeTap and mixer.width == 256
                        and set(mixer._parameters) == {'current', 'lag1', 'lag2'}
                        and not mixer._modules and not mixer._buffers
                        and all(tuple(p.shape) == (256,) and p.requires_grad for p in mixer.parameters()),
                        'Constructed mixer semantics differ')
        ungated = value.get('ffn_structure') == 'ungated_silu'
        width = 1008 if ungated else 672
        require(type(block.mlp) is (UngatedSiLU if ungated else SwiGLU)
                and block.hidden_dim == width and block.mlp.dim == 256
                and block.mlp.hidden_dim == width and block.mlp.dropout.p == 0.,
                'FFN dimensions/type/dropout differ')
        if ungated:
            require(type(block.mlp.activation) is nn.SiLU and block.mlp.activation.inplace is False
                    and set(block.mlp._modules) == {'linear_up','activation','linear_down','dropout'},
                    'Ungated SiLU module layout differs')
        product = value.get('ffn_product_normalization')
        if product is not None:
            norm = block.mlp.hidden_norm
            if product['kind'] == 'none':
                require(norm is None, 'Control product must remain unnormalized')
            else:
                require(type(norm) is nn.RMSNorm and norm.normalized_shape == (672,)
                        and norm.eps == 1e-5 and norm.elementwise_affine is True
                        and tuple(norm.weight.shape) == (672,) and norm.weight.requires_grad
                        and set(norm.state_dict()) == {'weight'}, 'Product RMS constructed semantics differ')
        shapes = {'linear_up': (width, 256), 'linear_down': (256, width)}
        if not ungated:
            shapes['linear_gate'] = (672, 256)
        for name, shape in shapes.items():
            layer = getattr(block.mlp, name)
            require(type(layer) is nn.Linear and tuple(layer.weight.shape) == shape
                    and layer.bias is None, 'SwiGLU projection differs')
        require(block.attn.causal is True and block.attn.attn_dropout.p == 0.
                and block.attn.out_dropout.p == 0., 'MLA causal/dropout semantics differ')


CHECKPOINT_KEY = 'block_normalization_signature'


def _same_typed(a, b):
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same_typed(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same_typed(x, y) for x, y in zip(a, b))
    return a == b


def checkpoint_signature(model):
    """Bind validated constructed semantics, not only mutable config labels."""
    value = contract(model.cfg)
    if value is None:
        require(getattr(model, 'block_norm_spec', None) is None,
                'New block normalization model cannot drop its policy')
        return None
    from .tokenizer_architecture import validate_architecture_model
    validate_architecture_model(model)
    return dict(version=1, architecture=value,
                parameter_names=[name + '.' + parameter for name in PLACEMENTS
                                 for parameter in ('weight', 'bias') if parameter == 'weight'
                                 or model.block_norm_spec.bias]
                + ((['transformer.ln_f.weight'] if value['final_normalization']['scale'] else []) + ['transformer.ln_f.bias'] if value['policy'] == READOUT_POLICY else [])
                + ([f'transformer.h.{i}.mlp.hidden_norm.weight' for i in range(4)]
                   if value.get('ffn_product_normalization', {}).get('kind') == 'rms' else [])
                + ([f'transformer.h.{i}.ffn_input_mixer.{tap}' for i in range(4)
                    for tap in ('current', 'lag1', 'lag2')]
                   if value.get('ffn_input_mixing', {}).get('kind') == 'causal3' else []))


def validate_checkpoint(model, checkpoint):
    expected = checkpoint_signature(model)
    if expected is None:
        require(CHECKPOINT_KEY not in checkpoint, 'Legacy model cannot accept block normalization signature')
        return
    saved = checkpoint.get(CHECKPOINT_KEY)
    require(_same_typed(saved, expected), 'Saved constructed block normalization signature differs or is missing')
    cfg = get(checkpoint.get('cfg', {}), 'model', {})
    require(_same_typed(contract(cfg), expected['architecture']), 'Saved block normalization config differs')
    state = checkpoint.get('model_state')
    current = model.state_dict()
    require(type(state) in (dict, type(current)) and state.keys() == current.keys(),
            'Block normalization checkpoint tensor inventory differs')
    for name, value in state.items():
        target = current[name]
        require(type(value) is torch.Tensor and value.shape == target.shape and value.dtype == target.dtype,
                'Block normalization checkpoint tensor shape/dtype differs: ' + name)
        require(bool(torch.isfinite(value).all()), 'Nonfinite block normalization checkpoint tensor: ' + name)



def validate_caches(model, caches, batch, start_pos):
    raise ValueError('Context-length pilot supports uncached logits only')
