"""Own truthful byte-input model identity; SOURCE only, no runtime admission."""
from dataclasses import dataclass
import json

POLICY = 'rsi_byte_input_model_math_v2'
VARIANTS = ('native', 'nonlexical_codes', 'raw_bytes')
NATIVE_MODEL_JSON = '{"attention_logit_scale":0.5,"attention_logit_scale_width_policy":"tokenizer_mha_gqa_score_v1","attention_width_policy":"mla_quarter_head32","attn_dropout":0.0,"attn_type":"mla","base_width":64,"block_normalization":{"affine":true,"bias":false,"dtype":"float32","epsilon":1.1920928955078125e-07,"kind":"rms_incumbent","placement":"pre_attention_and_pre_ffn","policy":"tokenizer_block_norm_fp32_v1","width":256},"block_size":128,"depth_width_policy":"depth4_unscaled","dim":256,"dropout":0.0,"embedding_backward":{"cpu_backward":"native_reference","dtype":"float32","forward":"native_dense_embedding","max_gradient_tokens":1024,"mps_autograd_order":1,"mps_backward":"unique_membership_matmul_index_copy","padding_idx":null,"policy":"tokenizer_grouped_mps_embedding_backward_v1","scale_grad_by_freq":false,"sparse":false,"version":1,"vocab_size":50257,"width":256},"ffn":"swiglu","ffn_bias":false,"ffn_width_policy":"swiglu_21_8","final_ln_bias_kind":"learned","final_ln_bias_policy":"mla4_final_ln_bias_v1","final_ln_scale_kind":"learned","final_ln_scale_policy":"mla4_final_ln_scale_v1","mlp_type":"dense","n_heads":8,"n_layers":4,"norm_type":"rmsnorm","output_bias_kind":"none","output_bias_prior":{"corpus_sha256":"86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed","prior_sha256":"470924d478e490521ed589c1ec38131ffad17d564990bc18dbb213e1fc04d792","tokenizer":{"implementation":"tiktoken","name":"gpt2","pattern":"\'(?:[sdmt]|ll|ve|re)| ?\\\\p{L}++| ?\\\\p{N}++| ?[^\\\\s\\\\p{L}\\\\p{N}]++|\\\\s++$|\\\\s+(?!\\\\S)|\\\\s","special_token_policy":"ordinary_text","special_tokens":{"<|endoftext|>":50256},"split_unit":"raw_text","version":"0.14.0","vocab_size":50257,"vocabulary_sha256":"bdc599f0a294cff99d60cf83aed3870857b30bc0d1d06164cae137831b55257c"},"train_fraction":0.8,"train_ids_sha256":"a41661e1634891819bb28a06227e211273be209098c878aecb034ae5864f834d","val_fraction":0.1,"version":2},"output_bias_width_policy":"tokenizer_attention_no_output_bias","parameterization":"mup","readout_norm_epsilon":1e-05,"readout_norm_kind":"layer_affine","readout_norm_width_policy":"mla4_final_ln_scale_v1","residual_gain_kind":"unit","residual_gain_width_policy":"mla4_fixed_complete_branch_gain_v1","rope":1,"tokenizer_architecture_policy":"tokenizer_mla_final_ln_scale_v1","vocab_size":50257}'


def require(value, message):
    if not value:
        raise ValueError(message)


def same_typed(a, b):
    if type(a) is not type(b):
        return False
    if type(b) is dict:
        return a.keys() == b.keys() and all(same_typed(a[k], v) for k, v in b.items())
    if type(b) in (tuple, list):
        return len(a) == len(b) and all(same_typed(x, y) for x, y in zip(a, b))
    return a == b


def native_model():
    return json.loads(NATIVE_MODEL_JSON)


def native_config():
    from omegaconf import OmegaConf
    return OmegaConf.create(native_model())


@dataclass(frozen=True)
class Spec:
    variant: str
    composer_seed: object = None
    codebook_identity: object = None

    def __post_init__(self):
        signature(self)


def signature(spec):
    require(type(spec) is Spec and type(spec.variant) is str and spec.variant in VARIANTS, 'Own immutable variant')
    if spec.variant == 'native':
        require(spec.composer_seed is None and spec.codebook_identity is None, 'Native has no composer/codebook')
    else:
        require(type(spec.composer_seed) is int and 0 <= spec.composer_seed < 2**63, 'Explicit private composer initializer')
        require(type(spec.codebook_identity) is tuple and len(spec.codebook_identity) == 10 and
                spec.codebook_identity[0] == 'rsi_byte_codebook_v1' and
                spec.codebook_identity[1] == spec.variant and spec.codebook_identity[4:6] == (50257, 130),
                'Exact own matching codebook identity')
    return (POLICY, spec.variant, spec.composer_seed, spec.codebook_identity, 256, 50257, 128)


def model_config(spec):
    signature(spec)
    return dict(policy=POLICY, variant=spec.variant, native_initializer_recipe=native_model(),
                input_embedding_kind='native_grouped' if spec.variant == 'native' else 'rsi_byte_composer_math_v1',
                composer_seed=spec.composer_seed, codebook_identity=spec.codebook_identity,
                primary_objective='ordinary_next_token_cross_entropy', cache='unsupported')


def counts(variant):
    require(type(variant) is str and variant in VARIANTS, 'Own grouping variant')
    return (88, 28492416, 44, 44) if variant == 'native' else (105, 16351859, 53, 52)


def primitive(value):
    if value is None:
        return ('none',)
    if type(value) in (str, bool, int, float):
        return (type(value).__name__, value)
    if type(value) in (tuple, list):
        items = tuple(primitive(v) for v in value)
        if all(x is not None for x in items):
            return (type(value).__name__, items)
    return None
