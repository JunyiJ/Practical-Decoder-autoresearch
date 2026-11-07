"""Exact native/duplicate-next/future-two model identities; no model construction."""
from research.body_allocation_model_v1.contract import (
    model_config as native_config, canonical, fingerprint, plain, require,
)

POLICY = 'causal_native_auxiliary_future_token_v1'
ARMS = ('native_next', 'duplicate_next', 'future_two')


def model_config(arm):
    require(type(arm) is str and arm in ARMS, 'Exact own future-token arm required')
    return dict(policy=POLICY, arm=arm, native_template=native_config('incumbent'),
                auxiliary=dict(kind='none' if arm == 'native_next' else 'copied_last_block_and_final_norm',
                    target_offset=0 if arm == 'native_next' else 1 if arm == 'duplicate_next' else 2,
                    primary_weight=1. if arm == 'native_next' else .5,
                    auxiliary_weight=0. if arm == 'native_next' else .5,
                    initializer='native_first_then_independent_deepcopy_no_rng',
                    readout='share_single_native_unembedding_not_input_embedding',
                    inference='primary_native_path_only'),
                readout=dict(kind='untied_native_logits', divisor=4., objective='genuine_cross_entropy'),
                optimizer='native_muon_half_lr_adamw_eps1e6_independent_decay',
                cache_identity='own_instance_signature_weight_epoch',
                target_layout='explicit_split_local_T_plus_2_no_wrap_or_padding')


def signature(cfg):
    value = plain(cfg)
    require(type(value) is dict and type(value.get('arm')) is str and value['arm'] in ARMS,
            'Own future-token configuration required')
    arm = value['arm']
    require(canonical(value) == canonical(model_config(arm)), 'Exact own configuration; overrides forbidden')
    auxiliary = arm != 'native_next'
    return dict(policy=POLICY, arm=arm, width=256, layers=4, vocab=50257, context=128,
                hidden=672, parameters=29183008 if auxiliary else 28492416,
                parameter_tensors=111 if auxiliary else 88,
                muon_groups=55 if auxiliary else 44, adam_groups=56 if auxiliary else 44,
                model_fingerprint=fingerprint(value))


def parameter_shapes(arm):
    require(type(arm) is str and arm in ARMS, 'Own parameter inventory arm required')
    native = {'transformer.wte.weight': (50257, 256)}
    attention = dict(dkv=(64, 256), uk=(256, 64), uv=(256, 64), dq=(64, 256),
                     uq=(256, 64), qr=(256, 64), kr=(32, 256), out=(256, 256))
    for i in range(4):
        prefix = f'transformer.h.{i}.'
        native[prefix+'ln1.weight'] = (256,)
        native[prefix+'ln2.weight'] = (256,)
        for name, shape in attention.items():
            native[prefix+'attn.'+name+'.weight'] = shape
            native[prefix+'attn.'+name+'.bias'] = (shape[0],)
        for name, shape in dict(linear_gate=(672, 256), linear_up=(672, 256), linear_down=(256, 672)).items():
            native[prefix+'mlp.'+name+'.weight'] = shape
    native['transformer.ln_f.weight'] = (256,)
    native['transformer.ln_f.bias'] = (256,)
    native['lm_head.weight'] = (50257, 256)
    result = {'native.'+name: shape for name, shape in native.items()}
    if arm != 'native_next':
        for name, shape in native.items():
            if name.startswith('transformer.h.3.'):
                result['auxiliary.block.'+name[len('transformer.h.3.'):]] = shape
        result['auxiliary.norm.weight'] = (256,)
        result['auxiliary.norm.bias'] = (256,)
    return result
