"""Exact actual FFN geometry; separate from closed tiny-component contracts."""
import hashlib
from research.body_allocation_model_v1.contract import (
    model_config as native_config, canonical, fingerprint, plain, require,
)

POLICY = 'native_mla_matched_matrix_ffn_distribution_v1'
DOMAIN = 'relu_squared_model_v1:ungated:'
ARMS = ('native_swiglu', 'ungated_silu', 'ungated_relu_squared')


def ungated_seed(base):
    require(type(base) is int and 0 <= base < 2**63, 'Explicit audited initializer required')
    return int.from_bytes(hashlib.sha256((DOMAIN+str(base)).encode()).digest()[:8], 'big') % (2**63)


def model_config(arm):
    require(type(arm) is str and arm in ARMS, 'Exact own FFN arm required')
    return dict(policy=POLICY, native_template=native_config('incumbent'),
                ffn=dict(arm=arm, hidden=672 if arm == 'native_swiglu' else 1008,
                    equation='silu_gate_times_up' if arm == 'native_swiglu' else
                             'silu_up' if arm == 'ungated_silu' else 'relu_up_squared',
                    bias=False, dropout=0., rescale=1., normalization='none',
                    initializer='genuine_native_first_then_private_default_linear_uniform',
                    initializer_domain=DOMAIN, matrix_elements_per_layer=516096),
                readout=dict(kind='untied_native_logits', divisor=4., objective='genuine_cross_entropy'),
                optimizer='native_muon_half_lr_adamw_eps1e6_independent_decay',
                cache_identity='own_instance_signature_weight_epoch')


def signature(cfg):
    value = plain(cfg)
    require(type(value) is dict and type(value.get('ffn')) is dict, 'Own FFN config required')
    arm = value['ffn'].get('arm')
    require(type(arm) is str and arm in ARMS and canonical(value) == canonical(model_config(arm)),
            'Exact own FFN config required; no foreign template or override')
    return dict(policy=POLICY, arm=arm, width=256, layers=4, vocab=50257, context=128,
                hidden=672 if arm == 'native_swiglu' else 1008, parameters=28492416,
                muon_groups=44 if arm == 'native_swiglu' else 40, adam_groups=44,
                model_fingerprint=fingerprint(value))


def parameter_shapes(arm):
    """Ordered actual parameter inventory, without constructing any tensors."""
    require(type(arm) is str and arm in ARMS, 'Own arm required')
    result = {'transformer.wte.weight': (50257, 256)}
    attention = dict(dkv=(64, 256), uk=(256, 64), uv=(256, 64), dq=(64, 256),
                     uq=(256, 64), qr=(256, 64), kr=(32, 256), out=(256, 256))
    for i in range(4):
        prefix = f'transformer.h.{i}.'
        result[prefix+'ln1.weight'] = (256,)
        result[prefix+'ln2.weight'] = (256,)
        for name, shape in attention.items():
            result[prefix+'attn.'+name+'.weight'] = shape
            result[prefix+'attn.'+name+'.bias'] = (shape[0],)
        hidden = 672 if arm == 'native_swiglu' else 1008
        if arm == 'native_swiglu':
            result[prefix+'mlp.linear_gate.weight'] = (hidden, 256)
        result[prefix+'mlp.linear_up.weight'] = (hidden, 256)
        result[prefix+'mlp.linear_down.weight'] = (256, hidden)
    result['transformer.ln_f.weight'] = (256,)
    result['transformer.ln_f.bias'] = (256,)
    result['lm_head.weight'] = (50257, 256)
    return result
