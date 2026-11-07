import hashlib
from research.body_allocation_model_v1.contract import model_config as native_config, canonical, fingerprint, require, plain

POLICY = 'native_mla_readout_distribution_v1'
ARMS = {'native': (28492416, 44, 44), 'context_mix2': (28624000, 46, 45), 'probability_mix2': (28624000, 46, 45)}
DOMAIN = 'readout_mixture_model_v1:context:'


def context_seed(base):
    require(type(base) is int and 0 <= base < 2**63, 'Explicit audited initializer role required')
    return int.from_bytes(hashlib.sha256((DOMAIN+str(base)).encode()).digest()[:8], 'big') % (2**63)


def model_config(arm):
    require(type(arm) is str and arm in ARMS, 'Explicit readout arm required')
    return dict(policy=POLICY, backbone_template=native_config('incumbent'),
                readout_model=dict(arm=arm, components=1 if arm == 'native' else 2,
                    context='tanh_biasfree_256x256', mixture='biasfree_2x256_softmax',
                    context_init='independent_private_uniform_pm_1_over_sqrt256', router_init='zeros_no_rng',
                    initializer_domain=DOMAIN, shared_untied_head=True, divisor=4.,
                    optimizer='context_muon_half_lr_router_adamw_base',
                    cache_identity='instance_and_weight_epoch'))


def signature(cfg):
    value = plain(cfg)
    arm = value.get('readout_model', {}).get('arm')
    require(arm in ARMS and canonical(value) == canonical(model_config(arm)), 'Exact own readout policy required')
    count, muon, adam = ARMS[arm]
    return dict(policy=POLICY, arm=arm, width=256, layers=4, vocab=50257, context=128,
                parameters=count, muon_groups=muon, adam_groups=adam, model_fingerprint=fingerprint(value))
