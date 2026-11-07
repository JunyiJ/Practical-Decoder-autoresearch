"""Exact actual geometry, separate from the closed tiny functional guards."""
import hashlib
from research.body_allocation_model_v1.contract import model_config as native_config, canonical, fingerprint, require, plain

POLICY = 'native_mla_single_projection_distribution_v1'
DOMAIN = 'readout_projection_model_v1:projection:'
ARMS = {'native': (28492416, 44, 44), 'linear_projection1': (28557952, 45, 44),
        'tanh_projection1': (28557952, 45, 44)}


def projection_seed(base):
    require(type(base) is int and 0 <= base < 2**63, 'Explicit audited initializer role required')
    return int.from_bytes(hashlib.sha256((DOMAIN+str(base)).encode()).digest()[:8], 'big') % (2**63)


def model_config(arm):
    require(type(arm) is str and arm in ARMS, 'Explicit single-projection arm required')
    return dict(policy=POLICY, backbone_template=native_config('incumbent'),
                readout_projection=dict(arm=arm, projection_count=0 if arm == 'native' else 1,
                    transform='identity' if arm == 'native' else 'linear' if arm == 'linear_projection1' else 'tanh',
                    projection='biasfree_256x256', initializer_domain=DOMAIN,
                    projection_init='private_uniform_pm_1_over_16_after_native',
                    shared_untied_head=True, divisor=4., objective='genuine_cross_entropy',
                    optimizer='projection_muon_half_lr_common_native',
                    cache_identity='instance_and_weight_epoch'))


def signature(cfg):
    value = plain(cfg)
    arm = value.get('readout_projection', {}).get('arm')
    require(type(arm) is str and arm in ARMS and canonical(value) == canonical(model_config(arm)),
            'Exact own single-projection configuration required')
    count, muon, adam = ARMS[arm]
    return dict(policy=POLICY, arm=arm, width=256, layers=4, vocab=50257, context=128,
                parameters=count, muon_groups=muon, adam_groups=adam, model_fingerprint=fingerprint(value))
