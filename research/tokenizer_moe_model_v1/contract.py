import copy
from research.body_allocation_model_v1.contract import model_config as native_config, canonical, fingerprint, require, plain

POLICY = 'tokenizer_moe_model_v1'
ARMS = {'dense':(672,28492416,44,44), 'wide':(1344,30556800,44,44), 'moe':(336,30560896,80,48)}


def model_config(arm):
    require(arm in ARMS,'Exact dense/wide/moe arm required')
    return dict(policy=POLICY,arm=arm,backbone_template=native_config('incumbent'),
                ffn=dict(kind='top2of4_swiglu' if arm=='moe' else 'dense_swiglu',
                         hidden=ARMS[arm][0],experts=4 if arm=='moe' else 1,
                         top_k=2 if arm=='moe' else 1,router_scale=4. if arm=='moe' else None,
                         router_initialization_multiplier=2. if arm=='moe' else None,
                         tie_policy='lower_expert_index',capacity='no_drop',shared_experts=0,
                         aux_coefficient=.01 if arm=='moe' else 0.,aux_reduction='mean_over_four_layers'),
                initialization='Construct genuine native incumbent first;preserve shared body/tables;replace changed FFNs using subsequent CPU draws. Changed arms consume extra RNG;no equal changed-shape weight claim.')


def signature(cfg):
    value=plain(cfg);arm=value.get('arm')
    require(arm in ARMS and canonical(value)==canonical(model_config(arm)),'Exact own FFN model policy required')
    hidden,parameters,muon,adam=ARMS[arm]
    return dict(policy=POLICY,arm=arm,width=256,hidden=hidden,layers=4,heads=8,latent=64,
                context=128,vocab=50257,head_dim=32,parameters=parameters,muon_groups=muon,
                adam_groups=adam,model_fingerprint=fingerprint(value))
