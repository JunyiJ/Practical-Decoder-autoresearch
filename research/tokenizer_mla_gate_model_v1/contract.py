from research.body_allocation_model_v1.contract import model_config as native_config,canonical,fingerprint,require,plain

POLICY='tokenizer_mla_elementwise_gate_model_v1'
ARMS={'none':(28492416,44,44),'static_half':(28492416,44,44),'sigmoid_elementwise_zero':(28754560,48,44)}

def model_config(arm):
    require(type(arm) is str and arm in ARMS,'Explicit native MLA gate arm required')
    return dict(policy=POLICY,arm=arm,backbone_template=native_config('incumbent'),
        gate=dict(kind=arm,granularity='elementwise_per_head_channel',placement='weighted_values_before_concat_and_output',input='current_query_pre_normalized_attention_input',activation='sigmoid',bias=False,temperature=1.,matrix_shape=[256,256] if arm=='sigmoid_elementwise_zero' else None,initialization='zeros_no_rng_draw',optimizer_role='body_muon_half_nominal_lr_match_rms_adamw'),
        initialization='Genuine unchanged native construction first,then new zero gate matrices;all common weights and CPU draws exactly preserved.')

def signature(cfg):
    v=plain(cfg);arm=v.get('arm');require(arm in ARMS and canonical(v)==canonical(model_config(arm)),'Exact own native MLA elementwise-gate model policy required')
    p,m,a=ARMS[arm]
    return dict(policy=POLICY,arm=arm,width=256,hidden=672,layers=4,heads=8,latent=64,context=128,vocab=50257,head_dim=32,parameters=p,muon_groups=m,adam_groups=a,model_fingerprint=fingerprint(v))
