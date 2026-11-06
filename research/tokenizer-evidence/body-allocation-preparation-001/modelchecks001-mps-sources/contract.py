"""Three exact geometries, not a general override of the incumbent policies."""
import copy
import hashlib
import json
from pathlib import Path
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-slot03-evidence001/config.json'
_BASE_CONFIG = json.loads(BASE.read_text())
POLICY = 'tokenizer_body_allocation_v1'
ARMS = dict(incumbent=(4,256,672,28492416), shallow=(4,192,1024,22057472), deeper=(7,192,504,22029632))

def require(ok, message):
    if not ok:raise ValueError(message)

def plain(value):
    return OmegaConf.to_container(value,resolve=True) if OmegaConf.is_config(value) else copy.deepcopy(value)

def canonical(value):return json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False)
def fingerprint(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

def model_config(arm):
    require(arm in ARMS,'Unknown allocation arm')
    l,d,h,_=ARMS[arm];m=copy.deepcopy(_BASE_CONFIG['model'])
    m.update(tokenizer_architecture_policy=POLICY,body_allocation_arm=arm,dim=d,n_heads=d//32,n_layers=l,
        ffn_hidden_dim=h,ffn_width_policy='body_allocation_explicit_swiglu_v1',
        depth_width_policy='body_allocation_unit_v1',attention_width_policy='body_allocation_mla_quarter_head32_v1',
        residual_gain_width_policy='body_allocation_unit_branch_v1',
        readout_norm_width_policy='body_allocation_final_affine_v1',
        final_ln_bias_policy='body_allocation_learned_v1',final_ln_scale_policy='body_allocation_learned_v1')
    m['block_normalization']['width']=d
    m['embedding_backward'].update(width=d,policy='body_allocation_grouped_embedding_v1')
    return m

def signature(cfg):
    value=plain(cfg);arm=value.get('body_allocation_arm');require(arm in ARMS,'Explicit allocation arm required')
    require(canonical(value)==canonical(model_config(arm)),'Exact allocation configuration required')
    l,d,h,p=ARMS[arm]
    return dict(policy=POLICY,arm=arm,layers=l,width=d,hidden=h,heads=d//32,latent=d//4,
        context=128,vocab=50257,head_dim=32,parameters=p,muon_groups=11*l,adam_groups=10*l+4,
        model_fingerprint=fingerprint(value))

def config(arm,steps=4,seed=91041,lr=.00125):
    require(type(steps) is int and steps in (4,20,1200),'Bounded supported horizon required')
    require(type(seed) is int and seed>=0,'Explicit nonnegative seed required')
    require(type(lr) is float and lr in (.000625,.00125,.0025),'Finite prospective LR allowance required')
    cfg=copy.deepcopy(_BASE_CONFIG);cfg['model']=model_config(arm)
    t=cfg['training'];start={4:2,20:13,1200:800}[steps]
    t.update(seed=seed,max_iters=steps,learning_rate=lr,resume_from=None)
    t['lr_schedule'].update(start_step=start,total_steps=steps)
    t['weight_averaging'].update(start_step=start,total_steps=steps,interval=10 if steps==1200 else 1)
    return OmegaConf.create(cfg)
