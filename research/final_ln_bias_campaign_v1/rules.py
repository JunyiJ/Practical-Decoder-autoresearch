"""Prospective fixed-recipe comparison; no learning-rate search or selection."""
from .evidence_contract import require
from research.local_mix_v1.output import WEIGHT_CAP
ARMS=('learned','zero')
DECAYS=dict(learned=.00006,zero=.00006)
COOLDOWNS=dict(learned=800,zero=800)
THRESHOLDS=dict(learned=1.,zero=1.)
BETAS=dict(learned=.999,zero=.999)
BIAS_KINDS=dict(learned='learned',zero='fixed_zero')
PILOT_STARTS=dict(learned=13,zero=13)
LR=.00125
RESERVE=8*1024**3
METADATA=256*1024**2
REQUIRED=6*WEIGHT_CAP+METADATA+RESERVE
QUALITY_DESIGN=dict(final_ln_bias=BIAS_KINDS,parameter_counts=dict(learned=28492416,zero=28492160),parameter_matched=False,final_ln_epsilon=1e-5,final_ln_scale='learned_both',adamw_beta2=BETAS,adamw_beta1=.9,adamw_epsilon=1e-8,muon="unchanged",final_normalization="layer_affine",pre_branch_norms="eight_unchanged_RMSNorms",gradient_clip_thresholds=THRESHOLDS,arms=list(ARMS),decays=DECAYS,peak_learning_rate=LR,lr_search_budget_per_arm=0,
    screens=0,lr_rescue=False,fresh_trajectories=6,fresh_pairs=3,
    fresh_order=[['learned','zero'],['zero','learned'],['learned','zero']],
    steps=1200,batch=8,context=128,targets_per_trajectory=1228800,
    cooldown_starts=COOLDOWNS,cooldown_min_ratio=.1,h200_start=800,h200_interval=10,h200_snapshots=41,
    selection=None,claim='standard_recipe_fixed_common_learning_rate_not_separately_tuned_optima',
    mean_zero_minus_learned_strictly_below=-.02,every_seed_improves=True,
    block_sizes=[2048,4096,8192],bootstrap_draws=2000,bootstrap_generator='PCG64',
    gate_upper_quantile=1-.05/3,all_gate_upper_bounds_strictly_below=0.,cost_only_route=False,
    pilot_loss_selection=False,test_holdout_scored=False)
def validate_seeds(seeds):
    require(type(seeds) is dict and set(seeds)=={'pilot','fresh','evaluation','bootstrap'},'Exact fixed-recipe seeds; no screening seed')
    require(type(seeds['fresh']) is list and len(seeds['fresh'])==3,'Three fresh seeds required')
    values=[seeds['pilot'],*seeds['fresh'],seeds['evaluation'],seeds['bootstrap']]
    require(all(type(v) is int and 0<v<2**32 for v in values) and len(set(values))==6,'Six distinct positive integer seeds required')
def fresh_specs(seeds):
    validate_seeds(seeds)
    return [dict(slot=2*i+j,pair=i,arm=arm,final_ln_bias_kind=BIAS_KINDS[arm],adamw_beta2=BETAS[arm],grad_clip=THRESHOLDS[arm],decay=DECAYS[arm],cooldown_start=COOLDOWNS[arm],seed=seed,learning_rate=LR,steps=1200)
            for i,seed in enumerate(seeds['fresh']) for j,arm in enumerate(QUALITY_DESIGN['fresh_order'][i])]
def require_capacity(free,retained=0):
    require(type(free) is int and type(retained) is int and 0<=retained<=2*WEIGHT_CAP,'Exact resource-pilot retained bytes required')
    require(free>=REQUIRED-retained,'Complete three-pair retained-output requirement does not fit')
