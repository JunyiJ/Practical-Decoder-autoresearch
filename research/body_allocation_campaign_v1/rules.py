"""One allocation question; both controls and all tuning/fresh costs are explicit."""
from research.body_allocation_model_v1.contract import require
ARMS=('incumbent','shallow','deeper')
LRS=(.000625,.00125,.0025)
WEIGHT_CAPS=dict(incumbent=560*1024**2,shallow=432*1024**2,deeper=432*1024**2)
RAW_CAPS=dict(incumbent=448*1024**2,shallow=336*1024**2,deeper=336*1024**2)
EXPORT_CAPS=dict(incumbent=112*1024**2,shallow=96*1024**2,deeper=96*1024**2)
RESERVE=8*1024**3
METADATA=256*1024**2
# Previously measured non-payload filesystem consumption during resume001,
# with cause unproved. Retain this extra allowance rather than assume it vanished.
FILESYSTEM_HEADROOM=1267904512
ENTRY_HEADROOM=1458176
PEAK_WEIGHTS=4*sum(WEIGHT_CAPS.values())
FLOOR=RESERVE+METADATA+FILESYSTEM_HEADROOM+ENTRY_HEADROOM+PEAK_WEIGHTS
TARGET=FLOOR+2*1024**3
DESIGN=dict(arms=list(ARMS),candidate='deeper',diagnostic_only='shallow',steps=1200,targets=1228800,
    learning_rates=list(LRS),screens_per_arm=3,fresh_seeds=3,fresh_trajectories=9,
    selection='For each arm select minimum full-validation H200 CE over its three rates; lower LR breaks exact ties.',
    comparisons=['deeper-minus-incumbent','deeper-minus-shallow'],mean_delta_threshold=-.02,every_seed_negative=True,
    blocks=[2048,4096,8192],draws=2000,bootstrap_generator='PCG64',quantile='linear',upper_quantile=1-.05/6,
    simultaneous_scope='Bonferroni over two contrasts and three block lengths, conditional on these trained seeds and reused validation text.',
    rescue=False,replacement_seeds=False,retry_failed_attempt=False,holdout_scored=False,cost_only_adoption=False,
    retained_high_water='Pilot plus all screens: four endpoint pairs per arm. After exact dependency-closed pilot/loser retirement: one selected plus three fresh pairs per arm.',
    no_retirement_credit_until_actual=True,selected_and_fresh_protected=True,
    parametrization='Native same constructor scheme, inherited width rates and UNIT residual branches; no claim of shared weights or optimal depth transfer.',
    compute_axis='Equal sampled targets and equally bounded family LR tuning; measured time/memory/cache reported separately, not equal training FLOPs.')

def capacity(free,used=0):
    require(type(used) is int and 0<=used<=PEAK_WEIGHTS,'Only actual accepted current-panel retained weight bytes are credited')
    required=max(RESERVE,FLOOR-used)
    return dict(free_bytes=free,required_free=required,cleanup_target=required+2*1024**3,
        mandatory_deficit=max(0,required-free),target_deficit=max(0,required+2*1024**3-free),capacity_pass=free>=required)
