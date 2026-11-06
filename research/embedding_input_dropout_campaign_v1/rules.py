"""Current pair geometry and original strict budgets, without future deletion credit."""
from research.body_allocation_model_v1.contract import require
ARMS=('input_zero','input_feature_p10');PROBABILITIES=(0.,.1)
PAIR_CAP=560*1024**2;RAW_CAP=448*1024**2;H200_CAP=112*1024**2
RESERVE=8*1024**3;METADATA=16*1024**2;EXTERNAL=1267904512;ENTRY=1458176
FLOOR=2*PAIR_CAP+RESERVE+METADATA+EXTERNAL+ENTRY;TARGET=FLOOR+2*1024**3
DESIGN=dict(kind='finite_resource_pilot_only',arms=list(ARMS),probabilities=list(PROBABILITIES),steps=20,targets=20480,common_lr=.00125,
    ema_snapshots=8,parameter_tensors=88,muon_groups=44,adamw_groups=44,scientific_quality=False,holdout_scored=False,
    original_full_trial_projection='2*1200/20*measured_training_loop_seconds<7200 for EACH arm,plus all actual original worker/parent guards.',
    equality='Same parameters/shared initialization,original data/recipe/targets,only training input feature mask differs;no new dropout site/rate search.')
def capacity(free,used=0):
    require(type(used) is int and 0<=used<=2*PAIR_CAP,'Only actual accepted current-pilot retained bytes credited once')
    floor=max(RESERVE,FLOOR-used)
    return dict(free_bytes=free,required_free=floor,wake_free=floor+ENTRY,cleanup_target=floor+2*1024**3,capacity_pass=free>=floor+ENTRY,mandatory_deficit=max(0,floor+ENTRY-free))
