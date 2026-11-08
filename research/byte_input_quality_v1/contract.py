"""Pure campaign rules; no model import, role draws or dispatch."""
import math
ARMS=('native','raw_bytes','nonlexical_codes')
LRS=(.000625,.00125,.0025)
STEPS=1200
OUTPUT_CAP=10*1024**3
ENTRY_BYTES=20*1024**3
DESIGN=dict(arms=list(ARMS),candidate='raw_bytes',mechanism_control='nonlexical_codes',
 steps=1200,targets_per_trajectory=1228800,batch=8,context=128,learning_rates=list(LRS),
 screens=9,confirmations=9,fresh_paired_seeds=3,
 selection='Minimum complete H200 validation CE independently per arm; exact ties choose lower LR. Freeze all three choices after all nine screens and before confirmation.',
 primary_contrast='raw_bytes-minus-native',mechanistic_contrast='raw_bytes-minus-nonlexical_codes',
 primary_mean_threshold=-.02,mechanistic_mean_threshold=0.,every_seed_negative=True,
 blocks=[2048,4096,8192],bootstrap_draws=2000,bootstrap_generator='PCG64',quantile='linear',upper_quantile=1-.05/6,
 multiplicity='Two predeclared contrasts times three block lengths; Bonferroni six conditional-text bounds.',
 uncertainty='Conditional on three trained seed bundles and reused validation text; not population-seed confidence or holdout evidence.',
 practical_adoption='Raw bytes eligible only if all native-relative primary gates pass and every trajectory/evidence/resource requirement passes. Independent final acceptance required. Mechanistic failure must be reported separately; a native-only gain cannot establish lexical benefit.',
 nonlexical_fallback_adoption=False,pilot_selection=False,cost_only_adoption=False,holdout_scored=False,
 compute_axis='Equal sampled targets and equal bounded LR tuning, not equal wall time/FLOPs.',
 cooldown_start=800,cooldown_min_ratio=.1,h200_start=800,h200_interval=10,h200_snapshots=41,
 diagnostic_validation_steps=[1,301,601,901,1200],diagnostic_eval_batches=10,
 retain_all_screen_and_fresh_payloads=True,no_seed_replacement=True,no_partial_panel_selection=True,
 category='Evidence-grounded structural exploration; all screening and confirmation charged to this hypothesis.')

def slots():
 result=[]
 for i,lr in enumerate(LRS):
  for arm in ARMS[i:]+ARMS[:i]:result.append(dict(slot=len(result),stage='screen',bundle='screen',variant=arm,learning_rate=lr))
 for i in range(3):
  for arm in ARMS[i:]+ARMS[:i]:result.append(dict(slot=len(result),stage='confirmation',bundle='fresh'+str(i),variant=arm,learning_rate=None))
 return result

def select(rows):
 assert len(rows)==9 and [r['slot'] for r in rows]==list(range(9))
 for spec,row in zip(slots()[:9],rows):
  assert row['variant']==spec['variant'] and row['learning_rate']==spec['learning_rate']
  assert type(row['ce']) is float and math.isfinite(row['ce']) and row['ce']>=0
 return {arm:min((r for r in rows if r['variant']==arm),key=lambda r:(r['ce'],r['learning_rate']))['learning_rate'] for arm in ARMS}

def gates(deltas,bounds,threshold):
 assert len(deltas)==len(bounds)==3 and all(math.isfinite(v) for v in deltas+bounds)
 return dict(mean_below_threshold=sum(deltas)/3<threshold,every_seed_negative=all(v<0 for v in deltas),all_bounds_negative=all(v<0 for v in bounds))
