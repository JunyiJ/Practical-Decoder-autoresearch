"""Prospective pure FFN product-normalization rules: no I/O, launcher, cleanup or adoption.

The caller must bind seed freshness, exact corpus/tokenizer/validation coverage,
source/config/checkpoints, resource readiness and independently audited receipts.
Booleans below stand for verified boundary receipts, never deletion permission.
"""
import math
import numpy as np

ARMS=('none','rms')
POLICY='tokenizer_ffn_product_norm_transfer_v1'
RATES=(.000625,.00125,.0025)
BLOCKS=(2048,4096,8192)
DRAWS=2000
GATE_QUANTILE=1-.05/3
MAX_BASE_BLOCKS=512  # Bound arithmetic workspace; not proof of validation coverage.


def require(condition,message):
    if not condition:raise ValueError(message)


def validate_seeds(seeds):
    require(type(seeds) is dict and set(seeds)=={'pilot','screen','fresh','evaluation','bootstrap'},'Exact seed reservation fields required')
    require(type(seeds['fresh']) is list and len(seeds['fresh'])==3,'Three fresh paired seeds required')
    values=[seeds['pilot'],seeds['screen'],*seeds['fresh'],seeds['evaluation'],seeds['bootstrap']]
    require(all(type(v) is int and 0<v<2**31 for v in values) and len(set(values))==len(values),'Distinct native positive bounded integer seeds required')


def _spec(stage,arm,seed,rate,steps):
    return dict(stage=stage,arm=arm,seed=seed,lr=rate,steps=steps,width=256,
                batch_size=8,context=128,batch_layout='paired_blocks512',policy=POLICY)


def pilot_specs(seeds):
    validate_seeds(seeds)
    return [_spec('pilot',arm,seeds['pilot'],.0025,20) for arm in ARMS]


def screen_specs(seeds):
    validate_seeds(seeds)
    return [_spec('screen',arm,seeds['screen'],lr,1200)
            for lr,order in zip(RATES,(ARMS,ARMS[::-1],ARMS)) for arm in order]


def validate_selection(selection):
    require(type(selection) is dict and set(selection)==set(ARMS)
            and all(type(v) is float and v in RATES for v in selection.values()),'Exactly two allowed floating-point LR choices required')


def confirmation_specs(seeds,selection):
    validate_seeds(seeds);validate_selection(selection)
    return [_spec('confirm',arm,seed,selection[arm],1200)
            for seed,order in zip(seeds['fresh'],(ARMS,ARMS[::-1],ARMS)) for arm in order]


def checked_prefix(records,specs,*,quality=True,complete=False):
    require(type(records) is list and len(records)<=len(specs),'Bounded ordered list of accepted records required')
    if complete:require(len(records)==len(specs),'Complete stage required')
    for record,spec in zip(records,specs):
        fields={'status','h200_ce'} if quality else {'status'}
        require(type(record) is dict and set(record)==set(spec)|fields,'Exact endpoint schema required')
        require(all(type(record[k]) is type(v) and record[k]==v for k,v in spec.items()),'Endpoint order or identity differs')
        require(type(record['status']) is str and record['status']=='completed','Only completed reviewed endpoints permitted')
        if quality:
            ce=record['h200_ce']
            require(type(ce) in (int,float) and math.isfinite(ce) and ce>=0,'Finite nonnegative H200 CE required')


def choose_rates(seeds,records):
    checked_prefix(records,screen_specs(seeds),complete=True)
    return {arm:min((row for row in records if row['arm']==arm),key=lambda row:(row['h200_ce'],row['lr']))['lr'] for arm in ARMS}


def next_action(seeds,pilots,records,selection=None,*,pilots_released=False,losers_released=False):
    """Return one suggestion from an exact accepted prefix; never execute it."""
    require(type(pilots_released) is bool and type(losers_released) is bool,'Boundary flags must be native booleans')
    require(type(records) is list,'Accepted quality records must be a list')
    pilot=pilot_specs(seeds);screen=screen_specs(seeds)
    checked_prefix(pilots,pilot,quality=False)
    if len(pilots)<2:
        require(not records and selection is None and not pilots_released and not losers_released,'Quality/selection/release before pilots')
        return dict(kind='pilot',spec=pilot[len(pilots)])
    if not pilots_released:
        require(not records and selection is None and not losers_released,'Quality before audited pilot boundary')
        return dict(kind='review_pilot_boundary')
    checked_prefix(records[:6],screen)
    if len(records)<6:
        require(selection is None and not losers_released,'Selection/release before complete screen')
        return dict(kind='quality',spec=screen[len(records)])
    chosen=choose_rates(seeds,records[:6])
    if selection is None:
        require(len(records)==6 and not losers_released,'Fresh records/release before persisted selection')
        return dict(kind='save_selection',value=chosen)
    validate_selection(selection);require(selection==chosen,'Persisted selection differs')
    if not losers_released:
        require(len(records)==6,'Fresh records before audited screen boundary')
        return dict(kind='review_screen_boundary')
    queue=screen+confirmation_specs(seeds,selection)
    checked_prefix(records,queue)
    return dict(kind='quality',spec=queue[len(records)]) if len(records)<12 else dict(kind='review_final',adopted=False)


def screening_retention(seeds,records,selection):
    chosen=choose_rates(seeds,records);validate_selection(selection)
    require(selection==chosen,'Retention cannot alter scientific selection')
    selected=[i for i,row in enumerate(records) if row['lr']==selection[row['arm']]]
    return dict(selected_indices=selected,losing_indices=[i for i in range(6) if i not in selected],
                fresh_pairs_always_protected=True,deletion_authorized=False)


def _quantile(values,q):
    a=np.sort(np.asarray(values,dtype=np.float64));position=(len(a)-1)*q
    lo,hi=math.floor(position),math.ceil(position)
    return float(a[lo]+(a[hi]-a[lo])*(position-lo))


def paired_statistics(seeds,loss_sums,counts):
    """[fresh seed order, none/rms, aligned 2048-token blocks].

    PCG64 is reset to the declared bootstrap seed for each block size. Each draw
    uses common text-block indices for all three trained pairs; group sums before
    resampling and divide by sampled target counts, including a possible short tail.
    This arithmetic alone cannot verify target identity/order/checkpoint provenance.
    """
    validate_seeds(seeds)
    raw=np.asarray(loss_sums);n=np.asarray(counts)
    require(n.ndim==1 and 4<=len(n)<=MAX_BASE_BLOCKS and n.dtype.kind in 'iu'
            and np.all(n>0) and np.all(n<=2048) and np.all(n[:-1]==2048),'Aligned full2048 blocks and optional final tail required')
    require(raw.dtype.kind in 'fiu' and raw.shape==(3,2,len(n)) and np.isfinite(raw).all() and np.all(raw>=0),'Complete finite nonnegative three-pair loss panel required')
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            loss=raw.astype(np.float64);weights=n.astype(np.float64);delta=loss[:,1]-loss[:,0]
            ds=delta.sum(axis=1)/weights.sum();bounds={}
            for size in BLOCKS:
                starts=np.arange(0,len(n),size//2048)
                grouped_counts=np.add.reduceat(weights,starts)
                grouped_delta=np.add.reduceat(delta.mean(axis=0),starts)
                rng=np.random.Generator(np.random.PCG64(seeds['bootstrap']))
                indices=rng.integers(0,len(starts),size=(DRAWS,len(starts)))
                draws=grouped_delta[indices].sum(axis=1)/grouped_counts[indices].sum(axis=1)
                bounds[str(size)]=dict(gate_upper=_quantile(draws,GATE_QUANTILE),
                                      descriptive_95pct=[_quantile(draws,.025),_quantile(draws,.975)])
            result=dict(seed_deltas=ds.tolist(),mean_delta=float(ds.mean()),bounds=bounds)
    except FloatingPointError as exc:raise ValueError('Nonfinite bootstrap arithmetic') from exc
    quality_gate(result)  # Also reject nonfinite computed statistics.
    return result


def quality_gate(statistics):
    """Strict arithmetic eligibility, always adopted=False pending full review."""
    require(type(statistics) is dict and set(statistics)=={'seed_deltas','mean_delta','bounds'},'Exact statistics schema required')
    require(type(statistics['seed_deltas']) is list and len(statistics['seed_deltas'])==3,'Three seed deltas required')
    ds=statistics['seed_deltas'];mean=statistics['mean_delta'];bounds=statistics['bounds']
    require(all(type(v) in (int,float) and math.isfinite(v) for v in [*ds,mean]),'Finite native numeric deltas required')
    require(math.isclose(mean,sum(ds)/3,rel_tol=0,abs_tol=1e-12),'Mean differs from three paired deltas')
    require(type(bounds) is dict and set(bounds)=={str(x) for x in BLOCKS},'All three declared block-size bounds required')
    upper=[]
    for row in bounds.values():
        require(type(row) is dict and set(row)=={'gate_upper','descriptive_95pct'},'Exact bound schema required')
        interval=row['descriptive_95pct']
        require(type(interval) is list and len(interval)==2,'Two descriptive interval endpoints required')
        values=[row['gate_upper'],*interval]
        require(all(type(v) in (int,float) and math.isfinite(v) for v in values)
                and values[1]<=values[2]<=values[0],'Finite ordered interval required')
        upper.append(values[0])
    return dict(strict_quality=bool(mean<-.02 and all(v<0 for v in ds) and all(v<0 for v in upper)),adopted=False)
