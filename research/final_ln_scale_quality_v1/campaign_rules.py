"""Fixed common recipe, zero search budget and unchanged strict statistical gates."""
import math
import numpy as np
from research.final_ln_scale_campaign_v1.rules import ARMS,DECAYS,COOLDOWNS,THRESHOLDS,BETAS,SCALE_KINDS,LR,REQUIRED,QUALITY_DESIGN,validate_seeds,fresh_specs
from research.final_ln_scale_campaign_v1.evidence_contract import require
BLOCKS=(2048,4096,8192)
DRAWS=2000
GATE_QUANTILE=1-.05/3
MAX_BASE_BLOCKS=512

def _quantile(values,q):
    a=np.sort(np.asarray(values,dtype=np.float64));position=(len(a)-1)*q
    lo,hi=math.floor(position),math.ceil(position)
    return float(a[lo]+(a[hi]-a[lo])*(position-lo))


def paired_statistics(seeds,loss_sums,counts):
    """[fresh seed order, learned/fixed-unit, aligned 2048-token blocks].

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
