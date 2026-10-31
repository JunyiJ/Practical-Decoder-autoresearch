"""Finite activation inference workload and independent saved-value review.

No checkpoint loading or launch authority. Cost is descriptive; only the frozen
paired quality criterion can make an activation eligible for adoption.
"""
import copy
import math
from statistics import median

from .protocol import digest
from .tokenizer_activation_rules import FAMILIES, FRESH_SEEDS, confirmation_specs
from .tokenizer_wide_hybrid_timing import token_digest, _correctness, _measurement

BATCHES=(1,8)
WARMUPS=2
REPEATS=8
CACHE_BYTES=196608
# Physical slot names belong to the unchanged bounded writer's three inference
# buckets. These are NOT training/timing seeds or references to an old campaign.
SLOTS=(3511,3512,3513)
CONTRACT=dict(kind='activation_inference_v1',seeds=list(FRESH_SEEDS),storage_slots=list(SLOTS),
    families=list(FAMILIES),batches=list(BATCHES),warmups=WARMUPS,repeats=REPEATS,
    context_tokens=128,prefill_tokens=32,decode_tokens=64,token_source='first96_training_ids',
    token_hash_encoding='uint32-little-endian',dtype='float32',device='mps',cpu_threads=4,
    rtol=3e-4,atol=3e-5,rss_limit_bytes=4*1024**3,driver_limit_bytes=6*1024**3,
    reserve_bytes=8*1024**3,job_seconds=600,output_limit_bytes=5*1024**2,
    log_limit_bytes=4*1024**2,panel_limit_bytes=16*1024**2,adoption_cost_route=False)
SCOPE=('Two fresh H200 models resident. Cache allocation/loading outside timed regions. '
       '32-token prefill and64 teacher-forced decode calls, synchronized at interval boundaries. '
       'Cache bytes count unique backing storage. MPS allocator observations are not single-model '
       'deployment peaks. Fixed training tokens; no holdout. Cost is descriptive, not an adoption gate.')
SAMPLE_KEYS={'prefill_seconds','decode_seconds','state_bytes','observed_mps_current_bytes','observed_mps_driver_bytes'}


def order(seed,repeat):
    if type(seed) is not int or seed not in FRESH_SEEDS or type(repeat) is not int or not 0<=repeat<REPEATS:
        raise ValueError('Undeclared inference seed/repeat')
    return FAMILIES if (FRESH_SEEDS.index(seed)+repeat)%2==0 else FAMILIES[::-1]


def slot(seed):
    order(seed,0)
    return SLOTS[FRESH_SEEDS.index(seed)]


def header(seed,selection,token_ids,sha):
    order(seed,0);specs=[s for s in confirmation_specs(selection) if s['seed']==seed]
    if token_digest(token_ids)!=sha:raise ValueError('Timing training tokens changed')
    return dict(kind=CONTRACT['kind'],status='complete_pending_parent_review',seed=seed,
        contract=copy.deepcopy(CONTRACT),selected_specs=specs,fixed_train_token_ids=list(token_ids),token_sha256=sha,
        scope=SCOPE,adopted=False)


def sample(value,batch):
    result=_measurement(value,batch)
    if (result['state_bytes']!=CACHE_BYTES*batch
            or result['observed_mps_driver_bytes']>=CONTRACT['driver_limit_bytes']
            or result['observed_mps_current_bytes']>result['observed_mps_driver_bytes']):
        raise ValueError('Measured cache/MPS resource contract differs')
    return result


def measure_pair(seed,selection,token_ids,sha,*,check,measure):
    result=header(seed,selection,tuple(token_ids),sha)
    checks=[]
    # All four cache/reset checks must finish before even the first warmup.
    for batch in BATCHES:
        for family in order(seed,0):
            checks.append(dict(batch=batch,family=family,**_correctness(check(family,batch))))
    rows={}
    for field,phase,count in [('warmup_records','warmup',WARMUPS),('records','measured',REPEATS)]:
        rows[field]=[]
        for batch in BATCHES:
            for repeat in range(count):
                for position,family in enumerate(order(seed,repeat)):
                    rows[field].append(dict(batch=batch,family=family,phase=phase,repeat=repeat,position=position,
                        **sample(measure(family,batch),batch)))
    return dict(result,correctness=checks,**rows)


def review_result(result,seed,selection,sha):
    """Rebuild exact order/coverage/derived rates, excluding warmups from medians."""
    expected=header(seed,selection,result['fixed_train_token_ids'],sha)
    if (set(result)!=set(expected)|{'correctness','warmup_records','records'}
            or digest({k:result[k] for k in expected})!=digest(expected)):
        raise ValueError('Inference identity/workload differs')
    checks=[(b,f) for b in BATCHES for f in order(seed,0)]
    if type(result['correctness']) is not list or len(result['correctness'])!=len(checks):
        raise ValueError('Complete correctness barrier required')
    for row,(batch,family) in zip(result['correctness'],checks):
        if set(row)!={'batch','family','passed','max_absolute_error'} or digest([row['batch'],row['family']])!=digest([batch,family]):
            raise ValueError('Correctness order/schema differs')
        _correctness({k:row[k] for k in ('passed','max_absolute_error')})
    for field,phase,count in [('warmup_records','warmup',WARMUPS),('records','measured',REPEATS)]:
        sequence=[dict(batch=b,phase=phase,repeat=r,position=p,family=f)
                  for b in BATCHES for r in range(count) for p,f in enumerate(order(seed,r))]
        if type(result[field]) is not list or len(result[field])!=len(sequence):
            raise ValueError('Incomplete timing sequence')
        for row,identity in zip(result[field],sequence):
            if set(row)!=set(identity)|SAMPLE_KEYS|{'decode_tokens_per_second'} or digest({k:row[k] for k in identity})!=digest(identity):
                raise ValueError('Timing order/schema differs')
            checked=sample({k:row[k] for k in SAMPLE_KEYS},identity['batch'])
            rate=row['decode_tokens_per_second']
            if type(rate) not in (float,int) or not math.isfinite(rate) or not math.isclose(rate,checked['decode_tokens_per_second'],rel_tol=1e-12):
                raise ValueError('Throughput arithmetic differs')
    cells=[]
    for batch in BATCHES:
        for family in FAMILIES:
            rows=[r for r in result['records'] if r['batch']==batch and r['family']==family]
            cells.append(dict(seed=seed,batch=batch,family=family,state_bytes=CACHE_BYTES*batch,
                median_prefill_seconds=median(r['prefill_seconds'] for r in rows),
                median_decode_seconds=median(r['decode_seconds'] for r in rows)))
    return dict(status='measurements_reviewed_pending_provenance',seed=seed,cells=cells,adopted=False)


def summarize(results,selection,sha):
    if len(results)!=3:raise ValueError('All three fresh inference pairs required')
    cells=[c for result,seed in zip(results,FRESH_SEEDS) for c in review_result(result,seed,selection,sha)['cells']]
    index={(c['seed'],c['batch'],c['family']):c for c in cells};ratios=[]
    for batch in BATCHES:
        for metric in ('prefill','decode'):
            key='median_'+metric+'_seconds'
            values=[index[s,batch,'gelu_exact'][key]/index[s,batch,'silu'][key] for s in FRESH_SEEDS]
            if not all(math.isfinite(v) and v>0 for v in values):raise ValueError('Invalid paired timing ratio')
            ratios.append(dict(batch=batch,metric=metric,seed_ratios=values,median_ratio=median(values)))
    return dict(status='panel_reviewed_pending_provenance',cells=cells,gelu_over_silu_latency_ratios=ratios,
        adopted=False,aggregation='Median of three seed-paired ratios of eight-repeat median latency; warmups excluded. No cost adoption route.')
