"""Finite timing/statistics tests; no model/corpus/MPS benchmark."""
import copy
import math
from statistics import median

import pytest

from src.research import tokenizer_activation_timing as t

IDS=list(range(96))
SHA=t.token_digest(IDS)
SELECTION={'silu':.000625,'gelu_exact':.0025}


def measurement(family,batch):
    scale=2 if family=='gelu_exact' else 1
    return dict(prefill_seconds=.01*scale,decode_seconds=.1*scale,state_bytes=t.CACHE_BYTES*batch,
        observed_mps_current_bytes=1000000,observed_mps_driver_bytes=2000000)


def measured(seed=3611):
    return t.measure_pair(seed,SELECTION,IDS,SHA,check=lambda f,b:dict(passed=True,max_absolute_error=0.),measure=measurement)


def test_complete_three_pair_order_and_independent_summary():
    results=[measured(s) for s in t.FRESH_SEEDS]
    for seed,r in zip(t.FRESH_SEEDS,results):
        assert len(r['correctness'])==4 and len(r['warmup_records'])==8 and len(r['records'])==32
        assert r['records'][0]['family']==('gelu_exact' if seed==3612 else 'silu')
        for batch in t.BATCHES:
            rows=[v for v in r['records'] if v['batch']==batch]
            assert sum(v['position']==0 and v['family']=='silu' for v in rows)==4
    summary=t.summarize(results,SELECTION,SHA)
    assert len(summary['cells'])==12 and all(r['median_ratio']==2 for r in summary['gelu_over_silu_latency_ratios'])
    assert summary['adopted'] is False
    assert [t.slot(s) for s in t.FRESH_SEEDS]==[3511,3512,3513]


def test_correctness_barrier_precedes_all_warmups_and_failure_has_no_timing():
    calls=[]
    def check(f,b):
        calls.append(('check',f,b))
        if len(calls)==4:raise ValueError('cache failed')
        return dict(passed=True,max_absolute_error=0.)
    with pytest.raises(ValueError,match='cache failed'):
        t.measure_pair(3611,SELECTION,IDS,SHA,check=check,measure=lambda f,b:pytest.fail('timing before complete barrier'))
    assert len(calls)==4


@pytest.mark.parametrize('case',['seed','token','contract','scope','missing_check','failed_check','missing_row','order','nan','zero','state','driver','current','throughput','extra'])
def test_review_rejects_changed_or_incomplete_saved_measurements(case):
    r=copy.deepcopy(measured())
    if case=='seed':r['seed']=3612
    elif case=='token':r['fixed_train_token_ids'][0]=999
    elif case=='contract':r['contract']['repeats']=7
    elif case=='scope':r['scope']='single-model peak claim'
    elif case=='missing_check':r['correctness'].pop()
    elif case=='failed_check':r['correctness'][0]['passed']=False
    elif case=='missing_row':r['records'].pop()
    elif case=='order':r['records'][0],r['records'][1]=r['records'][1],r['records'][0]
    elif case=='nan':r['records'][0]['decode_seconds']=math.nan
    elif case=='zero':r['records'][0]['prefill_seconds']=0
    elif case=='state':r['records'][0]['state_bytes']-=4
    elif case=='driver':r['records'][0]['observed_mps_driver_bytes']=6*1024**3
    elif case=='current':r['records'][0]['observed_mps_current_bytes']=3000000
    elif case=='throughput':r['records'][0]['decode_tokens_per_second']*=2
    else:r['unexpected']=True
    with pytest.raises(ValueError):t.review_result(r,3611,SELECTION,SHA)


def test_warmups_excluded_and_ratio_uses_seed_medians():
    results=[measured(s) for s in t.FRESH_SEEDS]
    for i,r in enumerate(results):
        for row in r['warmup_records']:
            row['decode_seconds']=100;row['decode_tokens_per_second']=64*row['batch']/100
        for row in r['records']:
            if row['family']=='gelu_exact':
                row['decode_seconds']=(i+1)*.1;row['decode_tokens_per_second']=64*row['batch']/row['decode_seconds']
    report=t.summarize(results,SELECTION,SHA)
    for row in report['gelu_over_silu_latency_ratios']:
        if row['metric']=='decode':assert row['seed_ratios']==pytest.approx([1,2,3]) and row['median_ratio']==2
    with pytest.raises(ValueError):t.summarize(results[:2],SELECTION,SHA)
    with pytest.raises(ValueError):t.summarize(results[::-1],SELECTION,SHA)


@pytest.mark.parametrize('seed',[3511,True,3614])
def test_undeclared_seeds_rejected(seed):
    with pytest.raises(ValueError):measured(seed)
