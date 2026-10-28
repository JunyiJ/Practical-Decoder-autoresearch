"""Synthetic measurement callbacks and tiny storage views only; no model runs."""
from collections import Counter
import hashlib
import math
import struct
import pytest
import torch
from src.research import tokenizer_wide_hybrid_timing as t


def inputs():
    ids = list(range(96))
    selection = {f: dict(lr=.00125, divisor=4) for f in t.FAMILIES}
    return selection, ids, t.token_digest(ids)


def sample(family, batch):
    return dict(prefill_seconds=.25, decode_seconds=2.,
                state_bytes=(t.FAMILIES.index(family)+1)*100*batch,
                observed_mps_current_bytes=10000, observed_mps_driver_bytes=20000)


def run(seed=3511, check=None, measure=sample):
    selection, ids, sha = inputs()
    return t.measure_triple(seed, selection, ids, sha,
        check=check or (lambda f,b: dict(passed=True,max_absolute_error=1e-6)), measure=measure)


@pytest.mark.parametrize('seed', t.SEEDS)
def test_exact_order_barrier_and_complete_measurements(seed):
    calls=[]
    def check(f,b):
        assert all(c[0]=='check' for c in calls)
        calls.append(('check',f,b)); return dict(passed=True,max_absolute_error=0.)
    def measure(f,b):
        assert len([c for c in calls if c[0]=='check'])==6
        calls.append(('timing',f,b)); return sample(f,b)
    result=run(seed,check,measure)
    assert len(result['records'])==48 and len(result['warmup_records'])==12
    assert result['adopted'] is False
    for rows,n in [(result['records'],8),(result['warmup_records'],2)]:
        expected=[(b,r,p,f) for b in (1,8) for r in range(n)
                  for p,f in enumerate(t.order(seed,r))]
        assert [(r['batch'],r['repeat'],r['position'],r['family']) for r in rows]==expected
        assert all(r['decode_tokens_per_second']==32*r['batch'] for r in rows)
    assert result['token_sha256']==hashlib.sha256(struct.pack('<96I',*range(96))).hexdigest()


def test_rotations_balanced_across_three_seeds_but_not_one():
    for position in range(3):
        counts=Counter(t.order(s,r)[position] for s in t.SEEDS for r in range(8))
        assert set(counts.values())=={8}
    assert sorted(Counter(t.order(3511,r)[0] for r in range(8)).values())==[2,3,3]


@pytest.mark.parametrize('bad', [dict(passed=False,max_absolute_error=0.),
    dict(passed=True,max_absolute_error=math.nan),dict(passed=1,max_absolute_error=0.),
    dict(passed=True,max_absolute_error=-1.),dict(passed=True)])
def test_correctness_failure_prevents_all_timings(bad):
    checks=[]; measured=[]
    def check(f,b):
        checks.append((f,b))
        return bad if len(checks)==6 else dict(passed=True,max_absolute_error=0.)
    with pytest.raises(ValueError):run(check=check,measure=lambda f,b:measured.append((f,b)))
    assert len(checks)==6 and not measured


@pytest.mark.parametrize('field,value', [('prefill_seconds',0),('decode_seconds',math.inf),
    ('decode_seconds',-1),('state_bytes',0),('state_bytes',True),
    ('observed_mps_current_bytes',-1),('observed_mps_driver_bytes',2.5)])
def test_invalid_warmup_stops_before_more_calls(field,value):
    calls=[]
    def measure(f,b):
        calls.append((f,b)); row=sample(f,b);row[field]=value;return row
    with pytest.raises(ValueError):run(measure=measure)
    assert len(calls)==1


def test_state_inconsistency_after_warmups_is_rejected():
    calls=[]
    def measure(f,b):
        calls.append((f,b));row=sample(f,b)
        if len(calls)==7:row['state_bytes']+=1
        return row
    with pytest.raises(ValueError,match='state changed'):run(measure=measure)
    assert len(calls)==7


def test_worker_callback_exception_propagates_without_retry():
    calls=[]
    def measure(f,b):
        calls.append((f,b));raise TimeoutError('synthetic worker timeout')
    with pytest.raises(TimeoutError):run(measure=measure)
    assert len(calls)==1


@pytest.mark.parametrize('bad', ['seed','selection','hash','length','range','bool'])
def test_bad_contract_rejected_before_callbacks(bad):
    selection,ids,sha=inputs();seed=3511;calls=[]
    if bad=='seed':seed=3501
    elif bad=='selection':selection['kda']['divisor']=16
    elif bad=='hash':sha='0'*64
    elif bad=='length':ids.pop()
    elif bad=='range':ids[0]=50257
    else:ids[0]=True
    def cb(*args):calls.append(args)
    with pytest.raises(ValueError):t.measure_triple(seed,selection,ids,sha,check=cb,measure=cb)
    assert not calls


def test_storage_counts_unique_backing_capacity_not_views():
    base=torch.zeros(32,dtype=torch.float32); other=torch.zeros(3,dtype=torch.float64)
    assert t.allocated_bytes([base[:2],base[8:12],other,other])==32*4+3*8
    assert t.allocated_bytes([torch.empty(0)])==0
    with pytest.raises(ValueError,match='Meta'):t.allocated_bytes([torch.empty(1,device='meta')])
