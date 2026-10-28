"""Corrupt synthetic saved timing records; no trained artifacts or real timing."""
import copy
import hashlib
import json
import pytest
from src.research import tokenizer_wide_hybrid_timing as t
from src.research import tokenizer_wide_hybrid_timing_review as a


def fixtures():
    selection={f:dict(lr=.00125,divisor=4) for f in t.FAMILIES}
    ids=list(range(96));sha=t.token_digest(ids);results=[]
    for seed in t.SEEDS:
        counts={}
        def measure(f,b):
            key=(f,b);counts[key]=counts.get(key,0)+1
            # Warmups intentionally extreme, so including them breaks expected medians.
            duration=1000. if counts[key]<=2 else float(counts[key]-2)
            factor={'hybrid':2.,'gated_mla':1.,'kda':4.}[f]
            return dict(prefill_seconds=duration*factor/2,decode_seconds=duration*factor,
                        state_bytes={'hybrid':90,'gated_mla':100,'kda':120}[f]*b,
                        observed_mps_current_bytes=10000,observed_mps_driver_bytes=20000)
        results.append(t.measure_triple(seed,selection,ids,sha,
            check=lambda f,b:dict(passed=True,max_absolute_error=0.),measure=measure))
    return selection,sha,results


def test_complete_roundtrip_and_independent_summary(tmp_path):
    selection,sha,results=fixtures();path=tmp_path/'timing.json'
    path.write_text(json.dumps(results[0]));file_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    review=a.review_file(path,file_sha,3511,selection,sha)
    assert review['cells'][0]['median_decode_seconds']==9.
    summary=a.summarize(json.loads(json.dumps(results)),selection,sha)
    assert len(summary['cells'])==18 and summary['memory_gate'] and not summary['adopted']
    for row in summary['paired_latency_ratios']:
        assert row['seed_ratios']==[2.,2.,2.] if row['control']=='gated_mla' else row['seed_ratios']==[.5,.5,.5]
        assert row['median_ratio']==row['seed_ratios'][0]


@pytest.mark.parametrize('bad', ['missing_record','duplicate_record','reordered','warmup_missing',
    'throughput','state_changed','nan','zero_time','float_batch','bool_repeat','extra_sample',
    'missing_check','failed_check','float_seed','wrong_rank','wrong_hash','wrong_tokens','adopted','extra_top'])
def test_corrupt_evidence_rejected(bad):
    selection,sha,results=fixtures();r=results[0]
    if bad=='missing_record':r['records'].pop()
    elif bad=='duplicate_record':r['records'][1]=copy.deepcopy(r['records'][0])
    elif bad=='reordered':r['records'][0],r['records'][1]=r['records'][1],r['records'][0]
    elif bad=='warmup_missing':r['warmup_records'].pop()
    elif bad=='throughput':r['records'][0]['decode_tokens_per_second']+=1
    elif bad=='state_changed':r['records'][0]['state_bytes']+=1
    elif bad=='nan':r['records'][0]['decode_seconds']=float('nan')
    elif bad=='zero_time':r['records'][0]['decode_seconds']=0
    elif bad=='float_batch':r['records'][0]['batch']=1.
    elif bad=='bool_repeat':r['records'][0]['repeat']=False
    elif bad=='extra_sample':r['records'][0]['extra']=True
    elif bad=='missing_check':r['correctness'].pop()
    elif bad=='failed_check':r['correctness'][0]['passed']=False
    elif bad=='float_seed':r['seed']=3511.
    elif bad=='wrong_rank':r['selected_specs'][0]['divisor']=8
    elif bad=='wrong_hash':r['token_sha256']='0'*64
    elif bad=='wrong_tokens':r['fixed_train_token_ids'][0]=99
    elif bad=='adopted':r['adopted']=True
    else:r['extra']=True
    with pytest.raises(ValueError):a.review_result(r,3511,selection,sha)


@pytest.mark.parametrize('bad',['missing','duplicate','order','token_mismatch'])
def test_all_fresh_triples_required(bad):
    selection,sha,results=fixtures()
    if bad=='missing':results.pop()
    elif bad=='duplicate':results[2]=results[1]
    elif bad=='order':results.reverse()
    else:results[2]['token_sha256']='0'*64
    with pytest.raises(ValueError):a.summarize(results,selection,sha)


def test_memory_gate_requires_every_seed_batch_and_both_controls():
    selection,sha,results=fixtures()
    for field in ('records','warmup_records'):
        for row in results[2][field]:
            if row['batch']==8 and row['family']=='hybrid':row['state_bytes']=91*8
    assert a.summarize(results,selection,sha)['memory_gate'] is False


@pytest.mark.parametrize('bad',['hash','size','duplicate_key'])
def test_bounded_saved_file_binding(tmp_path,monkeypatch,bad):
    selection,sha,results=fixtures();path=tmp_path/'timing.json'
    text=json.dumps(results[0])
    if bad=='duplicate_key':text=text.replace('"seed": 3511','"seed": 0, "seed": 3511',1)
    path.write_text(text);file_sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if bad=='hash':file_sha='0'*64
    elif bad=='size':monkeypatch.setattr(a,'LIMIT',16)
    with pytest.raises(ValueError):a.review_file(path,file_sha,3511,selection,sha)
