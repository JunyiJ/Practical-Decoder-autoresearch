"""Actual retained pair measurements and bounded independent saved-value checks."""
import copy
import gc
import json
import os
from pathlib import Path
import pytest
import torch
from research.block_norm_v1 import inference as i, campaign_contract as c


@pytest.fixture(scope='module')
def measured():
    plan=json.loads(Path(os.environ['BLOCK_NORM_PLAN_PATH']).read_text())
    before=torch.get_rng_state().clone();mps_before=torch.mps.get_rng_state().clone()
    result=i.measure_pair(plan['requests'],0)
    assert torch.equal(before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state())
    encoded=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();assert len(encoded)<128*1024
    path=Path(plan['result_path'])
    with path.open('xb') as f:f.write(encoded)
    saved=json.loads(path.read_text());review=i.review(saved,plan['requests'],0)
    yield dict(result=saved,review=review,identity=i.identity(plan['requests'],0),requests=plan['requests'])
    assert path.read_bytes()==encoded
    gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def test_real_pair_full_cache_reset_and_synchronized_timings(measured,record_property):
    result=measured['result'];review=measured['review']
    assert len(result['correctness'])==4 and len(result['warmup_records'])==8 and len(result['records'])==32
    assert all(row['passed'] for row in result['correctness']) and not review['quality_adopted']
    # Independent sorted-midpoint calculation excludes every warmup.
    for cell in review['cells']:
        rows=[r for r in result['records'] if r['batch']==cell['batch'] and r['arm']==cell['arm']]
        for metric in ('prefill','decode'):
            times=sorted(r[metric+'_seconds'] for r in rows)
            assert cell['median_'+metric+'_seconds']==(times[3]+times[4])/2
        for row in rows:assert row['decode_tokens_per_second']==64*cell['batch']/row['decode_seconds']
        assert cell['state_bytes']==4*128*(64+32)*4*cell['batch']
    record_property('actual_inference_review',dict(review=review,correctness=result['correctness'],rng_preserved=True,token_sha256=result['token_sha256']))


@pytest.mark.parametrize('case',['changed_tokens','wrong_request','model_parameters','changed_contract','wrong_pair','adopted',
    'missing_correctness','failed_correctness','nonfinite_error','missing_warmup','missing_measurement','swapped_order',
    'negative_time','nonfinite_time','cache_bytes','driver_limit','current_above_driver','wrong_rate','extra_field'])
def test_invalid_measurement_or_binding_rejected(measured,case):
    value=copy.deepcopy(measured['result']);row=value['records'][0]
    if case=='changed_tokens':value['fixed_train_token_ids'][0]+=1
    elif case=='wrong_request':value['requests_sha256']='0'*64
    elif case=='model_parameters':value['models'][i.ARMS[0]]['parameters']+=1
    elif case=='changed_contract':value['contract']['decode_tokens']=63
    elif case=='wrong_pair':value['pair_index']=1
    elif case=='adopted':value['quality_adopted']=True
    elif case=='missing_correctness':value['correctness'].pop()
    elif case=='failed_correctness':value['correctness'][0]['passed']=False
    elif case=='nonfinite_error':value['correctness'][0]['max_absolute_error']=float('nan')
    elif case=='missing_warmup':value['warmup_records'].pop()
    elif case=='missing_measurement':value['records'].pop()
    elif case=='swapped_order':value['records'][0],value['records'][1]=value['records'][1],value['records'][0]
    elif case=='negative_time':row['decode_seconds']=-1.
    elif case=='nonfinite_time':row['prefill_seconds']=float('inf')
    elif case=='cache_bytes':row['state_bytes']+=4
    elif case=='driver_limit':row['observed_mps_driver_bytes']=6*1024**3
    elif case=='current_above_driver':row['observed_mps_current_bytes']=row['observed_mps_driver_bytes']+1
    elif case=='wrong_rate':row['decode_tokens_per_second']*=2
    else:row['extra']=True
    with pytest.raises(ValueError):i.review_values(value,measured['identity'])


def test_wrong_arm_request_rejected_before_model_load(measured):
    requests=dict(measured['requests']);requests[i.ARMS[0]],requests[i.ARMS[1]]=requests[i.ARMS[1]],requests[i.ARMS[0]]
    with pytest.raises(ValueError,match='normalization arm'):i.identity(requests,0)
