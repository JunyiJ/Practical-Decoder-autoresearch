"""Bounded FP32 cached inference on the declared v2 pair; no training or launch."""
import copy
import gc
import math
from statistics import median
import time
import torch
from research.block_norm_v1 import campaign_contract as c
from . import validation as v
import json
from pathlib import Path
from omegaconf import OmegaConf
from research.local_mix_v1.runtime.attention.cache import build_caches
from research.local_mix_v1.runtime.models.block_norm import contract, validate_model
from research.local_mix_v1.validation_data import load as validation_data
from src.research.tokenizer_wide_hybrid_measurements import check_kernel, measure_kernel
from src.research.tokenizer_wide_hybrid_timing import token_digest, allocated_bytes, _measurement, _correctness

from .integration import identity as decay_identity
ARMS=('decay','zero')
BATCHES=(1,8)
PARAMETERS=dict(zip(ARMS,(28492416,28492416)))
CACHE_BYTES=dict(decay=196608,zero=196608)
CONTRACT=dict(kind='fixed_lr_decay_cached_pair_v1',batches=[1,8],warmups=2,repeats=8,
    context_tokens=128,prefill_tokens=32,decode_tokens=64,token_source='first96_training_ids',
    token_hash_encoding='uint32-little-endian',dtype='float32',device='mps',cpu_threads=4,
    rtol=3e-4,atol=3e-5,cache_bytes_per_batch_item=CACHE_BYTES,adoption_cost_route=False)
require=c.require


def order(pair_index,repeat):
    require(type(pair_index) is int and 0<=pair_index<3 and type(repeat) is int and 0<=repeat<8,'Finite pair/repeat required')
    return ARMS if (pair_index+repeat)%2==0 else ARMS[::-1]


def identity(requests,pair_index):
    order(pair_index,0);require(type(requests) is dict and set(requests)==set(ARMS),'Both v2 arms required')
    models={};ids=None
    for arm in ARMS:
        v.artifacts(requests[arm])
        cfg=OmegaConf.create(json.loads(Path(requests[arm]['config']['path']).read_text()))
        require(decay_identity(cfg)['decay']==dict(decay=.00006,zero=0.)[arm],'Requested decay arm differs')
        data=validation_data(cfg)
        require(set(data.splits)=={'train','val'},'Closed holdout required')
        row=data.train_data[:96].tolist();token_digest(row)
        require(ids is None or row==ids,'Paired training token identities differ');ids=row
        models[arm]=dict(files=copy.deepcopy(requests[arm]),
            config_sha256=c.fingerprint(OmegaConf.to_container(cfg,resolve=True)),parameters=PARAMETERS[arm],
            parameter_storage_bytes=4*PARAMETERS[arm])
    return dict(kind=CONTRACT['kind'],contract=copy.deepcopy(CONTRACT),pair_index=pair_index,
        requests_sha256=c.fingerprint(requests),environment=c.environment(),models=models,
        fixed_train_token_ids=ids,token_sha256=token_digest(ids),quality_adopted=False)


def sample(row,batch,arm):
    out=_measurement(row,batch)
    require(out['state_bytes']==CACHE_BYTES[arm]*batch,'Allocated cache bytes differ')
    require(out['observed_mps_current_bytes']<=out['observed_mps_driver_bytes']<6*1024**3,'MPS allocation bound differs')
    return out


def review_values(result,expected):
    require(type(result) is dict and set(result)==set(expected)|{'correctness','warmup_records','records'},'Exact inference report required')
    require(c.canonical({k:result[k] for k in expected})==c.canonical(expected),'Inference identity differs')
    pair_index=expected['pair_index'];checks=[(b,f) for b in BATCHES for f in order(pair_index,0)]
    require(type(result['correctness']) is list and len(result['correctness'])==4,'Complete cache correctness barrier required')
    for row,(batch,arm) in zip(result['correctness'],checks):
        require(set(row)=={'batch','arm','passed','max_absolute_error'} and c.canonical([row['batch'],row['arm']])==c.canonical([batch,arm]),'Correctness order differs')
        _correctness({k:row[k] for k in ('passed','max_absolute_error')})
    keys={'prefill_seconds','decode_seconds','state_bytes','observed_mps_current_bytes','observed_mps_driver_bytes'}
    for field,count in (('warmup_records',2),('records',8)):
        sequence=[dict(batch=b,arm=f,repeat=r,position=p) for b in BATCHES for r in range(count) for p,f in enumerate(order(pair_index,r))]
        require(type(result[field]) is list and len(result[field])==len(sequence),'Complete timing sequence required')
        for row,item in zip(result[field],sequence):
            require(set(row)==set(item)|keys|{'decode_tokens_per_second'},'Timing schema differs')
            require(c.canonical({k:row[k] for k in item})==c.canonical(item),'Timing order differs')
            checked=sample({k:row[k] for k in keys},item['batch'],item['arm'])
            rate=row['decode_tokens_per_second']
            require(type(rate) in (float,int) and math.isfinite(rate) and math.isclose(rate,checked['decode_tokens_per_second'],rel_tol=1e-12),'Throughput arithmetic differs')
    cells=[]
    for batch in BATCHES:
        for arm in ARMS:
            rows=[r for r in result['records'] if r['batch']==batch and r['arm']==arm]
            cells.append(dict(batch=batch,arm=arm,state_bytes=CACHE_BYTES[arm]*batch,
                median_prefill_seconds=median(r['prefill_seconds'] for r in rows),
                median_decode_seconds=median(r['decode_seconds'] for r in rows)))
    return dict(status='values_reviewed_transport_provenance_required',cells=cells,quality_adopted=False,
        scope='Two resident H200 models; cache construction/loading excluded from timings. Allocator observations are not single-model deployment peaks. Warmups excluded; cost descriptive only.')


def measure_pair(requests,pair_index=0):
    """Caller holds worker lock and strict600s/RSS/MPS/disk guard throughout."""
    expected=identity(requests,pair_index);models={};configs={}
    try:
        for arm in ARMS:
            model,data,cfg=v.load(expected['models'][arm]['files'],'mps')
            validate_model(model,contract(cfg.model));require(not model.training,'Evaluation mode required')
            require(all(p.dtype==torch.float32 and p.device.type=='mps' for p in model.parameters()),'FP32 MPS parameters required')
            require(sum(p.numel() for p in model.parameters())==PARAMETERS[arm]
                    and allocated_bytes(model.parameters())==4*PARAMETERS[arm],'Actual parameter storage differs')
            require(data.train_data[:96].tolist()==expected['fixed_train_token_ids'],'Loaded training IDs differ')
            models[arm]=model;configs[arm]=cfg
        fixed=torch.tensor(expected['fixed_train_token_ids'],dtype=torch.int64,device='mps')
        inputs={b:fixed.unsqueeze(0).repeat(b,1) for b in BATCHES}
        def arguments(arm,batch):
            require(not models[arm].training,'Model left evaluation mode')
            return models[arm],inputs[batch],lambda:build_caches(configs[arm],'mps',batch_size=batch,dtype=torch.float32)
        checks=[]
        # All four full/cache/reset comparisons precede even the first warmup.
        for batch in BATCHES:
            for arm in order(pair_index,0):
                checks.append(dict(batch=batch,arm=arm,**_correctness(check_kernel(*arguments(arm,batch)))))
        result=dict(expected,correctness=checks,warmup_records=[],records=[])
        for field,count in (('warmup_records',2),('records',8)):
            for batch in BATCHES:
                for repeat in range(count):
                    for position,arm in enumerate(order(pair_index,repeat)):
                        measured=measure_kernel(*arguments(arm,batch),synchronize=torch.mps.synchronize,
                            clock=time.perf_counter,memory=lambda:(torch.mps.current_allocated_memory(),torch.mps.driver_allocated_memory()))
                        result[field].append(dict(batch=batch,arm=arm,repeat=repeat,position=position,**sample(measured,batch,arm)))
        require(c.canonical(identity(requests,pair_index))==c.canonical(expected),'Pair provenance changed during timing')
        review_values(result,expected)
        return result
    finally:
        models.clear();gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def review(result,requests,pair_index):
    return review_values(result,identity(requests,pair_index))
