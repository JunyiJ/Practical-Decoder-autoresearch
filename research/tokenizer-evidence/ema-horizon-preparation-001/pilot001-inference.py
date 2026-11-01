"""Descriptive dual-EMA pilot timing; unchanged numerical/timing primitives."""
import gc,json,time
from statistics import median
import torch
from research.ema_horizon_v1 import contract as c,artifacts as a,execution as e
from research.block_norm_v1.fixture_dispatch import read,reference
from research.block_norm_v1.inference import sample
from research.block_norm_v1.runtime.attention.cache import build_caches
from src.research.tokenizer_wide_hybrid_measurements import check_kernel,measure_kernel
from src.research.tokenizer_wide_hybrid_timing import token_digest
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock

def measure(request,pair_index=0):
    models={};configs={};ids=None;checks=[];records=[];warmups=[]
    try:
        for half in (100,200):
            model,data,cfg=a.load_endpoint(request,half)
            row=data.train_data[:96].tolist()
            assert ids is None or row==ids
            ids=row;models[half]=model;configs[half]=cfg
            assert not model.training and sum(p.numel() for p in model.parameters())==28492416
            assert all(p.dtype==torch.float32 and p.device.type=='mps' for p in model.parameters())
        inputs={b:torch.tensor(ids,device='mps').unsqueeze(0).repeat(b,1) for b in (1,8)}
        def args(half,b):
            return models[half],inputs[b],lambda:build_caches(configs[half],'mps',batch_size=b,dtype=torch.float32)
        for b in (1,8):
            for half in (200,100):checks.append(dict(batch=b,half_life=half,**check_kernel(*args(half,b))))
        for output,count in ((warmups,2),(records,8)):
            for b in (1,8):
                for repeat in range(count):
                    order=(200,100) if (pair_index+repeat)%2==0 else (100,200)
                    for position,half in enumerate(order):
                        value=measure_kernel(*args(half,b),synchronize=torch.mps.synchronize,clock=time.perf_counter,
                            memory=lambda:(torch.mps.current_allocated_memory(),torch.mps.driver_allocated_memory()))
                        output.append(dict(batch=b,half_life=half,repeat=repeat,position=position,**sample(value,b)))
        a.verify_request(request)
        assert len(checks)==4 and all(x['passed'] for x in checks) and len(warmups)==8 and len(records)==32
        cells=[]
        for b in (1,8):
            for half in (200,100):
                rows=[x for x in records if x['batch']==b and x['half_life']==half]
                cells.append(dict(batch=b,half_life=half,state_bytes=196608*b,
                    prefill_median_seconds=median(x['prefill_seconds'] for x in rows),
                    decode_median_seconds=median(x['decode_seconds'] for x in rows)))
        return dict(status='passed',request_sha256=c.inherited.fingerprint(request),pair_index=pair_index,
            tokens_sha256=token_digest(ids),correctness=checks,warmups=warmups,records=records,cells=cells,
            scope='Two resident EMA models; first96 training IDs,32-token prefill/64 teacher-forced decode,2 warmups/8 alternating repeats,batches1/8. Loading/cache construction excluded. Descriptive pilot only; no cost-only adoption.',
            holdout_scored=False,quality_adopted=False)
    finally:
        models.clear();gc.collect();torch.mps.synchronize();torch.mps.empty_cache()

if __name__=='__main__':
    torch.set_num_threads(4)
    with session_lock(GLOBAL_LOCK.parent),session_lock(e.ROOT):
        e.idle()
        anchor=read(reference(e.anchor_path('pilots',0)));accepted=read(anchor['acceptance'])
        assert accepted['artifact_acceptance']['accepted']
        request=read(accepted['request'])['acceptance_request']
        plan=dict(acceptance=reference(e.anchor_path('pilots',0)),source=reference(__file__),sources=c.source_inventory(),
            request_sha256=c.inherited.fingerprint(request),batches=[1,8],half_lives=[200,100],warmups=2,repeats=8,
            prefill_tokens=32,decode_tokens=64,token_source='first96_training_ids',quality_adopted=False)
        e.exclusive(e.EVIDENCE/'pilot001-inference-plan.json',plan)
        with e.Guard(e.EVIDENCE/'pilot001-inference-guard.json',600):
            result=measure(request)
            assert c.source_inventory()==plan['sources'] and reference(__file__)==plan['source']
            e.exclusive(e.EVIDENCE/'pilot001-inference-result.json',dict(result,plan=reference(e.EVIDENCE/'pilot001-inference-plan.json')))
        print(json.dumps(result['cells']))
