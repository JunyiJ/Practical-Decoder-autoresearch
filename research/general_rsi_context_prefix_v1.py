"""Common-target within-document context truncation, within the frozen model limit."""
import math
from research.general_rsi_pseudo_update_diagnostic_v1 import batch_hash,check_stop
CONTEXTS=(16,48,96)

def case(values,document,target_start,context):
    assert context in CONTEXTS
    begin=document['offset'];end=begin+document['tokens'];t=target_start
    assert begin+96<=t and t+32<=end
    x=values[t-context:t+31][None,:];y=values[t:t+32]
    assert x.shape==(1,context+31) and y.shape==(32,)
    return x,y,slice(context-1,context+31)

def inventory(data,seed):
    import torch
    rng=torch.Generator().manual_seed(seed);values=data.view.train_data;windows=[]
    for source in ('fineweb_edu','wikipedia'):
        eligible=[i for i,d in enumerate(data.indexes['train']) if d['source']==source and d['tokens']>=128]
        assert len(eligible)>=64
        for i in torch.randperm(len(eligible),generator=rng)[:64].tolist():
            doc=eligible[i];d=data.indexes['train'][doc];start=d['offset']+int(torch.randint(d['tokens']-127,(),generator=rng));t=start+96
            xs={};targets=None
            for c in CONTEXTS:
                x,y,s=case(values,d,t,c);xs[str(c)]=batch_hash(x)
                if targets is None:targets=y.tolist()
                assert targets==y.tolist()
            windows.append(dict(window=len(windows),document=doc,source=source,span_start=start,target_start=t,target_ids=targets,input_sha256=xs))
    assert len(windows)==128 and len({w['document'] for w in windows})==128
    return windows

def score(model,values,document,target_start,context):
    import torch
    from torch.nn import functional as F
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    x,y,selection=case(values,document,target_start,context)
    prior=model.training;versions=tuple(p._version for _,p in model.ordered_parameters())
    try:
        model.eval()
        with torch.no_grad(),FP32RMSNormInference():
            check_stop();z=model.primary_logits(x.to('mps')).values[0,selection,:]
            assert z.shape==(32,50257)
            losses=F.cross_entropy(z,y.to('mps'),reduction='none');assert torch.isfinite(losses).all()
            nums=losses.cpu().double().tolist()
    finally:model.train(prior)
    assert versions==tuple(p._version for _,p in model.ordered_parameters())
    return dict(context=context,targets=32,target_ids=y.tolist(),input_sha256=batch_hash(x),input_tokens=x.numel(),token_losses=nums,loss_sum=math.fsum(nums),ce=math.fsum(nums)/32)
