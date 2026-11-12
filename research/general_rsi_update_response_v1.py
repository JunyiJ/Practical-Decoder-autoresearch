"""Fixed realized-update interpolation and explicit-target training-only scoring."""
import math
from research.general_rsi_pseudo_update_diagnostic_v1 import check_stop
SCALES=(0.,1/16,1/4,1.)

def parameters(model):
    return {n:p.detach().cpu().clone() for n,p in model.ordered_parameters()}

def set_point(model,before,after,scale):
    import torch
    assert scale in SCALES
    items=dict(model.ordered_parameters());assert list(items)==list(before)==list(after)
    checks=0
    with torch.no_grad():
        for n,p in items.items():
            a,b=before[n],after[n]
            assert a.device.type==b.device.type=='cpu' and a.dtype==b.dtype==torch.float32 and a.shape==b.shape==p.shape
            # Explicit unfused CPU FP32 subtraction, multiplication, addition.
            expected=a if scale==0. else b if scale==1. else a+(b-a)*scale
            assert bool(torch.isfinite(expected).all())
            p.copy_(expected)
            assert torch.equal(p.detach().cpu(),expected)
            checks+=1
    return checks

def summarize(sums,counts):
    assert len(sums)==len(counts)>0 and all(isinstance(n,int) and n>0 for n in counts)
    assert all(math.isfinite(x) for x in sums)
    return dict(ce=math.fsum(sums)/sum(counts),window_loss_sums=sums,window_targets=counts,targets=sum(counts))

def score(model,batches):
    import torch
    from torch.nn import functional as F
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    prior=model.training;versions=tuple(p._version for _,p in model.ordered_parameters());sums=[];counts=[]
    try:
        model.eval()
        with torch.no_grad(),FP32RMSNormInference():
            for x,y in batches:
                check_stop();assert x.ndim==y.ndim==2 and x.shape==y.shape and x.shape[0]==1
                logits=model.primary_logits(x.to('mps')).values[0]
                losses=F.cross_entropy(logits,y[0].to('mps'),reduction='none')
                assert losses.numel()==y.numel() and bool(torch.isfinite(losses).all())
                sums.append(math.fsum(losses.cpu().double().tolist()));counts.append(y.numel())
    finally:model.train(prior)
    assert versions==tuple(p._version for _,p in model.ordered_parameters())
    return summarize(sums,counts)

def inputs(config,data):
    import torch
    from research.general_rsi_pseudo_update_diagnostic_v1 import ce_batch,batch_hash
    values=data.view.train_data;cases=[];excluded=[];updates=[]
    for c in config['cases']:
        rng=torch.Generator().manual_seed(c['roles']['train_sampler'])
        starts=torch.randint(len(values)-512,(2,),generator=rng).tolist();x,y=ce_batch(values,starts)
        cases.append(dict(case=c['case'],starts=starts,sha256=batch_hash(x,y)));updates.append((x,y));excluded.extend((s,s+513) for s in starts)
    rng=torch.Generator().manual_seed(config['evaluation_seed']);eligible=[i for i,r in enumerate(data.indexes['train']) if r['tokens']>=129];windows=[];proposals=0
    while len(windows)<64:
        proposals+=1;assert proposals<10000
        doc=eligible[int(torch.randint(len(eligible),(),generator=rng))];d=data.indexes['train'][doc];start=d['offset']+int(torch.randint(d['tokens']-128,(),generator=rng))
        if any(start<b and a<start+129 for a,b in excluded):continue
        windows.append(dict(document=doc,source=d['source'],start=start,targets=128));excluded.append((start,start+129))
    batches=[(values[w['start']:w['start']+128][None,:],values[w['start']+1:w['start']+129][None,:]) for w in windows]
    return dict(cases=cases,evaluation_windows=windows,evaluation_sha256=batch_hash(*(t for b in batches for t in b)),targets=8192,probes_disjoint_from_all_update_source_positions=True),updates,batches
