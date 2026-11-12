"""Same within-document target coverage at context128/256; no test reader."""
import math
from research.general_rsi_pseudo_update_diagnostic_v1 import check_stop,batch_hash

def chunks(values,documents,context):
    assert type(context) is int and context in (128,256)
    for doc,row in enumerate(documents):
        start,size=row['offset'],row['tokens'];assert size>=2 and start>=0 and start+size<=len(values)
        for offset in range(0,size-1,context):
            count=min(context,size-1-offset)
            yield doc,row['source'],start+offset+1,values[start+offset:start+offset+count][None,:],values[start+offset+1:start+offset+count+1][None,:]

def evaluate(model,values,documents,context):
    import torch
    from torch.nn import functional as F
    from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference
    assert context<=model.spec.context
    rows=[];prior=model.training
    try:
        model.eval()
        with torch.no_grad(),FP32RMSNormInference():
            for doc,source,target_start,x,y in chunks(values,documents,context):
                check_stop();logits=model.primary_logits(x.to('mps')).values
                losses=F.cross_entropy(logits.reshape(-1,50257),y.to('mps').reshape(-1),reduction='none').cpu().double().tolist()
                assert all(math.isfinite(n) for n in losses)
                rows.append(dict(document=doc,source=source,target_start=target_start,targets=y.numel(),target_ids=y.flatten().tolist(),input_sha256=batch_hash(x),loss_sum=math.fsum(losses),token_losses=losses))
    finally:model.train(prior)
    assert sum(r['targets'] for r in rows)==sum(d['tokens']-1 for d in documents)
    return dict(context=context,rows=rows,targets=sum(r['targets'] for r in rows),loss_sum=math.fsum(r['loss_sum'] for r in rows))
