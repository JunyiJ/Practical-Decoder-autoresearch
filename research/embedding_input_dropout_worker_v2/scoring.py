"""Same full-target scoring equations; explicit new model/config contract."""
import math
import torch
import torch.nn.functional as F
from research.body_allocation_model_v1.contract import require
from research.embedding_input_dropout_v1.model import GPT
from research.final_ln_scale_worker_v1.validation import target_contract

def score(model,data):
    require(type(model) is GPT,'Explicit input-dropout model required')
    model.validate()
    require(not model.training and next(model.parameters()).dtype==torch.float32
        and model.cfg['base_model']['block_size']==128,'Own-policy eval FP32 context128 required')
    target=target_contract(data);ids=data.val_data;device=next(model.parameters()).device
    losses=[]
    with torch.no_grad():
        for start in range(0,len(ids)-1,128):
            count=min(128,len(ids)-1-start)
            x=ids[start:start+count].unsqueeze(0).to(device);y=ids[start+1:start+count+1].to(device)
            logits,_=model(x)
            require(logits.shape==(1,count,50257) and logits.dtype==torch.float32,'Exact GPT2 logit layout required')
            # Preserve FP32 bytes during MPS transfer; cast only after CPU arrival.
            values=F.cross_entropy(logits[0],y,reduction='none').to('cpu').to(dtype=torch.float64)
            require(torch.isfinite(values).all().item() and (values>=0).all().item(),'Finite nonnegative target losses required')
            losses.append(values)
    loss=torch.cat(losses);require(loss.numel()==target['targets'],'Target coverage differs')
    blocks=[float(loss[i:i+2048].sum()) for i in range(0,len(loss),2048)]
    windows=[float(v.sum()) for v in losses];total=float(loss.sum())
    return dict(target=target,window_loss_sums=windows,block_loss_sums=blocks,total_loss=total,ce=total/target['targets'],
                bits_per_byte=total/math.log(2)/target['target_bytes'],loss_dtype='float32_cross_entropy_float64_aggregation')

