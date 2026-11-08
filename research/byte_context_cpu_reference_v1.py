"""Stable FP32 CPU CE reference; primary MPS scorer remains exactly v2."""
import math
import time
import torch
from research.byte_context_scoring_v2 import score as primary_score


def cpu_cross_entropy(logits, targets):
    """Same unweighted CE, with an explicit shifted FP32 reduction.

    The installed CPU fused log-softmax reduction exceeded the fixed numerical
    tolerance on identical saved logits. This function changes only the CPU
    verification reference; it is never used by training or primary MPS scoring.
    """
    assert logits.device.type=='cpu' and logits.dtype==torch.float32
    assert targets.device.type=='cpu' and targets.dtype==torch.int64
    shifted=logits-logits.amax(dim=-1,keepdim=True)
    normalizer=shifted.exp().sum(dim=-1).log()
    return normalizer-shifted.gather(-1,targets.unsqueeze(-1)).squeeze(-1)


def score(model,view,device):
    if device=='mps':
        return primary_score(model,view,device)
    assert device=='cpu' and not torch.is_inference_mode_enabled()
    assert all(p.grad is None for _,p in model.ordered_parameters())
    was_training=model.training;model.eval();losses=[]
    sampler=view.generators['train'].get_state().clone();started=time.perf_counter()
    try:
        with torch.enable_grad():
            for start in range(0,34278,128):
                count=min(128,34278-start)
                ids=view.val_data[start:start+count].unsqueeze(0).to(device)
                targets=view.val_data[start+1:start+count+1].to(device)
                logits,_=model(ids)
                values=cpu_cross_entropy(logits[0].detach(),targets)
                assert torch.isfinite(values).all() and (values>=0).all()
                losses.extend(values.double().tolist())
                del logits,values,ids,targets
    finally:
        model.train(was_training)
    assert torch.equal(sampler,view.generators['train'].get_state()) and len(losses)==34278
    assert all(p.grad is None for _,p in model.ordered_parameters())
    return dict(ce=math.fsum(losses)/34278,per_target_losses=losses,target_count=34278,
        seconds=time.perf_counter()-started,holdout_scored=False,device=device,
        scoring_policy='byte_context_cpu_stable_fp32_reference_v1')
