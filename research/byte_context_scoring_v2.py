"""FP32 scoring through the training-equivalent RMSNorm path on PyTorch2.14.1.

The installed MPS no-grad fused RMSNorm defaults to double epsilon when eps=None;
its autograd/composite path honors the declared float epsilon. Enabling graph
construction during forward avoids that demonstrated backend defect. No backward
is called, losses are detached immediately, and model weights/gradients are not
changed. Frozen model and optimizer sources remain untouched.
"""
import math,time
import torch
from torch.nn import functional as F


def score(model,view,device):
    assert not torch.is_inference_mode_enabled(), 'Forward graph workaround requires ordinary mode'
    assert all(p.grad is None for _,p in model.ordered_parameters())
    was_training=model.training;model.eval();losses=[]
    sampler=view.generators['train'].get_state().clone();started=time.perf_counter()
    try:
        # Do not replace with no_grad: that selects the erroneous MPS epsilon.
        with torch.enable_grad():
            for start in range(0,34278,128):
                count=min(128,34278-start)
                ids=view.val_data[start:start+count].unsqueeze(0).to(device)
                targets=view.val_data[start+1:start+count+1].to(device)
                logits,_=model(ids)
                values=F.cross_entropy(logits[0],targets,reduction='none').detach().cpu().double()
                assert bool(torch.isfinite(values).all()) and bool((values>=0).all())
                losses.extend(values.tolist())
                del logits,values,ids,targets
    finally:
        model.train(was_training)
    assert torch.equal(sampler,view.generators['train'].get_state()) and len(losses)==34278
    assert all(p.grad is None for _,p in model.ordered_parameters())
    return dict(ce=math.fsum(losses)/34278,per_target_losses=losses,target_count=34278,
                seconds=time.perf_counter()-started,holdout_scored=False,device=device,
                scoring_policy='byte_context_fp32_grad_enabled_no_backward_v2')
