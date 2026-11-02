"""Measure the existing PyTorch operation without changing its gradient arithmetic."""
import math
import torch


@torch.no_grad()
def clip_gradients(parameters, threshold):
    if type(threshold) is not float or threshold not in (1.0, 0.5):
        raise ValueError('Explicit global clipping threshold1.0 or0.5 required')
    parameters = list(parameters)
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        raise ValueError('Measured full-model update requires gradients')
    # Keep the incumbent default foreach selection and error-before-update behavior.
    pre = torch.nn.utils.clip_grad_norm_(parameters, threshold, norm_type=2.0,
                                        error_if_nonfinite=True)
    coefficient = torch.clamp(threshold / (pre + 1e-6), max=1.0)
    post = torch.linalg.vector_norm(torch.stack([
        torch.linalg.vector_norm(g, 2).to(pre.device) for g in gradients
    ]), 2)
    result = dict(threshold=threshold, pre_norm=float(pre), post_norm=float(post),
                  coefficient=float(coefficient), gradient_tensors=len(gradients))
    if not all(math.isfinite(result[k]) for k in ('pre_norm','post_norm','coefficient')):
        raise ValueError('Nonfinite clipping observation')
    if result['post_norm'] > threshold + max(1e-5, threshold * 1e-5):
        raise ValueError('Post-clip norm exceeds threshold tolerance')
    return result
