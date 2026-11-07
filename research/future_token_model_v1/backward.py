"""Experiment-specific sequential-head backward; no clipping or optimizer step."""
import torch
from torch.nn import functional as F
from research.future_token_objective_v1 import align, objective
from .contract import require


def preflight(model, sequence, *, require_empty_gradients):
    require(type(require_empty_gradients) is bool, 'Explicit gradient-slot policy required')
    model.validate()
    require(torch.is_grad_enabled() and model.training and all(m.training for m in model.modules()),
            'Own gradient-enabled training mode required')
    x, y1, y2 = align(sequence, 50257)
    model.tokens(x)
    if require_empty_gradients:
        require(all(p.grad is None for p in model.parameters()), 'Clear all grad slots before sequential backward')
    require(all(bool(torch.isfinite(p).all()) for p in model.parameters()), 'Finite learned parameters required')
    return x, y1, y2


def trunk(model, x):
    h = model.native.transformer.wte(x)
    for block in model.native.transformer.h[:3]:
        h = block(h, None, 0)
    return h


def head(model, h, *, auxiliary):
    require(type(auxiliary) is bool, 'Exact head selector required')
    if auxiliary:
        require(model.signature['arm'] != 'native_next', 'Native has no auxiliary head')
        block, norm = model.auxiliary.block, model.auxiliary.norm
    else:
        block, norm = model.native.transformer.h[3], model.native.transformer.ln_f
    return model.native.lm_head(norm(block(h, None, 0)))/4.


def joint_reference(model, sequence):
    """Small generated proof reference only; never the real full-batch worker."""
    x, _, _ = preflight(model, sequence, require_empty_gradients=False)
    require(x.shape[0] <= 2 and x.shape[1] <= 16, 'Joint reference bounded to B2T16 generated proof')
    h = trunk(model, x)
    primary = head(model, h, auxiliary=False)
    auxiliary = None if model.signature['arm'] == 'native_next' else head(model, h, auxiliary=True)
    return objective(model.signature['arm'], primary, auxiliary, sequence)


def finite_ce(logits, target):
    loss = F.cross_entropy(logits.reshape(-1, 50257), target.reshape(-1), reduction='mean')
    require(loss.dtype == torch.float32 and bool(torch.isfinite(loss)) and bool(loss >= 0),
            'Finite genuine FP32 CE required')
    return loss


def sequential_backward(model, sequence):
    """Populate complete accumulated grads once; caller clips once and steps once.

    Structural/ID/nonfinite-parameter preflight precedes grad mutation. A numeric
    failure during backward may leave partial grad slots: the caller must stop
    without clipping/stepping/exporting; this is not a transactional optimizer.
    """
    x, y1, y2 = preflight(model, sequence, require_empty_gradients=True)
    arm = model.signature['arm']
    if arm == 'native_next':
        # Exact genuine native call and ordinary complete-model backward.
        _, primary = model.native(x, y1)
        require(bool(torch.isfinite(primary)) and bool(primary >= 0), 'Finite native primary CE required')
        value = float(primary.detach().cpu()); primary.backward(); del primary
        result = dict(primary_ce=value, auxiliary_ce=None, training_loss=value)
    else:
        h = trunk(model, x)
        leaf = h.detach().requires_grad_(True)
        logits = head(model, leaf, auxiliary=False)
        primary = finite_ce(logits, y1)
        primary_value = float(primary.detach().cpu())
        (primary*.5).backward()
        del primary, logits
        # The primary vocabulary-sized tensors/graph are released before head2.
        logits = head(model, leaf, auxiliary=True)
        auxiliary = finite_ce(logits, y1 if arm == 'duplicate_next' else y2)
        auxiliary_value = float(auxiliary.detach().cpu())
        (auxiliary*.5).backward()
        del auxiliary, logits
        require(leaf.grad is not None and bool(torch.isfinite(leaf.grad).all()), 'Finite summed trunk gradient required')
        h.backward(leaf.grad)
        del h, leaf
        result = dict(primary_ce=primary_value, auxiliary_ce=auxiliary_value,
                      training_loss=.5*(primary_value+auxiliary_value))
    for p in model.parameters():
        require(p.grad is not None and p.grad.shape == p.shape and p.grad.dtype == p.dtype
                and p.grad.device == p.device and not p.grad.is_sparse and bool(torch.isfinite(p.grad).all()),
                'Complete finite learned parameter gradients required')
    return result
