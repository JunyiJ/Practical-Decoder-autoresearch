"""New precision-only composer Muon integration; no frozen implementation edits."""
import torch
from torch.optim import _muon as installed
from research.byte_input_model_v6.optimizer import ByteInputOptimizer
from research.general_rsi_training_v2 import Context as OriginalContext, KIND as ORIGINAL_KIND

KIND = 'general_rsi_composer_fp32_quality_training_v1'
ARMS = ('existing_byte', 'byte_composer_fp32')


def orthogonalize(grad, ns_coefficients, ns_steps, eps):
    """Installed torch2.14.1 NS expression, preserving input precision.

    Only the baseline grad.bfloat16() cast is replaced by grad.clone().
    FP64 CPU callers are diagnostics; production inputs must be FP32.
    """
    assert grad.ndim == 2 and grad.dtype in (torch.float32, torch.float64)
    assert 0 <= ns_steps < 100 and len(ns_coefficients) == 3
    a, b, c = ns_coefficients
    ortho_grad = grad.clone()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c)
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad


class FP32ComposerOptimizer(ByteInputOptimizer):
    @torch.no_grad()
    def step(self):
        self.validate(require_gradients=True)
        names = {id(p): n for n, p in self._items}
        for _, parameter in self._items:
            parameter.mul_(1. - .00006)
        child = self._optimizers[0]
        for group in child.param_groups:
            params, grads, buffers = [], [], []
            has_complex = child._init_group(group, params, grads, buffers)
            assert not has_complex and len(params) == 1
            if names[id(params[0])].startswith('composer.'):
                p, grad, buf = params[0], grads[0], buffers[0]
                assert grad.dtype == torch.float32
                buf.lerp_(grad, 1 - group['momentum'])
                update = grad.lerp(buf, group['momentum']) if group['nesterov'] else buf
                update = orthogonalize(update, group['ns_coefficients'], group['ns_steps'], group['eps'])
                rate = installed._adjust_lr(group['lr'], group['adjust_lr_fn'], p.shape)
                p.mul_(1 - group['lr'] * group['weight_decay'])
                p.add_(update, alpha=-rate)
            else:
                installed.muon(params, grads, buffers, lr=group['lr'],
                    weight_decay=group['weight_decay'], momentum=group['momentum'],
                    nesterov=group['nesterov'], ns_coefficients=group['ns_coefficients'],
                    eps=group['eps'], ns_steps=group['ns_steps'],
                    adjust_lr_fn=group['adjust_lr_fn'], has_complex=has_complex)
        self._optimizers[1].step()
        self._steps += 1


class Context(OriginalContext):
    def __init__(self, arm, lr, horizon, roles, book, binding, device='mps'):
        assert arm in ARMS
        super().__init__('raw_bytes', lr, horizon, roles, book, binding, device)
        self.config['quality_arm'] = arm
        self.config['execution'] = KIND
        if arm == 'byte_composer_fp32':
            self.optimizer = FP32ComposerOptimizer(self.model, float(lr))
        self.base_lrs = tuple(g['lr'] for g in self.optimizer.param_groups)

    def capture(self):
        value = super().capture()
        value['kind'] = KIND
        return value

    def restore(self, payload):
        assert payload['kind'] == KIND and payload['config'] == self.config
        return super().restore(dict(payload, kind=ORIGINAL_KIND))
