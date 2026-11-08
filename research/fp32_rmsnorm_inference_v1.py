"""Unqualified scoped inference repair for the demonstrated MPS default-eps bug.

This is not used by the active quality panel or the adopted public consumer.
The caller must explicitly enter the mode around a no-grad inference operation.
It supplies the already-declared FP32 epsilon without changing module attributes,
parameters, checkpoint identities, or global PyTorch functions. Explicit epsilon
arguments pass through unchanged. Training is deliberately unsupported.
"""
import torch
from torch.nn import functional as F
from torch.overrides import TorchFunctionMode


class FP32RMSNormInference(TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.corrected_calls = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if func is F.rms_norm or func is torch.rms_norm:
            return self._rms_norm(func, *args, **kwargs)
        return func(*args, **kwargs)

    def _rms_norm(self, func, input, normalized_shape, weight=None, eps=None):
        if eps is not None:
            return func(input, normalized_shape, weight, eps)
        if torch.is_grad_enabled() or torch.is_inference_mode_enabled():
            raise ValueError('The RMSNorm repair requires ordinary no-grad mode')
        if (type(input) is not torch.Tensor or input.dtype != torch.float32
                or input.device.type not in ('cpu', 'mps')
                or tuple(normalized_shape) != (256,)
                or torch.is_autocast_enabled(input.device.type)):
            raise ValueError('The repair requires ordinary CPU/MPS FP32 width256 inputs')
        if (type(weight) is not torch.nn.Parameter or weight.dtype != torch.float32
                or weight.device != input.device or tuple(weight.shape) != (256,)):
            raise ValueError('The repair requires the original FP32 affine RMSNorm weight')
        self.corrected_calls += 1
        return func(input, normalized_shape, weight, 2.0**-23)
