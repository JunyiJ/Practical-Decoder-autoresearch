"""Unqualified model candidate for the next focused serialization comparison.

Only immutable-buffer serialization differs from model-v3. Construction,
forward/backward math and every validation call are inherited unchanged.
This is not wired into the frozen quality worker. Existing optimizer/state
consumers require the exact model-v3 class and intentionally reject this class.
An execution must bind this source as well as its inherited dependencies.
"""
import hashlib

import torch

from research.byte_input_model_v3.model import ByteInputModel


class DirectByteHashModel(ByteInputModel):
    """Retain full content hashing while avoiding a Python object per byte."""

    def _buffer_digest(self, tensor):
        result = hashlib.sha256()
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), 131072):
            part = flat[start:start+131072].to(device='cpu').contiguous()
            result.update(part.view(torch.uint8).numpy().tobytes())
        return result.hexdigest()
