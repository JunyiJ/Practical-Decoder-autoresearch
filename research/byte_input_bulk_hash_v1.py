"""Isolated complete-buffer hashing prototype; not adopted.

For the fixed validated model buffers (NumPy-compatible bool/FP32/int64),
transfer each logical contiguous buffer once and hash all bytes without an
intermediate Python bytes copy. The original expected SHA256 values and every
model check remain. Largest current buffer is52,267,280 bytes. Exact-type
training/checkpoint consumers intentionally reject this isolated subclass.
"""
import hashlib
from research.byte_input_model_v5.model import ByteInputModel


class BulkHashModel(ByteInputModel):
    def _buffer_digest(self, tensor):
        result = hashlib.sha256()
        flat = tensor.detach().reshape(-1).to(device='cpu').contiguous()
        result.update(memoryview(flat.numpy()).cast('B'))
        return result.hexdigest()
