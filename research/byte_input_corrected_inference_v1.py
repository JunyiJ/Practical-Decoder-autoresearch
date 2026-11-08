"""Corrected read-only public inference for the adopted model-v6 checkpoint.

Callers admit these sources and own the research lock/resource guard. This API
does not train, choose checkpoints, or change the original checkpoint format.
Ordinary no-grad is owned here; torch.inference_mode and autocast are unsupported.
"""
import torch
from torch.nn import functional as F
from research import byte_input_checkpoint_consumer_v4 as original
from research import byte_input_inference_v5 as inference
from research.byte_input_cache_v4 import Session as OriginalSession
from research.byte_input_model_v6 import ByteInputModel
from research.byte_input_model_v3.contract import require, same_typed
from research.byte_context_cpu_reference_v1 import cpu_cross_entropy
from research.fp32_rmsnorm_inference_v1 import FP32RMSNormInference


def execution_mode(device):
    require(not torch.is_inference_mode_enabled()
            and not torch.is_autocast_enabled(device.type),
            'Ordinary FP32 execution required; no inference_mode or autocast')


class Session(OriginalSession):
    """Original cache ownership and position checks with corrected normalization."""
    def logits(self, inputs, start_pos):
        execution_mode(self.device)
        with FP32RMSNormInference():
            return super().logits(inputs, start_pos)


class Consumer:
    def __init__(self, model, identity, artifact):
        require(type(model) is ByteInputModel and model.spec.variant == 'raw_bytes'
                and not model.training, 'Adopted raw-byte eval model required')
        inference.identity(identity)
        inference.receipt(artifact, inference.KIND, inference.FILE_CAP)
        require(same_typed(model.identity, identity['model']), 'Exact loaded model identity')
        self.model = model
        self.identity = identity
        self.artifact = artifact
        self.device = model.validate()
        self._state = self._versions()

    def _versions(self):
        return tuple((name, id(t), t.untyped_storage().data_ptr(), t._version,
                      t.device, t.dtype, tuple(t.shape))
                     for name, t in (*self.model.ordered_parameters(), *self.model.named_buffers()))

    def check(self):
        require(not self.model.training and all(not m.training for m in self.model.modules())
                and self._versions() == self._state, 'Unchanged eval model for consumer lifetime')
        self.model.validate()

    def logits(self, inputs):
        """Return FP32 [batch,time,vocabulary] logits without a gradient graph."""
        self.check(); execution_mode(self.device)
        with torch.no_grad(), FP32RMSNormInference():
            result = inference._logits(self.model, inputs)
        self.check()
        return result

    def token_losses(self, inputs, targets):
        """Unreduced FP32 next-token CE, shaped like the supplied targets.

        CPU uses the separately verified shifted FP32 reduction. MPS retains
        native cross-entropy. This returns no aggregate or quality decision.
        """
        require(type(inputs) is torch.Tensor and type(targets) is torch.Tensor
                and targets.dtype == torch.int64 and targets.device == self.device
                and tuple(targets.shape) == tuple(inputs.shape) and targets.numel() > 0,
                'Matching ordinary int64 target tensor on the model device')
        require(bool((targets >= 0).all()) and bool((targets < 50257).all()),
                'Targets must be ordinary vocabulary IDs')
        logits = self.logits(inputs).reshape(-1, 50257)
        labels = targets.reshape(-1)
        with torch.no_grad():
            values = (cpu_cross_entropy(logits, labels) if self.device.type == 'cpu'
                      else F.cross_entropy(logits, labels, reduction='none'))
        require(values.dtype == torch.float32 and not values.requires_grad
                and bool(torch.isfinite(values).all()) and bool((values >= 0).all()),
                'Finite FP32 token losses without gradients')
        return values.reshape(targets.shape)

    def session(self, batch):
        self.check(); execution_mode(self.device)
        return Session(self.model, batch)


def load(implementation_record, plan_record, acceptance_record, backend):
    """Load the original selected weights through the existing bound loader."""
    model, identity, artifact = original.load(
        implementation_record, plan_record, acceptance_record, backend)
    return Consumer(model, identity, artifact)
