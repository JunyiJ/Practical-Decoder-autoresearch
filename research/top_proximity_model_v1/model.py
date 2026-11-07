"""One unchanged native child and optional independent readout; SOURCE ONLY.

No checkpoint/cache/export/scorer/worker adapter or model qualification.
All training heads share one hidden graph; caller performs joint backward.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from research.final_ln_scale_model_v1.runtime.models.gpt import GPT as Native
from research.final_ln_scale_model_v1.runtime.models.tokenizer_architecture import validate_architecture_model
from research.top_proximity_targets_v1 import rows, probabilities
from research.top_proximity_objective_v1 import sparse_proximity_ce
from . import contract as C


@dataclass(frozen=True)
class PrimaryLogits:
    identity: tuple
    kind: str
    values: torch.Tensor


@dataclass(frozen=True)
class TrainingResult:
    identity: tuple
    kind: str
    primary: PrimaryLogits
    auxiliary_logits: object
    primary_loss: torch.Tensor
    auxiliary_loss: object
    total_loss: torch.Tensor


class TopModel(nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self._identity = C.signature(spec)
        C.require(torch.get_default_dtype() == torch.float32 and torch.get_default_device().type == 'cpu',
                  'Native-first CPU FP32 construction required')
        C.require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'),
                  'Autocast forbidden')
        # Preserve every genuine native initialization draw BEFORE copying a head.
        self.native = Native(C.native_config())
        if spec.variant == 'native':
            self.register_parameter('aux_weight', None)
        else:
            self.aux_weight = nn.Parameter(self.native.lm_head.weight.detach().clone())
        self._native_inventory = tuple((name, tuple(p.shape)) for name, p in self.native.named_parameters())
        self._buffer_inventory = tuple((name, tuple(p.shape), p.dtype) for name, p in self.native.named_buffers())
        self.validate()

    @property
    def identity(self):
        return self._identity

    def validate(self):
        C.require(C.signature(self.spec) == self._identity, 'Mutated TOP spec')
        C.require(type(self.native) is Native, 'Exact genuine native child required')
        cfg = OmegaConf.to_container(self.native.cfg, resolve=True)
        C.require(C.same_typed(cfg, C.native_model_dict()), 'True native child cfg differs')
        C.require(set(self._modules) == {'native'} and set(self._parameters) == {'aux_weight'}
                  and not self._buffers, 'Unexpected own module/parameter/buffer')
        validate_architecture_model(self.native)
        inventory = tuple((name, tuple(p.shape)) for name, p in self.native.named_parameters())
        C.require(inventory == self._native_inventory and len(inventory) == 88,
                  'Native parameter inventory differs')
        C.require(tuple((name, tuple(p.shape), p.dtype) for name, p in self.native.named_buffers())
                  == self._buffer_inventory, 'Native buffer inventory differs')
        C.require(self.native.width_mult == 4. and self.native.position_encoding == 'rope'
                  and self.native.output_bias is None and self.native.lm_head.bias is None
                  and self.native.input_embedding_scale == 1. and self.native.logit_z_signature is None,
                  'Fixed native projection/input/objective semantics differ')
        value = self.native.value_residual_signature
        C.require(value is None or value[0] == 'local', 'No cross-layer value reuse')
        C.require(len(self.native.transformer.h) == 4 and self.native.training == self.training,
                  'Native block count or training mode differs')
        if self.spec.variant == 'native':
            C.require(self.aux_weight is None, 'Native cannot own an auxiliary readout')
        else:
            C.require(type(self.aux_weight) is nn.Parameter and tuple(self.aux_weight.shape) == (50257, 256),
                      'Exact independent auxiliary readout required')
        items = self.ordered_parameters()
        expected = 28492416 if self.spec.variant == 'native' else 41358208
        C.require(len(items) == (88 if self.spec.variant == 'native' else 89)
                  and sum(p.numel() for _, p in items) == expected, 'Own parameter count differs')
        devices = {p.device for _, p in items}
        C.require(len(devices) == 1 and next(iter(devices)).type in ('cpu', 'mps'), 'CPU/MPS device required')
        pointers = []
        for _, p in items:
            C.require(type(p) is nn.Parameter and p.dtype == torch.float32 and p.layout == torch.strided
                      and p.requires_grad and bool(torch.isfinite(p).all()), 'Finite trainable FP32 parameters required')
            pointers.append(p.untyped_storage().data_ptr())
        C.require(len(set(pointers)) == len(pointers), 'Aliased parameter storage forbidden')
        C.require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'), 'Autocast forbidden')
        return next(iter(devices))

    def ordered_parameters(self):
        # Canonical unprefixed native names preserve width grouping despite child ownership.
        items = tuple(self.native.named_parameters())
        return items if self.aux_weight is None else items + (('aux_weight', self.aux_weight),)

    def _ids(self, ids, *, length, device):
        C.require(type(ids) is torch.Tensor and ids.layout == torch.strided and ids.dtype == torch.int64
                  and ids.device == device and ids.ndim == 2, 'Plain same-device rank2 int64 IDs required')
        C.require(1 <= ids.shape[0] <= 8 and ids.shape[1] == length, 'Exact bounded ID shape required')
        C.require(bool((ids >= 0).all()) and bool((ids < 50257).all()), 'No padding/OOV IDs')

    def _hidden(self, inputs):
        # This is the fixed native training path, not a second native.forward call.
        hidden = self.native.transformer.wte(inputs)
        for block in self.native.transformer.h:
            hidden, aux = block(hidden)
            C.require(aux is None, 'No unrelated block auxiliary objective')
        return self.native.transformer.ln_f(hidden)

    def forward_train(self, raw_ids, context):
        device = self.validate()
        C.require(type(context) is int and 1 <= context <= 128, 'Explicit context1..128 required')
        self._ids(raw_ids, length=context + self.spec.window, device=device)
        targets = None
        if self.spec.variant == 'future_proximity':
            # Complete ID/spec preflight above precedes target construction and all model graph work.
            sequences = raw_ids.detach().to('cpu').tolist()
            targets = tuple(probabilities(row) for sequence in sequences
                            for row in rows(sequence, context=context, window=self.spec.window, vocabulary=50257))
        inputs = raw_ids[:, :context].contiguous()
        labels = raw_ids[:, 1:context + 1].contiguous()
        auxiliary_logits, auxiliary_loss = None, None
        if self.spec.variant == 'native':
            primary_logits, primary_loss = self.native(inputs, labels)
            total_loss = primary_loss
        else:
            hidden = self._hidden(inputs)
            primary_logits = self.native.lm_head(hidden) / self.native.width_mult
            auxiliary_logits = F.linear(hidden, self.aux_weight) / self.native.width_mult
            primary_loss = F.cross_entropy(primary_logits.reshape(-1, 50257), labels.reshape(-1), reduction='mean')
            if self.spec.variant == 'copied_readout_next':
                auxiliary_loss = F.cross_entropy(auxiliary_logits.reshape(-1, 50257), labels.reshape(-1), reduction='mean')
            else:
                auxiliary_loss = sparse_proximity_ce(auxiliary_logits, targets)
            total_loss = .5 * primary_loss + .5 * auxiliary_loss
        for loss in (primary_loss, auxiliary_loss, total_loss):
            if loss is not None:
                C.require(type(loss) is torch.Tensor and loss.ndim == 0 and loss.dtype == torch.float32
                          and loss.device == device and bool(torch.isfinite(loss)), 'Finite FP32 scalar losses required')
        primary = PrimaryLogits(self.identity, 'top_model_primary_next_logits_v1', primary_logits)
        return TrainingResult(self.identity, 'top_model_training_loss_v1', primary,
                              auxiliary_logits, primary_loss, auxiliary_loss, total_loss)

    def primary_logits(self, ids):
        device = self.validate()
        C.require(type(ids) is torch.Tensor and ids.ndim == 2 and 1 <= ids.shape[1] <= 128,
                  'Bounded uncached primary sequence required')
        self._ids(ids, length=ids.shape[1], device=device)
        logits, _ = self.native(ids)
        return PrimaryLogits(self.identity, 'top_model_primary_next_logits_v1', logits)

    def forward(self, *args, **kwargs):
        raise ValueError('Use forward_train or uncached primary_logits; general/cache API unqualified')

    def state_dict(self, *args, **kwargs):
        raise ValueError('TOP model-math source has no qualified checkpoint/export API')

    def load_state_dict(self, *args, **kwargs):
        raise ValueError('TOP model-math source has no qualified state loader')
