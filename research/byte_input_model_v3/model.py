"""Own fixed-GPT2 byte-input model math SOURCE, runtime unqualified."""
from dataclasses import dataclass
import hashlib
import sys
import torch
from torch import nn
from torch.nn import functional as F
from omegaconf import OmegaConf
from research.final_ln_scale_model_v1.runtime.models.gpt import GPT as Native
from research.byte_composer_v2 import ByteComposer
from .codebook import Codebook
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
    loss: torch.Tensor


class ByteInputModel(nn.Module):
    def __init__(self, spec, codebook=None):
        super().__init__()
        self.spec = spec
        self._identity = C.signature(spec)
        self.config = C.model_config(spec)
        C.require(torch.get_default_dtype() == torch.float32 and
                  torch.get_default_device().type == 'cpu' and sys.byteorder == 'little',
                  'Little-endian CPU FP32 construction required')
        C.require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'), 'Autocast forbidden')
        if spec.variant == 'native':
            C.require(codebook is None, 'Native cannot accept a codebook')
        else:
            C.require(type(codebook) is Codebook, 'Own builder codebook required')
            codebook.validate()
            C.require(codebook.variant == spec.variant and codebook.identity == spec.codebook_identity,
                      'Codebook/spec identity differs')
        # Preserve the complete genuine native initialization, before any new private draw.
        native = Native(C.native_config())
        self._native_cfg = native.cfg
        self._own_native_shapes = tuple((n, tuple(p.shape)) for n, p in native.named_parameters())
        C.require(len(self._own_native_shapes) == 88 and
                  sum(p.numel() for p in native.parameters()) == 28492416,
                  'Genuine native initializer inventory')
        C.require(native.width_mult == 4. and native.position_encoding == 'rope' and
                  native.input_embedding_scale == 1. and native.output_bias is None and
                  native.lm_head.bias is None and native.logit_z_signature is None and
                  native.weight_tying_signature is None and
                  (native.value_residual_signature is None or native.value_residual_signature[0] == 'local'),
                  'Fixed native body/readout/primary objective required')
        if spec.variant == 'native':
            self.native = native
        else:
            self.blocks = native.transformer.h
            self.final_norm = native.transformer.ln_f
            self.readout = native.lm_head
            self.composer = ByteComposer(spec.composer_seed)
            # The genuine native local variable is not registered in this tree.
            # Its displaced input table is not retained as a hidden parameter.
            codes = torch.frombuffer(bytearray(codebook.codes), dtype=torch.uint16).to(dtype=torch.int64)
            lengths = torch.frombuffer(bytearray(codebook.lengths), dtype=torch.uint16).to(dtype=torch.int64)
            self.register_buffer('code_ids', codes.reshape(50257, 130))
            self.register_buffer('code_lengths', lengths.reshape(50257))
        self._own_parameter_bindings = self.ordered_parameters()
        self._own_parameter_inventory = tuple((n, tuple(p.shape)) for n, p in self._own_parameter_bindings)
        self._own_module_bindings = tuple(self.named_modules())
        self._own_buffer_inventory = tuple((n, tuple(x.shape), x.dtype) for n, x in self.named_buffers())
        self._own_expected_buffer_hashes = {n: self._buffer_digest(x) for n, x in self.named_buffers()}
        if spec.variant != 'native':
            C.require(self._own_expected_buffer_hashes['code_ids'] == codebook.codes_i64_sha256 and
                      self._own_expected_buffer_hashes['code_lengths'] == codebook.lengths_i64_sha256,
                      'Staged codebook bytes differ')
        self._refresh_buffer_bindings()
        self._own_semantics = self._semantic_inventory()
        self.validate()

    @property
    def identity(self):
        return self._identity

    def ordered_parameters(self):
        if self.spec.variant == 'native':
            return tuple(self.native.named_parameters())
        items = tuple(('transformer.h.'+n, p) for n, p in self.blocks.named_parameters())
        items += tuple(('transformer.ln_f.'+n, p) for n, p in self.final_norm.named_parameters())
        items += tuple(('lm_head.'+n, p) for n, p in self.readout.named_parameters())
        return items + tuple(('composer.'+n, p) for n, p in self.composer.named_parameters())

    def _refresh_buffer_bindings(self):
        self._own_buffer_bindings = tuple(self.named_buffers())

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        # Standard device migration replaces buffer objects, not their expected contents.
        self._refresh_buffer_bindings()
        self.validate()
        return result

    def _buffer_digest(self, tensor):
        result = hashlib.sha256()
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), 131072):
            part = flat[start:start+131072].to(device='cpu').contiguous()
            result.update(bytes(part.view(torch.uint8).reshape(-1).tolist()))
        return result.hexdigest()

    def _semantic_inventory(self):
        result = []
        for name, module in self.named_modules():
            attributes = []
            for key, value in vars(module).items():
                if key == 'training' or key.startswith('_own_'):
                    continue
                frozen = C.primitive(value)
                if frozen is not None:
                    attributes.append((key, frozen))
            result.append((name, tuple(sorted(attributes))))
        return tuple(result)

    def validate(self):
        C.require(C.signature(self.spec) == self._identity and
                  C.same_typed(self.config, C.model_config(self.spec)), 'Changed own model identity/config')
        C.require(C.same_typed(OmegaConf.to_container(self._native_cfg, resolve=True), C.native_model()),
                  'Changed true native initializer recipe')
        expected_modules = {'native'} if self.spec.variant == 'native' else {'blocks', 'final_norm', 'readout', 'composer'}
        C.require(set(self._modules) == expected_modules and not self._parameters,
                  'Foreign root module/hidden parameter')
        modules = tuple(self.named_modules())
        C.require(len(modules) == len(self._own_module_bindings) and
                  all(n == old and m is original for (n, m), (old, original) in zip(modules, self._own_module_bindings)),
                  'Foreign or replaced common/composer modules')
        for _, module in modules:
            C.require(module.training == self.training and not module._forward_hooks and
                      not module._forward_pre_hooks and not module._backward_hooks and
                      not module._backward_pre_hooks, 'Consistent mode and no hooks required')
        C.require(C.same_typed(self._semantic_inventory(), self._own_semantics), 'Changed body/composer scalar policy')
        items = self.ordered_parameters()
        tensors, elements, _, _ = C.counts(self.spec.variant)
        C.require(len(items) == tensors and sum(p.numel() for _, p in items) == elements and
                  tuple((n, tuple(p.shape)) for n, p in items) == self._own_parameter_inventory,
                  'Exact full own parameter inventory')
        C.require(len(items) == len(self._own_parameter_bindings) and
                  all(n == old and p is original for (n, p), (old, original) in zip(items, self._own_parameter_bindings)),
                  'Replaced/reordered parameter objects')
        registered = tuple(self.parameters())
        C.require(len(registered) == tensors and {id(p) for p in registered} == {id(p) for _, p in items},
                  'Registered and canonical coverage differ')
        device = items[0][1].device
        C.require(device.type in ('cpu', 'mps') and not torch.is_autocast_enabled('cpu')
                  and not torch.is_autocast_enabled('mps'), 'CPU/MPS FP32 without autocast required')
        addresses = []
        for _, p in items:
            C.require(type(p) is nn.Parameter and p.device == device and p.dtype == torch.float32
                      and p.layout == torch.strided and p.is_leaf and p.requires_grad
                      and bool(torch.isfinite(p).all()), 'Finite dense trainable FP32 parameters')
            addresses.append(p.untyped_storage().data_ptr())
        buffers = tuple(self.named_buffers())
        C.require(tuple((n, tuple(x.shape), x.dtype) for n, x in buffers) == self._own_buffer_inventory and
                  len(buffers) == len(self._own_buffer_bindings) and
                  all(n == old and x is original for (n, x), (old, original) in zip(buffers, self._own_buffer_bindings)),
                  'Foreign/replaced buffer identity or inventory')
        for name, value in buffers:
            C.require(value.device == device and value.layout == torch.strided and value.is_contiguous(),
                      'Own same-device constant buffers')
            C.require(self._buffer_digest(value) == self._own_expected_buffer_hashes[name], 'Codebook bytes differ or common buffer mutated')
            addresses.append(value.untyped_storage().data_ptr())
        C.require(len(set(addresses)) == len(addresses), 'Aliased parameter/buffer storage')
        if self.spec.variant == 'native':
            C.require(type(self.native) is Native, 'Genuine native child only')
        else:
            C.require(type(self.composer) is ByteComposer and len(self.blocks) == 4, 'Qualified component and four native blocks')
            self.composer.validate()
        return device

    def _ids(self, ids, device, shape=None):
        C.require(type(ids) is torch.Tensor and ids.layout == torch.strided and ids.dtype == torch.int64
                  and ids.device == device and ids.ndim == 2 and 1 <= ids.shape[0] <= 8
                  and 1 <= ids.shape[1] <= 128, 'Bounded plain same-device ID tensor')
        C.require(shape is None or tuple(ids.shape) == shape, 'Exact label/input shape required')
        C.require(bool(((ids >= 0) & (ids < 50257)).all()), 'All ordinary IDs in fixed vocabulary')

    def _composed_hidden(self, inputs):
        indices = inputs.reshape(-1)
        codes = self.code_ids.index_select(0, indices)
        lengths = self.code_lengths.index_select(0, indices)
        hidden = self.composer(codes, lengths).reshape(inputs.shape[0], inputs.shape[1], 256)
        for block in self.blocks:
            hidden, auxiliary = block(hidden, None, 0)
            C.require(auxiliary is None, 'No block auxiliary objective')
        return self.final_norm(hidden)

    def forward_train(self, inputs, labels):
        device = self.validate()
        self._ids(inputs, device)
        self._ids(labels, device, tuple(inputs.shape))
        inputs, labels = inputs.contiguous(), labels.contiguous()
        if self.spec.variant == 'native':
            logits, loss = self.native(inputs, labels)
        else:
            hidden = self._composed_hidden(inputs)
            logits = self.readout(hidden) / 4.
            loss = F.cross_entropy(logits.reshape(-1, 50257), labels.reshape(-1), reduction='mean')
        C.require(loss.ndim == 0 and loss.dtype == torch.float32 and loss.device == device
                  and bool(torch.isfinite(loss)), 'Finite ordinary primary mean CE required')
        return TrainingResult(self.identity, 'byte_input_primary_training_v1',
                              PrimaryLogits(self.identity, 'byte_input_primary_logits_v1', logits), loss)

    def primary_logits(self, inputs):
        device = self.validate()
        self._ids(inputs, device)
        if self.spec.variant == 'native':
            logits, _ = self.native(inputs.contiguous())
        else:
            logits = self.readout(self._composed_hidden(inputs.contiguous())) / 4.
        return PrimaryLogits(self.identity, 'byte_input_primary_logits_v1', logits)

    def forward(self, *args, **kwargs):
        raise ValueError('Only forward_train and uncached primary_logits are qualified source APIs')

    def state_dict(self, *args, **kwargs):
        raise ValueError('Byte-input model serialization/export is unqualified')

    def load_state_dict(self, *args, **kwargs):
        raise ValueError('Byte-input model state loading is unqualified')
