"""Unqualified candidate: aggregate all parameter-finiteness predicates once.

Retains complete structural, content, ownership and finite checks. No model math,
parameter update, hashing policy or original consumer is changed. Existing exact
model-type consumers intentionally reject this isolated subclass.
"""
import torch
from torch import nn
from omegaconf import OmegaConf
from research.byte_input_model_v3 import contract as C
from research.byte_input_model_v4.model import ByteInputModel, Native, ByteComposer


class BatchedFiniteModel(ByteInputModel):
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
        finite_checks = []
        for _, p in items:
            C.require(type(p) is nn.Parameter and p.device == device and p.dtype == torch.float32
                      and p.layout == torch.strided and p.is_leaf and p.requires_grad, 'Finite dense trainable FP32 parameters')
            finite_checks.append(torch.isfinite(p).all())
            addresses.append(p.untyped_storage().data_ptr())
        C.require(bool(torch.stack(finite_checks).all()), 'Finite dense trainable FP32 parameters')
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
