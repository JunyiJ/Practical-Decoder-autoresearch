"""Isolated optimizer-state validation prototype; not adopted.

All original state predicates remain. Device finite/nonnegative predicates are
aggregated; CPU scalar step checks remain immediate. With multiple independent
faults, first error ordering may change; rejection before mutation is required.
Exact-type checkpoint/training consumers intentionally reject this subclass.
"""
from collections import defaultdict
import torch
from research.byte_input_model_v3 import contract as C
from research.byte_input_model_v5.optimizer import ByteInputOptimizer


class BatchedStateOptimizer(ByteInputOptimizer):
    def _validate_state(self, device):
        # Child state is inspected without defaultdict indexing: validation itself
        # must neither allocate lazy state nor change an invalid transaction.
        finite_checks, nonnegative_checks = [], []
        storages = {p.untyped_storage().data_ptr() for _, p in self._items}
        storages.update(b.untyped_storage().data_ptr() for b in self.model.buffers())
        storages.update(p.grad.untyped_storage().data_ptr() for _, p in self._items
                        if p.grad is not None and p.grad.layout == torch.strided
                        and p.grad.device.type != 'meta')
        for index, child in enumerate(self._optimizers):
            C.require(not child._optimizer_step_pre_hooks and not child._optimizer_step_post_hooks,
                      'No optimizer step hooks')
            C.require(type(child.state) is defaultdict and child.state.default_factory is dict,
                      'Original lazy state mapping type')
            parameters = [group['params'][0] for group in child.param_groups]
            expected_ids = {id(p) for p in parameters}
            C.require(all(id(p) in expected_ids for p in child.state), 'No foreign state keys')
            if self._steps == 0:
                C.require(not child.state, 'No state before first update')
                continue
            C.require(len(child.state) == len(parameters), 'Complete child state after update')
            fields = {'momentum_buffer'} if index == 0 else {'step', 'exp_avg', 'exp_avg_sq'}
            for parameter in parameters:
                state = child.state.get(parameter)
                C.require(type(state) is dict and set(state) == fields, 'Exact child state fields')
                for name, value in state.items():
                    scalar = name == 'step'
                    C.require(type(value) is torch.Tensor and value.layout == torch.strided and
                              value.dtype == torch.float32 and not value.requires_grad and
                              value.is_contiguous() and
                              value.device == (torch.device('cpu') if scalar else device) and
                              tuple(value.shape) == (() if scalar else tuple(parameter.shape)),
                              'Finite correctly shaped child state')
                    if scalar:
                        C.require(bool(torch.isfinite(value).all()), 'Finite correctly shaped child state')
                        C.require(value.item() == self._steps, 'Exact completed Adam step')
                    else:
                        finite_checks.append(torch.isfinite(value).all())
                        if name == 'exp_avg_sq':
                            nonnegative_checks.append((value >= 0).all())
                    address = value.untyped_storage().data_ptr()
                    C.require(address not in storages, 'No state/parameter/buffer/gradient aliases')
                    storages.add(address)
        if finite_checks:
            C.require(bool(torch.stack(finite_checks).all()), 'Finite correctly shaped child state')
        if nonnegative_checks:
            C.require(bool(torch.stack(nonnegative_checks).all()), 'Nonnegative Adam second moment')
