"""Finite invalid transactions on real models; called only by admitted driver."""
import contextlib
import hashlib
import json
import torch


def tensor_record(value, identity=True):
    record = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device),
                  layout=str(value.layout), requires_grad=value.requires_grad)
    if identity:
        record['id'] = id(value)
    if value.device.type == 'meta':
        record['storage'] = 'meta'
    elif value.layout == torch.sparse_coo:
        dense = value.coalesce()
        record['indices'] = tensor_record(dense.indices(), False)
        record['values'] = tensor_record(dense.values(), False)
    else:
        if identity:
            record['storage'] = value.untyped_storage().data_ptr()
        h = hashlib.sha256()
        flat = value.detach().reshape(-1)
        for start in range(0, flat.numel(), 65536):
            block = flat[start:start+65536].to(device='cpu').contiguous()
            h.update(block.view(torch.uint8).numpy().tobytes())
        record['sha256'] = h.hexdigest()
    return record


def snapshot(model, optimizer, inputs, labels):
    tensors = {}

    def freeze(value):
        if isinstance(value, torch.Tensor):
            key = str(id(value))
            if key not in tensors:
                tensors[key] = tensor_record(value)
            return ['tensor', key]
        if isinstance(value, dict):
            return ['dict', [[freeze(k), freeze(v)] for k, v in value.items()]]
        if isinstance(value, (tuple, list)):
            return [type(value).__name__, [freeze(x) for x in value]]
        if value is None or type(value) in (str, bool, int):
            return [type(value).__name__, value]
        if type(value) is float:
            return ['float', value.hex()]
        return [type(value).__name__, id(value), repr(value)]

    parameters = list(model.named_parameters())
    result = dict(parameters=freeze(parameters), gradients=freeze([(n, p.grad) for n, p in parameters]),
                  buffers=freeze(list(model.named_buffers())), inputs=freeze(inputs), labels=freeze(labels),
                  config=freeze(model.config), spec=repr(model.spec), steps=optimizer._steps,
                  groups=freeze(optimizer.param_groups),
                  children=[dict(id=id(c), type=type(c).__name__, state=freeze(c.state),
                                 groups=freeze(c.param_groups),
                                 pre_hooks=freeze(c._optimizer_step_pre_hooks),
                                 post_hooks=freeze(c._optimizer_step_post_hooks))
                            for c in optimizer._optimizers],
                  modules=[(n, id(m), m.training) for n, m in model.named_modules()],
                  cpu_rng=tensor_record(torch.random.get_rng_state(), False),
                  mps_rng=tensor_record(torch.mps.get_rng_state(), False), tensors=tensors)
    return result


@contextlib.contextmanager
def invalid_case(name, model, optimizer, inputs, labels):
    """Inject one predeclared invalidity and restore it after the transaction."""
    undo = []
    stack = contextlib.ExitStack()
    x, y = inputs, labels
    call = lambda: model.forward_train(x, y)
    parameters = dict(model.ordered_parameters())
    small_name, small = min(parameters.items(), key=lambda item: item[1].numel())
    muon, adam = optimizer._optimizers
    mp = min((g['params'][0] for g in muon.param_groups), key=lambda p:p.numel())
    ap = min((g['params'][0] for g in adam.param_groups), key=lambda p:p.numel())
    other_device = 'mps' if inputs.device.type == 'cpu' else 'cpu'

    def item(mapping, key, value, remove=False):
        exists = key in mapping
        old = mapping.get(key)
        def restore():
            if exists: mapping[key] = old
            else: mapping.pop(key, None)
        undo.append(restore)
        if remove: del mapping[key]
        else: mapping[key] = value

    def attr(obj, key, value):
        old = getattr(obj, key)
        undo.append(lambda:setattr(obj, key, old))
        setattr(obj, key, value)

    def scalar(tensor, value):
        old = tensor.detach().reshape(-1)[0].clone()
        def restore():
            with torch.no_grad(): tensor.reshape(-1)[0].copy_(old)
        undo.append(restore)
        with torch.no_grad(): tensor.reshape(-1)[0] = value

    try:
        if name == 'ids_float': x = inputs.float()
        elif name == 'ids_bool': x = inputs.bool()
        elif name == 'ids_int32': x = inputs.int()
        elif name == 'ids_list': x = inputs.tolist()
        elif name == 'ids_rank1': x = inputs[0]
        elif name == 'ids_rank3': x = inputs[None]
        elif name == 'ids_empty_batch': x = inputs[:0]
        elif name == 'ids_batch9': x = inputs[:1].expand(9, -1)
        elif name == 'ids_time129': x = torch.cat((inputs, inputs[:, :1]), 1)
        elif name == 'ids_empty_time': x = inputs[:, :0]
        elif name in ('ids_negative', 'ids_50257'):
            x = inputs.clone(); x[0, 0] = -1 if name == 'ids_negative' else 50257
        elif name == 'labels_wrong_shape': y = labels[:, :-1]
        elif name == 'labels_float': y = labels.float()
        elif name in ('labels_negative', 'labels_50257'):
            y = labels.clone(); y[0, 0] = -1 if name == 'labels_negative' else 50257
        elif name == 'foreign_device_ids': x = inputs.to(other_device)
        elif name == 'foreign_device_labels': y = labels.to(other_device)
        elif name == 'ids_sparse': x = inputs.to_sparse()
        elif name == 'ids_meta': x = torch.empty(inputs.shape, dtype=torch.int64, device='meta')
        elif name == 'autocast': stack.enter_context(torch.autocast(inputs.device.type, dtype=torch.float16))
        elif name == 'changed_config': item(model.config, 'primary_objective', 'invalid')
        elif name == 'changed_spec': attr(model, 'spec', None)
        elif name == 'replaced_parameter':
            actual_name = next(n for n,p in model.named_parameters() if p is small)
            path, leaf = actual_name.rsplit('.', 1)
            parent = model.get_submodule(path)
            item(parent._parameters, leaf, torch.nn.Parameter(small.detach().clone()))
        elif name == 'nonfinite_parameter': scalar(small, float('nan'))
        elif name == 'extra_parameter': item(model._parameters, 'unexpected', torch.nn.Parameter(torch.zeros(1, device=inputs.device)))
        elif name in ('replaced_buffer', 'mutated_buffer'):
            actual_name, buffer = next(iter(model.named_buffers()))
            if name == 'mutated_buffer': scalar(buffer, buffer.reshape(-1)[0] + 1)
            else:
                parent_path, _, leaf = actual_name.rpartition('.')
                parent = model.get_submodule(parent_path) if parent_path else model
                item(parent._buffers, leaf, buffer.clone())
        elif name == 'model_forward': call = lambda:model(inputs)
        elif name == 'model_export': call = model.state_dict
        elif name == 'model_load': call = lambda:model.load_state_dict({})
        elif name == 'optimizer_export': call = optimizer.state_dict
        elif name == 'optimizer_load': call = lambda:optimizer.load_state_dict({})
        else:
            call = optimizer.step
            if name == 'missing_gradient': attr(small, 'grad', None)
            elif name in ('nan_gradient', 'inf_gradient'): scalar(small.grad, float('nan') if name=='nan_gradient' else float('inf'))
            elif name == 'foreign_state_key': item(muon.state, torch.nn.Parameter(torch.zeros(1)), {})
            elif name == 'missing_state_parameter': item(muon.state, mp, None, remove=True)
            elif name == 'missing_state_field': item(muon.state[mp], 'momentum_buffer', None, remove=True)
            elif name == 'extra_state_field': item(muon.state[mp], 'unexpected', 1)
            elif name == 'nan_momentum': scalar(muon.state[mp]['momentum_buffer'], float('nan'))
            elif name == 'inf_adam_average': scalar(adam.state[ap]['exp_avg'], float('inf'))
            elif name == 'negative_second_moment': scalar(adam.state[ap]['exp_avg_sq'], -1.)
            elif name == 'wrong_state_shape': item(muon.state[mp], 'momentum_buffer', torch.zeros(1, device=inputs.device))
            elif name == 'wrong_state_dtype': item(muon.state[mp], 'momentum_buffer', muon.state[mp]['momentum_buffer'].half())
            elif name == 'wrong_state_device': item(muon.state[mp], 'momentum_buffer', muon.state[mp]['momentum_buffer'].to(other_device))
            elif name == 'state_alias_parameter': item(muon.state[mp], 'momentum_buffer', mp.detach())
            elif name == 'state_alias_gradient': item(muon.state[mp], 'momentum_buffer', mp.grad)
            elif name == 'state_alias_other_state': item(adam.state[ap], 'exp_avg', adam.state[ap]['exp_avg_sq'])
            elif name in ('adam_step_nan', 'adam_step_wrong'):
                scalar(adam.state[ap]['step'], float('nan') if name=='adam_step_nan' else optimizer._steps+3.)
            elif name == 'adam_step_rank1': item(adam.state[ap], 'step', adam.state[ap]['step'].reshape(1))
            elif name == 'state_before_first_step': attr(optimizer, '_steps', 0)
            elif name == 'replaced_child': attr(optimizer, '_optimizers', tuple(reversed(optimizer._optimizers)))
            elif name == 'child_hook':
                handle = muon.register_step_pre_hook(lambda *args:None); undo.append(handle.remove)
            elif name == 'changed_group_option': item(muon.param_groups[0], 'momentum', .2)
            elif name == 'inconsistent_base_lr': item(adam.param_groups[0], 'lr', .1)
            elif name == 'duplicate_parameter_group':
                muon.param_groups.append(muon.param_groups[0]); undo.append(muon.param_groups.pop)
            elif name == 'foreign_parameter_group': item(muon.param_groups[0], 'params', [torch.nn.Parameter(mp.detach().clone())])
            elif name == 'state_sparse': item(muon.state[mp], 'momentum_buffer', muon.state[mp]['momentum_buffer'].to_sparse())
            elif name == 'state_meta': item(muon.state[mp], 'momentum_buffer', torch.empty(mp.shape, device='meta'))
            elif name == 'state_float64': item(muon.state[mp], 'momentum_buffer', muon.state[mp]['momentum_buffer'].double())
            else: raise AssertionError('Unimplemented fixed rejection: '+name)
        yield call, x, y
    finally:
        for restore in reversed(undo): restore()
        stack.close()


def run(name, model, optimizer, inputs, labels, publish, prefix):
    with invalid_case(name, model, optimizer, inputs, labels) as (call, x, y):
        before = snapshot(model, optimizer, x, y)
        error = None
        try:
            call()
        except ValueError as exc:
            error = str(exc)
        after = snapshot(model, optimizer, x, y)
        publish(prefix+'-reject-'+name+'.json', dict(case=name, rejected=error is not None,
                error=error, preserved=before == after, before=before, after=after))
        assert error is not None and before == after, 'Rejected without mutation: '+name
