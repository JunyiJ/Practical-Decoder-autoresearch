"""Isolated composer-only FP32 Muon arithmetic and explicit state identity."""
import math
import torch

KIND = 'composer_fp32_muon_v1'
SHAPES = {'composer.highway_weight': (525, 525), 'composer.transform_weight': (525, 525),
          'composer.projection_weight': (256, 525),
          **{f'composer.filters.{i}': (25*(i+1), 15*(i+1)) for i in range(6)}}


def orthogonalize(update):
    assert update.dtype == torch.float32 and update.ndim == 2
    transposed = update.shape[0] > update.shape[1]
    value = update.T.clone() if transposed else update.clone()
    value.div_(value.norm().clamp(min=1e-7))
    for _ in range(5):
        gram = value @ value.T
        polynomial = torch.addmm(gram, gram, gram, beta=-4.775, alpha=2.0315)
        value = torch.addmm(value, polynomial, value, beta=3.4445)
    return value.T if transposed else value


class ComposerFP32Muon:
    def __init__(self, named_parameters, lr=.00125):
        self.items = tuple(named_parameters)
        assert [n for n, p in self.items] == list(SHAPES)
        assert type(lr) is float and math.isfinite(lr) and lr >= 0
        self.param_groups = [dict(name=n, params=[p], lr=lr) for n, p in self.items]
        self.buffers = {}
        self.steps = 0
        self.validate()

    def validate(self):
        device = self.items[0][1].device
        assert device.type in ('cpu', 'mps')
        assert type(self.steps) is int and self.steps >= 0
        assert list(self.buffers) == (list(SHAPES) if self.steps else [])
        addresses = []
        for (name, p), group in zip(self.items, self.param_groups, strict=True):
            assert type(p) is torch.nn.Parameter and p.dtype == torch.float32 and p.device == device
            assert tuple(p.shape) == SHAPES[name] and p.is_leaf and p.is_contiguous() and torch.isfinite(p).all()
            assert set(group) == {'name', 'params', 'lr'} and group['name'] == name
            assert len(group['params']) == 1 and group['params'][0] is p
            assert type(group['lr']) is float and math.isfinite(group['lr']) and group['lr'] >= 0
            addresses.append(p.untyped_storage().data_ptr())
            if self.steps:
                b = self.buffers[name]
                assert b.shape == p.shape and b.device == device and b.dtype == torch.float32 and not b.requires_grad
                assert b.is_contiguous() and torch.isfinite(b).all()
                addresses.append(b.untyped_storage().data_ptr())
        assert len(addresses) == len(set(addresses))
        return device

    @torch.no_grad()
    def step(self):
        self.validate()
        # Validate every gradient before allocating state or changing parameters.
        for name, p in self.items:
            g = p.grad
            assert type(g) is torch.Tensor and g.shape == p.shape and g.device == p.device and g.dtype == p.dtype
            assert g.layout == torch.strided and torch.isfinite(g).all()
        staged = []
        for (name, p), group in zip(self.items, self.param_groups, strict=True):
            buf = self.buffers[name].clone() if self.steps else torch.zeros_like(p)
            buf.lerp_(p.grad, 1-.95)
            update = p.grad.lerp(buf, .95)
            direction = orthogonalize(update)
            rate = group['lr']*.2*math.sqrt(max(p.shape))
            value = p.clone().add_(direction, alpha=-rate)
            assert torch.isfinite(value).all() and torch.isfinite(buf).all()
            staged.append((name, p, value, buf))
        for name, p, value, buf in staged:
            p.copy_(value)
            self.buffers[name] = buf
        self.steps += 1

    def snapshot(self):
        self.validate()
        return dict(kind=KIND, steps=self.steps, lrs=[g['lr'] for g in self.param_groups],
                    parameters={n: p.detach().cpu().clone() for n, p in self.items},
                    momentum={n: b.detach().cpu().clone() for n, b in self.buffers.items()})

    @torch.no_grad()
    def restore(self, payload):
        device = self.validate()
        assert self.steps == 0 and not self.buffers
        assert set(payload) == {'kind','steps','lrs','parameters','momentum'} and payload['kind'] == KIND
        step = payload['steps']
        assert type(step) is int and step >= 0 and list(payload['parameters']) == list(SHAPES)
        assert list(payload['momentum']) == (list(SHAPES) if step else [])
        assert len(payload['lrs']) == len(self.items)
        assert all(type(v) is float and math.isfinite(v) and v >= 0 for v in payload['lrs'])
        for panel in [payload['parameters'], payload['momentum']]:
            for n, value in panel.items():
                assert value.device.type == 'cpu' and value.dtype == torch.float32 and tuple(value.shape) == SHAPES[n]
                assert torch.isfinite(value).all()
        staged = {n: v.to(device).clone() for n, v in payload['momentum'].items()}
        for i, (n, p) in enumerate(self.items):
            p.copy_(payload['parameters'][n])
            self.param_groups[i]['lr'] = payload['lrs'][i]
        self.buffers = staged
        self.steps = step
        self.validate()
