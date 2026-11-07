"""Raw-byte token composer math SOURCE; unexecuted and unqualified.

One input token per row; no tokenizer, model, target, optimizer, cache or state
adapter. Functional execution requires its own independent finite proof admission.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F

KIND = 'rsi_byte_composer_math_v1'
WIDTHS = (1, 2, 3, 4, 5, 6)
CHANNELS = (25, 50, 75, 100, 125, 150)
SIGNATURE = (KIND, 259, 15, WIDTHS, CHANNELS, 525, 1, 256, 130, 1024)


def require(value, reason):
    if not value:
        raise ValueError(reason)


class ByteComposer(nn.Module):
    def __init__(self, private_seed):
        super().__init__()
        require(type(private_seed) is int and 0 <= private_seed < 2**63,
                'Explicit bounded private initializer required')
        require(torch.get_default_dtype() == torch.float32 and
                torch.get_default_device().type == 'cpu', 'CPU FP32 construction required')
        require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'),
                'Autocast forbidden')
        generator = torch.Generator(device='cpu').manual_seed(private_seed)

        def parameter(shape, lo=-.05, hi=.05):
            value = torch.empty(shape, device='cpu', dtype=torch.float32)
            with torch.no_grad():
                value.uniform_(lo, hi, generator=generator)
            return nn.Parameter(value)

        # Trainable rows: bytes0..255, begin257, end258, special259.
        # Pad256 has no trainable row; a differentiable zero mask removes it.
        self.byte_weight = parameter((259, 15))
        self.filters = nn.ParameterList([parameter((c, 15*w)) for w, c in zip(WIDTHS, CHANNELS)])
        self.filter_biases = nn.ParameterList([parameter((c,)) for c in CHANNELS])
        self.highway_weight = parameter((525, 525))
        self.highway_bias = parameter((525,))
        self.transform_weight = parameter((525, 525))
        self.transform_bias = parameter((525,), -2.05, -1.95)
        bound = 1. / math.sqrt(525)
        self.projection_weight = parameter((256, 525), -bound, bound)
        self._signature = SIGNATURE
        self._bindings = tuple((name, p) for name, p in self.named_parameters())
        self.validate()

    @property
    def identity(self):
        return self._signature

    def validate(self):
        require(self._signature == SIGNATURE, 'Changed composer identity')
        expected = {'byte_weight': (259, 15), 'highway_weight': (525, 525),
                    'highway_bias': (525,), 'transform_weight': (525, 525),
                    'transform_bias': (525,), 'projection_weight': (256, 525)}
        expected.update({f'filters.{i}': (c, 15*w) for i, (w, c) in enumerate(zip(WIDTHS, CHANNELS))})
        expected.update({f'filter_biases.{i}': (c,) for i, c in enumerate(CHANNELS)})
        items = tuple(self.named_parameters())
        require(len(items) == 18 and set(name for name, _ in items) == set(expected),
                'Exact complete component parameter inventory')
        require(len(items) == len(self._bindings) and all(name == old and p is original
                for (name, p), (old, original) in zip(items, self._bindings)),
                'Replaced or reordered component parameters')
        require(not tuple(self.named_buffers()), 'No hidden codebook/state buffers')
        require(set(self._modules) == {'filters', 'filter_biases'} and
                type(self.filters) is nn.ParameterList and type(self.filter_biases) is nn.ParameterList,
                'No hidden component modules')
        device = self.byte_weight.device
        require(device.type in {'cpu', 'mps'}, 'CPU or MPS component only')
        require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'),
                'Autocast forbidden')
        require(sum(p.numel() for _, p in items) == 725235, 'Exact parameter element count')
        addresses = []
        for name, p in items:
            require(type(p) is nn.Parameter and tuple(p.shape) == expected[name] and
                    p.dtype == torch.float32 and p.device == device and p.layout == torch.strided and
                    p.is_contiguous() and p.is_leaf and p.requires_grad and bool(torch.isfinite(p).all()),
                    'Own finite dense FP32 leaf parameters required before lookup')
            addresses.append(p.untyped_storage().data_ptr())
        require(len(set(addresses)) == len(addresses), 'No aliased parameter storage')
        return device

    def forward(self, codes, lengths):
        device = self.validate()
        require(type(codes) is torch.Tensor and type(lengths) is torch.Tensor and
                codes.layout == lengths.layout == torch.strided and
                codes.dtype == lengths.dtype == torch.int64 and
                codes.device == lengths.device == device,
                'Dense same-device int64 code and length tensors required')
        require(codes.ndim == 2 and lengths.ndim == 1 and
                1 <= codes.shape[0] <= 1024 and 3 <= codes.shape[1] <= 130 and
                lengths.shape[0] == codes.shape[0], 'Bounded exact input shapes required')
        require(bool(((codes >= 0) & (codes <= 259)).all()) and
                bool(((lengths >= 3) & (lengths <= codes.shape[1])).all()), 'Bounded codes and lengths')
        positions = torch.arange(codes.shape[1], device=device)[None, :]
        valid = positions < lengths[:, None]
        interior = (positions > 0) & (positions < lengths[:, None]-1)
        require(bool((codes[:, 0] == 257).all()) and
                bool((codes.gather(1, (lengths-1)[:, None]) == 258).all()), 'Exact boundary symbols')
        require(bool((codes.masked_select(~valid) == 256).all()), 'All trailing positions are pad256')
        middle = codes.masked_select(interior)
        require(bool(((middle <= 255) | (middle == 259)).all()), 'Bytes or explicit special interior only')
        has_special = ((codes == 259) & interior).any(dim=1)
        require(bool((~has_special | ((lengths == 3) & (codes[:, 1] == 259))).all()),
                'Special appears alone between boundaries')
        # All input/parameter preflight above precedes differentiable lookup.
        # Local minimum-width padding allows unused wide filters to have genuine
        # zero gradients, rather than absent gradient slots, on short tokens.
        if codes.shape[1] < 6:
            codes = F.pad(codes, (0, 6-codes.shape[1]), value=256)
        positions = torch.arange(codes.shape[1], device=device)[None, :]
        valid = positions < lengths[:, None]
        rows = codes - (codes > 256).to(torch.int64)
        embedded = F.embedding(rows, self.byte_weight).masked_fill(~valid[:, :, None], 0.)
        pooled = []
        for width, weight, bias in zip(WIDTHS, self.filters, self.filter_biases):
            count = codes.shape[1]-width+1
            windows = embedded.unfold(1, width, 1).contiguous().view(codes.shape[0], count, 15*width)
            values = torch.tanh(F.linear(windows, weight, bias))
            allowed = positions[:, :count]+width <= lengths[:, None]
            maximum = values.masked_fill(~allowed[:, :, None], -2.).amax(dim=1)
            pooled.append(torch.where((lengths >= width)[:, None], maximum, torch.zeros_like(maximum)))
        y = torch.cat(pooled, dim=1)
        transformed = F.relu(F.linear(y, self.highway_weight, self.highway_bias))
        gate = torch.sigmoid(F.linear(y, self.transform_weight, self.transform_bias))
        z = gate*transformed + (1.-gate)*y
        output = F.linear(z, self.projection_weight)
        require(bool(torch.isfinite(output).all()), 'Finite component output required')
        return output

    def state_dict(self, *args, **kwargs):
        raise ValueError('Byte composer serialization is unqualified')

    def load_state_dict(self, *args, **kwargs):
        raise ValueError('Byte composer loading is unqualified')
