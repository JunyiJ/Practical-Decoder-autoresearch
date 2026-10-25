"""Tiny CPU tensor checks only; no model training or inference benchmarks."""
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.attention.output_gate import HeadwiseOutputGate, make_output_gate


@pytest.mark.parametrize('width', [64, 128, 256])
def test_zero_init_and_static_half_have_identical_outputs_and_input_grad(width):
    torch.manual_seed(9309)
    before = torch.get_rng_state().clone()
    learned = HeadwiseOutputGate(width, width // 32, 'sigmoid_headwise_zero').double()
    fixed = HeadwiseOutputGate(width, width // 32, 'static_half').double()
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in learned.parameters()) == width * width // 32
    assert len(fixed.state_dict()) == 0
    x = torch.randn(1, 3, width, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, width // 32, 3, 32, dtype=torch.float64, requires_grad=True)
    a, b = learned(x, y), fixed(x, y)
    assert torch.equal(a, b) and torch.equal(a, y * .5)
    a.sum().backward()
    assert torch.equal(x.grad, torch.zeros_like(x))
    assert torch.equal(y.grad, torch.full_like(y, .5))
    # Zero gate weights do not block learning of the gate itself.
    assert learned.weight.grad.abs().sum() > 0


@pytest.mark.parametrize('width', [64, 128, 256])
def test_numpy_formula_and_both_input_and_weight_gradients(width):
    rng = np.random.default_rng(9310)
    x = rng.normal(size=(2, 3, width))
    y = rng.normal(size=(2, width // 32, 3, 32))
    weight = rng.normal(0, .1, size=(width // 32, width))
    coefficient = rng.normal(size=y.shape)
    gate = HeadwiseOutputGate(width, width // 32, 'sigmoid_headwise_zero').double()
    with torch.no_grad():
        gate.weight.copy_(torch.from_numpy(weight))
    tx = torch.tensor(x, requires_grad=True)
    ty = torch.tensor(y, requires_grad=True)
    result = gate(tx, ty)
    (result * torch.from_numpy(coefficient)).sum().backward()
    sigmoid = 1 / (1 + np.exp(-np.einsum('bsd,hd->bsh', x, weight)))
    scores = sigmoid.transpose(0, 2, 1)[..., None]
    dz = (coefficient * y).sum(-1).transpose(0, 2, 1) * sigmoid * (1 - sigmoid)
    expected_dx = np.einsum('bsh,hd->bsd', dz, weight)
    expected_dw = np.einsum('bsh,bsd->hd', dz, x)
    for actual, expected in [(result.detach(), y * scores), (tx.grad, expected_dx),
                             (ty.grad, coefficient * scores), (gate.weight.grad, expected_dw)]:
        np.testing.assert_allclose(actual.numpy(), expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize('width', [64, 128, 256])
@pytest.mark.parametrize('kind', ['static_half', 'sigmoid_headwise_zero'])
def test_query_slices_match_full_output_and_have_no_stateful_cache(width, kind):
    gate = HeadwiseOutputGate(width, width // 32, kind)
    with torch.no_grad():
        if gate.weight is not None:
            gate.weight.copy_(torch.linspace(-.1, .1, gate.weight.numel()).reshape_as(gate.weight))
    x = torch.arange(4 * width, dtype=torch.float32).reshape(1, 4, width) / width
    y = torch.arange(4 * width, dtype=torch.float32).reshape(1, width // 32, 4, 32) / width
    full = gate(x, y)
    incremental = torch.cat([gate(x[:, j:j+1], y[:, :, j:j+1]) for j in range(4)], dim=2)
    torch.testing.assert_close(full, incremental, atol=1e-6, rtol=1e-6)
    changed = x.clone(); changed[:, 2:] += 100
    assert torch.equal(full[:, :, :2], gate(changed, y)[:, :, :2])


@pytest.mark.parametrize('kind', [False, None, 'sigmoid', 'static', 0])
def test_unknown_kind_rejected(kind):
    with pytest.raises(ValueError):
        make_output_gate(SimpleNamespace(attention_output_gate=kind, dim=64, n_heads=2))


@pytest.mark.parametrize('dim,heads', [(32, 1), (64, 1), (128, 2), (512, 16), (64., 2), (64, True)])
def test_unreviewed_shape_rejected(dim, heads):
    with pytest.raises(ValueError):
        HeadwiseOutputGate(dim, heads, 'static_half')


def test_default_has_no_module_or_rng_draw():
    before = torch.get_rng_state().clone()
    assert make_output_gate(SimpleNamespace()) is None
    assert make_output_gate(SimpleNamespace(attention_output_gate='none')) is None
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize('x,y', [(torch.zeros(1, 3, 63), torch.zeros(1, 2, 3, 32)),
                                (torch.zeros(1, 3, 64), torch.zeros(1, 2, 4, 32)),
                                (torch.zeros(1, 3, 64), torch.zeros(1, 3, 2, 32))])
def test_misaligned_query_or_head_axes_rejected(x, y):
    with pytest.raises(ValueError):
        HeadwiseOutputGate(64, 2, 'static_half')(x, y)
