import copy
import json
import math
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

from src.attention.cache import build_caches
from src.models.gpt import GPT
from src.models.tokenizer_activation import POLICY
from src.models.tokenizer_architecture import architecture_contract, validate_architecture_model
from src.train.optimizers import build_optimizer
from src.utils.checkpoint import load_model_weights


def config(kind=None):
    root = Path(__file__).resolve().parents[1]
    cfg = OmegaConf.create(json.loads((root / 'research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json').read_text())['config'])
    if kind is not None:
        cfg.model.update(tokenizer_architecture_policy=POLICY,
                         gated_activation_kind=kind, gated_activation_width_policy=POLICY)
    return cfg


def groups(optimizer):
    return [{k: v for k, v in g.items() if k != 'params'} for g in optimizer.param_groups]


def test_silu_preserves_incumbent_weights_logits_and_optimizer():
    old, new = config(), config('silu')
    torch.manual_seed(71); a = GPT(old.model)
    torch.manual_seed(71); b = GPT(new.model)
    assert a.state_dict().keys() == b.state_dict().keys()
    for k, v in a.state_dict().items():
        torch.testing.assert_close(v, b.state_dict()[k], rtol=0, atol=0)
    ids = torch.tensor([[100, 201, 302]])
    with torch.no_grad():
        torch.testing.assert_close(a(ids)[0], b(ids)[0], rtol=0, atol=0)
    assert groups(build_optimizer(a, old.training)) == groups(build_optimizer(b, new.training))
    validate_architecture_model(b)


@pytest.mark.parametrize('kind', ['silu', 'gelu_exact'])
def test_meta_parameter_optimizer_counts_and_cache_shapes(kind):
    cfg = config(kind)
    with torch.device('meta'):
        m = GPT(cfg.model)
        o = build_optimizer(m, cfg.training)
        caches = build_caches(cfg, 'meta', batch_size=1, dtype=torch.float32)
    assert sum(p.numel() for p in m.parameters()) == 28492416
    assert sum(p.numel() for child in o.optimizers if type(child) is torch.optim.AdamW
               for g in child.param_groups for p in g['params']) == 25739904
    covered = [id(p) for g in o.param_groups for p in g['params']]
    assert len(covered) == len(set(covered)) == len(list(m.parameters()))
    assert len(caches) == 4
    assert all(b.mlp.gate_activation == kind for b in m.transformer.h)


@pytest.mark.parametrize('field,value', [
    ('gated_activation_kind', 'relu'), ('gated_activation_kind', True),
    ('gated_activation_width_policy', 'mha4_gated_activation_21_8'),
    ('dim', 64), ('attn_type', 'gqa'), ('hidden_dim', 672),
    ('dropout', .1), ('ffn_bias', True), ('gate_multiplier', 1.),
])
def test_rejects_unreviewed_combinations(field, value):
    cfg = config('gelu_exact'); cfg.model[field] = value
    with pytest.raises(ValueError):
        architecture_contract(cfg.model)


def test_old_policy_still_rejects_new_gate_fields():
    cfg = config(); cfg.model.gated_activation_kind = 'gelu_exact'
    with pytest.raises(ValueError):
        architecture_contract(cfg.model)


def test_exact_gelu_formula_and_input_parameter_gradients():
    from src.models.mlp import SwiGLU
    torch.manual_seed(72)
    actual = SwiGLU(8, 21, 0., gate_activation='gelu_exact').double()
    reference = copy.deepcopy(actual)
    x = torch.randn(2, 5, 8, dtype=torch.float64, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    output = actual(x)
    gate = reference.linear_gate(y)
    expected = reference.linear_down(.5 * gate * (1 + torch.erf(gate / math.sqrt(2)))
                                     * reference.linear_up(y))
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    output.square().sum().backward(); expected.square().sum().backward()
    for p, q in zip([x, *actual.parameters()], [y, *reference.parameters()]):
        torch.testing.assert_close(p.grad, q.grad, rtol=1e-11, atol=1e-12)


def test_gelu_full_cached_causal_and_same_shape_checkpoint_rejection():
    cfg = config('gelu_exact'); m = GPT(cfg.model).eval()
    ids = torch.tensor([[11, 22, 33, 44, 55]])
    caches = build_caches(cfg, 'cpu', batch_size=1, dtype=torch.float32)
    with torch.no_grad():
        full = m(ids)[0]
        cached = torch.cat([m(ids[:, :2], cache=caches, start_pos=0)[0],
                            m(ids[:, 2:], cache=caches, start_pos=2)[0]], dim=1)
        torch.testing.assert_close(full, cached, rtol=1e-5, atol=1e-6)
        changed = ids.clone(); changed[:, 3:] += 100
        torch.testing.assert_close(full[:, :3], m(changed)[0][:, :3], rtol=0, atol=0)
    payload = {'cfg': OmegaConf.to_container(cfg), 'model_state': m.state_dict()}
    load_model_weights(m, payload)
    wrong = dict(payload, cfg=OmegaConf.to_container(config('silu')))
    with pytest.raises(ValueError):
        load_model_weights(m, wrong)
    m.transformer.h[0].mlp.gate_activation = 'silu'
    with pytest.raises(ValueError):
        build_optimizer(m, cfg.training)
