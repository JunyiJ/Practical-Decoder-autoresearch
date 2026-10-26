"""Actual optimizer algebra gates; execute only after active research exits."""
import copy
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.train.optimizers import HybridOptimizer, build_optimizer
from src.research.decay_quality_study import DECAYS, suite_spec
from tests.test_decay_quality_study import plan


@pytest.mark.parametrize('family', list(DECAYS))
@pytest.mark.parametrize('width', [64, 256])
def test_zero_lr_nominal_decay_all_roles_once_and_absent_gradient_unchanged(family, width):
    torch.set_num_threads(1)
    cfg = OmegaConf.create(suite_spec(plan(common_norm='layer'), family, .005)['candidate']['config'])
    cfg.model.update(dim=width, n_heads=width//32, vocab_size=17)
    cfg.training.learning_rate = 0.
    model = GPT(cfg.model)
    optimizer = build_optimizer(model, cfg.training)
    assert isinstance(optimizer, HybridOptimizer)
    before = {}
    with torch.no_grad():
        for i, (name, p) in enumerate(model.named_parameters()):
            p.fill_(.2 + .01*i)
            before[name] = p.double().clone()
            p.grad = None if name == 'transformer.h.0.mlp.hidden_norm.bias' else torch.zeros_like(p)
    seen = []
    for child in optimizer.optimizers:
        for group in child.param_groups:
            assert group['weight_decay'] == 0.
            seen.extend(group['parameter_names'])
    assert len(seen) == len(set(seen)) == len(before)
    for _ in range(5):
        optimizer.step()
    factor = (1-DECAYS[family])**5
    for name, p in model.named_parameters():
        expected = before[name] * (1. if p.grad is None else factor)
        torch.testing.assert_close(p.double(), expected, rtol=8e-7, atol=1e-8)
    state = copy.deepcopy(optimizer.state_dict())
    assert state['independent_decay'] == DECAYS[family]
    optimizer.load_state_dict(state)
    assert all(a is b for a,b in zip(optimizer.param_groups, [g for child in optimizer.optimizers for g in child.param_groups]))
    wrong = copy.deepcopy(state); wrong['independent_decay'] = 0.
    with pytest.raises(ValueError, match='independent decay changed'):
        optimizer.load_state_dict(wrong)


def test_decay_precedes_updates_with_frozen_known_gradient_and_resume():
    # Analytical SGD recurrence is an independent oracle for wrapper order.
    # It does not validate Muon's internal transformation or claim quality.
    for delta in DECAYS.values():
        parameter = torch.nn.Parameter(torch.tensor([1.3, -.7], dtype=torch.float64))
        child = torch.optim.SGD([parameter], lr=.13, momentum=0., weight_decay=0.)
        optimizer = HybridOptimizer([child], delta)
        expected = [1.3, -.7]
        for index in range(5):
            grad = [.31*(index+1), -.17*(index+1)]
            parameter.grad = torch.tensor(grad, dtype=torch.float64)
            optimizer.step()
            expected = [(1-delta)*p-.13*g for p,g in zip(expected, grad)]
            torch.testing.assert_close(parameter, torch.tensor(expected, dtype=torch.float64), rtol=1e-14, atol=1e-14)
            if index == 2:
                optimizer.load_state_dict(copy.deepcopy(optimizer.state_dict()))
        # Independent coefficient remains unchanged when an LR scheduler
        # changes the exposed group, including after state restoration.
        optimizer.param_groups[0]['lr'] = .013
        parameter.grad = torch.tensor([.2,.4], dtype=torch.float64)
        optimizer.step()
        expected = [(1-delta)*p-.013*g for p,g in zip(expected,[.2,.4])]
        torch.testing.assert_close(parameter, torch.tensor(expected, dtype=torch.float64), rtol=1e-14, atol=1e-14)
        assert math.isclose(optimizer.decay, delta, rel_tol=0, abs_tol=0)
