"""Rank-contract checks only; no continuation retry or quality trajectory."""
import copy
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.attention.cache import build_caches
from src.models.gpt import GPT
from src.models.tokenizer_architecture import architecture_contract, validate_architecture_model
from src.models.tokenizer_hybrid import RANK_POLICY
from src.research.tokenizer_wide_hybrid_config import FAMILIES, config, rank_config
from src.train.optimizers import build_optimizer
from src.utils.checkpoint import load_model_weights


@pytest.mark.parametrize('width', [512, 1024])
@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('divisor', [8, 4, 2])
def test_rank_meta_geometry_optimizer_and_independent_count(tmp_path, width, family, divisor):
    cfg = OmegaConf.create(rank_config(tmp_path, family, width, bottleneck_divisor=divisor))
    contract = architecture_contract(cfg.model)
    assert contract['latent_rank'] == contract['kda_alpha_rank'] == width//divisor
    with torch.device('meta'):
        model = GPT(cfg.model)
        anchor = GPT(OmegaConf.create(config(tmp_path, family, width)).model)
        validate_architecture_model(model)
        opt = build_optimizer(model, cfg.training)
        caches = build_caches(cfg, 'meta', batch_size=8)
    rank_delta = width//divisor - width//4
    # KDA has two D*r matrices; gated MLA has six D*r matrices and two rank biases.
    param_delta = sum(rank_delta*(2*width if f == 'kda' else 6*width+2) for f in contract['layout'])
    assert sum(p.numel() for p in model.parameters()) == sum(p.numel() for p in anchor.parameters())+param_delta
    expected = sum(8*4*(41*width if f == 'kda' else 128*(width//divisor+32)) for f in contract['layout'])
    assert sum(b.numel()*b.element_size() for c in caches for b in c.buffers()) == expected
    for g in opt.param_groups:
        name, p = g['parameter_names'][0], g['params'][0]
        scale = 1/math.sqrt(width/64) if name.startswith('transformer.h.') and p.ndim == 2 else 1.
        assert g['lr'] == pytest.approx(cfg.training.learning_rate*scale)
    if divisor == 4:
        assert {k:tuple(v.shape) for k,v in model.state_dict().items()} == {k:tuple(v.shape) for k,v in anchor.state_dict().items()}


@pytest.mark.parametrize('value', [True, 1, 3, 16, 4., '4', None])
def test_rank_guard_rejects_unsupported_values(tmp_path, value):
    with pytest.raises(ValueError):
        rank_config(tmp_path, 'hybrid', bottleneck_divisor=value)


def test_old_policies_still_reject_rank_overrides(tmp_path):
    cfg = config(tmp_path, 'hybrid')
    cfg['model']['attention_bottleneck_divisor'] = 4
    with pytest.raises(ValueError):
        architecture_contract(cfg['model'])
    cfg['model']['tokenizer_architecture_policy'] = RANK_POLICY
    with pytest.raises(ValueError):
        architecture_contract(cfg['model'])


@pytest.mark.parametrize('divisor', [8, 2])
def test_rank_equations_and_gradients(tmp_path, monkeypatch, divisor):
    import tests.test_tokenizer_hybrid as reference
    monkeypatch.setattr(reference, 'hybrid_config', lambda output, width:
        OmegaConf.create(rank_config(output, 'hybrid', width, bottleneck_divisor=divisor)))
    reference.test_hybrid_attention_equations_and_gradients(tmp_path, 512)


@pytest.mark.parametrize('family', FAMILIES)
@pytest.mark.parametrize('divisor', [8, 4, 2])
def test_rank_full_cached_causality_and_lineage(tmp_path, family, divisor):
    torch.set_num_threads(2)
    torch.manual_seed(3489)
    cfg = OmegaConf.create(rank_config(tmp_path, family, bottleneck_divisor=divisor))
    model = GPT(cfg.model).eval()
    for batch in (1, 8):
        ids = torch.arange(7).unsqueeze(0).repeat(batch, 1)+100
        caches = build_caches(cfg, 'cpu', batch_size=batch)
        with torch.no_grad():
            full = model(ids)[0]
            parts = [model(ids[:,lo:hi], cache=caches, start_pos=lo)[0] for lo,hi in ((0,2),(2,5),(5,7))]
            torch.testing.assert_close(torch.cat(parts,1), full, rtol=2e-5, atol=1e-5)
            changed = ids.clone(); changed[:,5:] += 20
            torch.testing.assert_close(model(changed)[0][:,:5], full[:,:5], rtol=0, atol=0)
            for c in caches: c.reset()
            tokens = [model(ids[:,i:i+1], cache=caches, start_pos=i)[0] for i in range(7)]
            torch.testing.assert_close(torch.cat(tokens,1), full, rtol=2e-5, atol=1e-5)
            other_cfg = copy.deepcopy(cfg)
            other_cfg.model.attention_bottleneck_divisor = 8 if divisor != 8 else 2
            wrong = build_caches(other_cfg, 'cpu', batch_size=batch)
            with pytest.raises(ValueError): model(ids, cache=wrong)
            assert all(c.position == 0 for c in wrong)
    payload = dict(cfg=OmegaConf.to_container(cfg), model_state=model.state_dict())
    load_model_weights(model, payload)
    payload['cfg']['model']['attention_bottleneck_divisor'] = 8 if divisor != 8 else 2
    with pytest.raises(ValueError): load_model_weights(model, payload)
