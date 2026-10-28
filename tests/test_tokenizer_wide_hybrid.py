"""CPU/meta prerequisites only; no wider hybrid quality or MPS claim."""
import copy
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.attention.cache import build_caches
from src.models.gpt import GPT
from src.models.tokenizer_architecture import architecture_contract, validate_architecture_model
from src.models.tokenizer_hybrid import GEOMETRY, POLICY
from src.research.tokenizer_wide_hybrid_config import FAMILIES, config
from src.train.optimizers import build_optimizer
from src.utils.checkpoint import load_model_weights


@pytest.mark.parametrize('width', [512, 1024])
@pytest.mark.parametrize('family', FAMILIES)
def test_meta_geometry_optimizer_and_guards(tmp_path, width, family):
    cfg = OmegaConf.create(config(tmp_path, family, width))
    contract = architecture_contract(cfg.model)
    with torch.device('meta'):
        model = GPT(cfg.model)
        validate_architecture_model(model)
        opt = build_optimizer(model, cfg.training)
        caches = build_caches(cfg, 'meta', batch_size=8)
    expected_cache = sum(8*4*(41*width if f == 'kda' else 128*(width//4+32))
                         for f in contract['layout'])
    assert sum(b.numel()*b.element_size() for c in caches for b in c.buffers()) == expected_cache
    names = []
    for g in opt.param_groups:
        name, p = g['parameter_names'][0], g['params'][0]
        scale = 1/math.sqrt(width/64) if name.startswith('transformer.h.') and p.ndim == 2 else 1.
        assert g['lr'] == pytest.approx(cfg.training.learning_rate*scale)
        names.append(name)
    assert len(names) == len(set(names)) == len(list(model.parameters()))
    assert opt.decay == .00006
    bad = copy.deepcopy(cfg.model)
    bad.tokenizer_architecture_policy = POLICY
    bad.attention_width_policy = GEOMETRY
    with pytest.raises(ValueError):
        architecture_contract(bad)
    for k, v in [('dim', 256), ('dim', 768), ('dim', True), ('n_heads', 8),
                 ('rope', 1), ('g_min', -3.), ('dim_alpha', 16), ('dimckv', 128),
                 ('attention_width_policy', GEOMETRY), ('n_layers', 8)]:
        bad = copy.deepcopy(cfg.model)
        bad[k] = v
        with pytest.raises(ValueError):
            architecture_contract(bad)


def test_width512_equations_and_gradients(tmp_path, monkeypatch):
    # Reuse the independently written FP64 KDA matrix and gated-MLA references.
    import tests.test_tokenizer_hybrid as references
    monkeypatch.setattr(references, 'hybrid_config',
        lambda output, width: OmegaConf.create(config(output, 'hybrid', width)))
    references.test_hybrid_attention_equations_and_gradients(tmp_path, 512)


@pytest.mark.parametrize('family', FAMILIES)
def test_width512_caches_causality_and_lineage(tmp_path, family):
    torch.set_num_threads(2)
    torch.manual_seed(3481)
    cfg = OmegaConf.create(config(tmp_path, family))
    model = GPT(cfg.model).eval()
    for batch in (1, 8):
        ids = torch.arange(11).unsqueeze(0).repeat(batch, 1)+100
        caches = build_caches(cfg, 'cpu', batch_size=batch)
        other = build_caches(cfg, 'cpu', batch_size=batch)
        with torch.no_grad():
            full = model(ids)[0]
            parts = []
            for lo, hi in ((0, 3), (3, 7), (7, 11)):
                parts.append(model(ids[:, lo:hi], cache=caches, start_pos=lo)[0])
                model(ids[:, lo:hi]+10, cache=other, start_pos=lo)
            torch.testing.assert_close(torch.cat(parts, 1), full, rtol=2e-5, atol=1e-5)
            changed = ids.clone()
            changed[:, 7:] += 30
            torch.testing.assert_close(model(changed)[0][:, :7], full[:, :7], rtol=0, atol=0)
            for c in caches:
                c.reset()
            tokens = [model(ids[:, i:i+1], cache=caches, start_pos=i)[0] for i in range(11)]
            torch.testing.assert_close(torch.cat(tokens, 1), full, rtol=2e-5, atol=1e-5)
            with pytest.raises(ValueError):
                model(ids, cache=caches, start_pos=0)
            assert all(c.position == 11 for c in caches)
            wrong = build_caches(cfg, 'cpu', batch_size=batch+1)
            with pytest.raises(ValueError):
                model(ids, cache=wrong)
            assert all(c.position == 0 for c in wrong)
    payload = dict(cfg=OmegaConf.to_container(cfg), model_state=model.state_dict())
    load_model_weights(model, payload)
    payload['cfg']['model']['attention_layout'] = ['kda']*4 if family != 'kda' else ['gated_mla']*4
    with pytest.raises(ValueError):
        load_model_weights(model, payload)
    a = model.transformer.h[0].attn
    if family == 'gated_mla':
        a.dimc_kv += 1
    else:
        a.g_min = -4.
    with pytest.raises(ValueError):
        build_optimizer(model, cfg.training)


@pytest.mark.parametrize('family', FAMILIES)
def test_width512_real_trainer_in_memory_resume_and_ema(tmp_path, monkeypatch, family):
    # Share the full-vs-resumed model/optimizer/RNG/EMA hash check and real
    # trainer/serialization path, changing only the explicit family config.
    # This creates no retained checkpoint files; disk export still needs a pilot.
    import tests.test_tokenizer_wide as continuation
    monkeypatch.setattr(continuation, 'config', config)
    continuation.test_width512_real_trainer_in_memory_resume_and_ema(tmp_path, monkeypatch, family)
