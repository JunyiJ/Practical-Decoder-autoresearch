"""NoPE layerwise tokenizer contracts; synthetic checks, not quality evidence."""
import copy
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.attention.cache import KVCache, build_caches
from src.models.gpt import GPT
from src.models.tokenizer_architecture import (
    architecture_contract,
    validate_architecture_model,
)
from src.models.tokenizer_hybrid import GEOMETRY, LAYOUTS, POLICY
from src.research.tokenizer_attention_study import config
from src.train.optimizers import build_optimizer
from src.train.train import run_training
from src.train.weight_averaging import export_averaged_checkpoint
from src.utils.checkpoint import load_model_weights, read_checkpoint
from tests.test_kda import matrix_reference


def hybrid_config(output, width=64, layout=LAYOUTS[0], steps=4):
    cfg = config(output, width=width, steps=steps, device='cpu')
    cfg['model'].update(attn_type='hybrid', tokenizer_architecture_policy=POLICY,
                        attention_width_policy=GEOMETRY, depth_width_policy='depth4_hybrid_unscaled',
                        rope=0, attention_layout=list(layout))
    return OmegaConf.create(cfg)


@pytest.mark.parametrize('width', [64, 256])
def test_hybrid_attention_equations_and_gradients(tmp_path, width):
    torch.set_num_threads(1)
    cfg = hybrid_config(tmp_path, width)
    model = GPT(cfg.model).double()
    x = torch.randn(2, 5, width, dtype=torch.float64, requires_grad=True)
    kda = model.transformer.h[0].attn
    actual = kda(x)[0]
    expected = matrix_reference(kda, x)[0]
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
    inputs = [x, *kda.parameters()]
    ga = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), inputs)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-10)
    a = model.transformer.h[3].attn
    actual = a(x)[0]
    cq, ckv = a.dq(x), a.dkv(x)
    q = a.uq(cq).reshape(2, 5, width//32, 32).transpose(1, 2)
    k = a.uk(ckv).reshape(2, 5, width//32, 32).transpose(1, 2)
    v = a.uv(ckv).reshape(2, 5, width//32, 32).transpose(1, 2)
    qr = a.qr(cq).reshape(2, 5, width//32, 32).transpose(1, 2)
    kr = a.kr(x).unsqueeze(1)
    score = (q@k.transpose(-1, -2)+qr@kr.transpose(-1, -2)) * (.5/math.sqrt(64))
    score = score.masked_fill(~torch.ones(5, 5, dtype=torch.bool).tril(), float('-inf'))
    y = (score.softmax(-1)@v).transpose(1, 2).reshape(2, 5, width)
    expected = a.out(torch.sigmoid(a.gate(x))*y)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = [x, *a.parameters()]
    ga = torch.autograd.grad(actual.square().sum(), inputs, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), inputs)
    for av, bv in zip(ga, gb):
        assert torch.isfinite(av).all()
        torch.testing.assert_close(av, bv, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize('width', [64, 256])
@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_nope_layout_cache_causality_reset_and_optimizer(tmp_path, width, layout, dtype):
    torch.set_num_threads(1)
    torch.manual_seed(3101)
    cfg = hybrid_config(tmp_path, width, layout)
    model = GPT(cfg.model).to(dtype=dtype).eval()
    validate_architecture_model(model)
    assert model.position_encoding == 'none' and 'wpe' not in model.transformer
    assert all(not b.enable_rope for b in model.transformer.h)
    optimizer = build_optimizer(model, cfg.training)
    names = []
    for group in optimizer.param_groups:
        name, p = group['parameter_names'][0], group['params'][0]
        scale = 1/math.sqrt(width/64) if name.startswith('transformer.h.') and p.ndim == 2 else 1.
        assert group['lr'] == pytest.approx(cfg.training.learning_rate*scale)
        names.append(name)
    assert len(names) == len(set(names)) == len(list(model.parameters()))
    assert optimizer.decay == .00006
    ids = torch.arange(13).unsqueeze(0).repeat(2, 1)+100
    caches = build_caches(cfg, 'cpu', batch_size=2, dtype=dtype)
    other = build_caches(cfg, 'cpu', batch_size=2, dtype=dtype)
    # FP64 discriminates algebra/state mistakes from FP32 projection rounding.
    rtol, atol = (1e-10, 1e-10) if dtype == torch.float64 else (2e-5, 1e-5)
    with torch.no_grad():
        full = model(ids)[0]
        parts = []
        for lo, hi in ((0, 3), (3, 7), (7, 13)):
            parts.append(model(ids[:, lo:hi], cache=caches, start_pos=lo)[0])
            model(ids[:, lo:hi]+10, cache=other, start_pos=lo)
        torch.testing.assert_close(torch.cat(parts, 1), full, rtol=rtol, atol=atol)
        changed = ids.clone(); changed[:, 7:] += 30
        torch.testing.assert_close(model(changed)[0][:, :7], full[:, :7], rtol=0, atol=0)
        for c in caches:
            c.reset()
        tokens = [model(ids[:, i:i+1], cache=caches, start_pos=i)[0] for i in range(13)]
        torch.testing.assert_close(torch.cat(tokens, 1), full, rtol=rtol, atol=atol)
    size = torch.empty((), dtype=dtype).element_size()
    expected = sum(2*size*(41*width if f == 'kda' else 128*(width//4+32)) for f in layout)
    assert sum(b.numel()*b.element_size() for c in caches for b in c.buffers()) == expected
    assert [c.position for c in caches] == [13]*4


def test_invalid_caches_rejected_before_any_state_mutation(tmp_path):
    torch.set_num_threads(1)
    cfg = hybrid_config(tmp_path)
    model = GPT(cfg.model).eval()
    caches = build_caches(cfg, 'cpu', batch_size=1)
    ids = torch.tensor([[100, 101]])
    with pytest.raises(ValueError, match='layer index'):
        KVCache(cfg, 'cpu', batch_size=1)
    with torch.no_grad():
        for wrong in ([caches[1], caches[0], *caches[2:]], build_caches(cfg, 'cpu', batch_size=2)):
            with pytest.raises(ValueError):
                model(ids, cache=wrong)
            assert all(c.position == 0 for c in wrong)
        with pytest.raises(ValueError):
            model(ids, cache=caches, start_pos=1)
        assert all(c.position == 0 for c in caches)
        model(ids, cache=caches)
        with pytest.raises(ValueError):
            model(ids, cache=caches, start_pos=0)
    with pytest.raises(ValueError, match='gradients disabled'):
        model(ids, cache=build_caches(cfg, 'cpu', batch_size=1))


@pytest.mark.parametrize('key,value', [('rope', 1), ('enable_rope', False), ('dim_alpha', 16),
    ('g_min', -3.), ('dimckv', 16), ('attention_logit_scale', 1.), ('attn_type', 'gated_mla'),
    ('attention_layout', ['gated_mla', 'kda', 'kda', 'kda']), ('attention_layout', 'kkkm')])
def test_hybrid_exact_contract(tmp_path, key, value):
    cfg = hybrid_config(tmp_path); cfg.model[key] = value
    with pytest.raises((ValueError, TypeError)):
        architecture_contract(cfg.model)


def test_hybrid_mutated_model_and_checkpoint(tmp_path):
    cfg = hybrid_config(tmp_path)
    model = GPT(cfg.model)
    payload = {'cfg': OmegaConf.to_container(cfg), 'model_state': model.state_dict()}
    payload['cfg']['model']['attention_layout'] = list(LAYOUTS[1])
    with pytest.raises(ValueError):
        load_model_weights(model, payload)
    model.transformer.h[0].attn.g_min = -4.
    with pytest.raises(ValueError):
        build_optimizer(model, cfg.training)


def test_hybrid_real_resume_ema_and_tokenizer(tmp_path):
    torch.set_num_threads(1)
    cfg = hybrid_config(tmp_path/'full', steps=3)
    # Synthetic CPU prerequisite with the real GPT-2 training split, never test scoring.
    full = run_training(cfg)
    expected = read_checkpoint(full['final_checkpoint'])
    cfg.training.checkpoint_dir = str(tmp_path/'partial')
    cfg.training.max_train_seconds = 1e-9
    partial = run_training(cfg)
    assert partial['completed_steps'] == 1
    cfg.training.max_train_seconds = None
    cfg.training.resume_from = partial['final_checkpoint']
    resumed = run_training(cfg)
    actual = read_checkpoint(resumed['final_checkpoint'])
    for key, value in expected['model_state'].items():
        torch.testing.assert_close(actual['model_state'][key], value, rtol=0, atol=0)
    for split in ('train', 'val', 'test'):
        assert torch.equal(expected['training_state']['loader_rng'][split], actual['training_state']['loader_rng'][split])
    avg = export_averaged_checkpoint(resumed['final_checkpoint'], tmp_path/'ema.pt', 200)
    payload = read_checkpoint(avg['checkpoint'])
    clone = GPT(OmegaConf.create(payload['cfg']['model']))
    load_model_weights(clone, payload)
    assert payload['tokenizer_contract']['name'] == 'gpt2'
    # Same-shape semantic changes must fail even before resuming any optimizer step.
    changed = copy.deepcopy(cfg); changed.model.attention_layout = list(LAYOUTS[2])
    with pytest.raises(ValueError):
        run_training(changed)
