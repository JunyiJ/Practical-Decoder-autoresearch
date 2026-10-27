"""MLA tokenizer prerequisites; no quality measurements or holdout scoring."""
import copy
import math

import pytest
import torch
from omegaconf import OmegaConf

from src.attention.cache import KVCache
from src.models.gpt import GPT
from src.models.rope import apply_rotary_emb
from src.models.tokenizer_architecture import (
    MLA_POLICY,
    architecture_contract,
    validate_architecture_model,
)
from src.research.tokenizer_attention_study import config
from src.train.optimizers import build_optimizer
from src.train.train import run_training
from src.train.weight_averaging import export_averaged_checkpoint
from src.utils.checkpoint import load_model_weights, read_checkpoint


def mla_config(output, width=64, steps=4):
    cfg = config(output, width=width, steps=steps, device='cpu')
    cfg['model'].update(attn_type='mla', attention_width_policy='mla_quarter_head32',
                        tokenizer_architecture_policy=MLA_POLICY)
    return OmegaConf.create(cfg)


@pytest.mark.parametrize('width', [64, 256])
def test_mla_scaled_equation_and_all_projection_gradients(tmp_path, width):
    torch.set_num_threads(1)
    cfg = mla_config(tmp_path, width)
    model = GPT(cfg.model).double()
    block = model.transformer.h[0]
    att = block.attn
    x = torch.randn(2, 7, width, dtype=torch.float64, requires_grad=True)
    actual = att(x, freq_cos=block.freq_cos, freq_sin=block.freq_sin)[0]
    cq, ckv = att.dq(x), att.dkv(x)
    q = att.uq(cq).reshape(2, 7, width//32, 32).transpose(1, 2)
    k = att.uk(ckv).reshape(2, 7, width//32, 32).transpose(1, 2)
    v = att.uv(ckv).reshape(2, 7, width//32, 32).transpose(1, 2)
    qr = att.qr(cq).reshape(2, 7, width//32, 32).transpose(1, 2)
    kr = att.kr(x).unsqueeze(1)
    qr = apply_rotary_emb(qr, block.freq_cos[:7], block.freq_sin[:7])
    kr = apply_rotary_emb(kr, block.freq_cos[:7], block.freq_sin[:7])
    # Separate content/rotary contractions independently check per-head concatenation.
    score = (q@k.transpose(-1, -2) + qr@kr.transpose(-1, -2)) * (.5/math.sqrt(64))
    score = score.masked_fill(~torch.ones(7, 7, dtype=torch.bool).tril(), float('-inf'))
    expected = att.out((score.softmax(-1)@v).transpose(1, 2).reshape(2, 7, width))
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    targets = [x, *att.parameters()]
    ga = torch.autograd.grad(actual.square().sum(), targets, retain_graph=True)
    gb = torch.autograd.grad(expected.square().sum(), targets)
    for a, b in zip(ga, gb):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-10)
    # The historical unscaled default remains an exact identity, with no new tensors.
    old_cfg = copy.deepcopy(cfg.model)
    for key in ('tokenizer_architecture_policy', 'attention_logit_scale', 'attention_logit_scale_width_policy', 'output_bias_kind', 'output_bias_width_policy', 'output_bias_prior'):
        del old_cfg[key]
    from src.attention.mla import MLA
    old = MLA(old_cfg).double()
    old.load_state_dict(att.state_dict())
    assert old.attention_logit_scale == 1.
    att.attention_logit_scale = 1.
    torch.testing.assert_close(old(x, freq_cos=block.freq_cos, freq_sin=block.freq_sin)[0],
                               att(x, freq_cos=block.freq_cos, freq_sin=block.freq_sin)[0], rtol=0, atol=0)


@pytest.mark.parametrize('width', [64, 256])
@pytest.mark.parametrize('batch', [1, 2])
def test_mla_tokenizer_cached_offsets_causality_reset_memory_and_optimizer(tmp_path, width, batch):
    torch.set_num_threads(1)
    cfg = mla_config(tmp_path, width)
    model = GPT(cfg.model).eval()
    validate_architecture_model(model)
    optimizer = build_optimizer(model, cfg.training)
    assert optimizer.decay == .00006
    for group in optimizer.param_groups:
        name, p = group['parameter_names'][0], group['params'][0]
        scale = 1/math.sqrt(width/64) if name.startswith('transformer.h.') and p.ndim == 2 else 1.
        assert group['lr'] == pytest.approx(cfg.training.learning_rate*scale)
    ids = torch.arange(17).view(1, -1).expand(batch, -1)+100
    cache = [KVCache(cfg, 'cpu', batch_size=batch) for _ in range(4)]
    with torch.no_grad():
        full = model(ids)[0]
        parts = [model(ids[:, lo:hi], cache=cache, start_pos=lo)[0] for lo, hi in ((0, 3), (3, 8), (8, 17))]
        torch.testing.assert_close(torch.cat(parts, 1), full, rtol=1e-5, atol=1e-6)
        changed = ids.clone(); changed[:, 8:] += 20
        torch.testing.assert_close(model(changed)[0][:, :8], full[:, :8], rtol=0, atol=0)
        other = [KVCache(cfg, 'cpu', batch_size=batch) for _ in range(4)]
        model(changed[:, :5], cache=other)
        for c in cache:
            c.reset()
        torch.testing.assert_close(model(ids, cache=cache)[0], full, rtol=1e-5, atol=1e-6)
    assert sum(b.numel()*b.element_size() for c in cache for b in c.buffers()) == 4*batch*128*(width//4+32)*4


@pytest.mark.parametrize('key,value', [('dimckv', 16), ('dimcq', 16), ('head_dim_r', 32), ('rope', 0), ('attention_logit_scale', 1.), ('attn_type', 'gqa'), ('attention_width_policy', 'gqa_half_head32')])
def test_mla_contract_rejects_unreviewed_geometry(tmp_path, key, value):
    cfg = mla_config(tmp_path)
    cfg.model[key] = value
    with pytest.raises(ValueError):
        architecture_contract(cfg.model)


def test_mla_mutation_and_checkpoint_identity(tmp_path):
    cfg = mla_config(tmp_path)
    model = GPT(cfg.model)
    payload = {'cfg': OmegaConf.to_container(cfg), 'model_state': model.state_dict()}
    payload['cfg']['model']['attention_logit_scale'] = 1.
    with pytest.raises(ValueError):
        load_model_weights(model, payload)
    model.transformer.h[0].attn.dimc_kv += 1
    with pytest.raises(ValueError):
        build_optimizer(model, cfg.training)


def test_mla_real_training_resume_ema_tokenizer(tmp_path):
    torch.set_num_threads(1)
    cfg = mla_config(tmp_path/'full')
    full = run_training(cfg)
    a = read_checkpoint(full['final_checkpoint'])
    cfg.training.checkpoint_dir = str(tmp_path/'partial')
    cfg.training.max_train_seconds = 1e-9
    partial = run_training(cfg)
    assert partial['completed_steps'] == 1
    cfg.training.max_train_seconds = None
    cfg.training.resume_from = partial['final_checkpoint']
    resumed = run_training(cfg)
    b = read_checkpoint(resumed['final_checkpoint'])
    for key, value in a['model_state'].items():
        torch.testing.assert_close(value, b['model_state'][key], rtol=0, atol=0)
    avg = export_averaged_checkpoint(resumed['final_checkpoint'], tmp_path/'ema.pt', 200)
    payload = read_checkpoint(avg['checkpoint'])
    clone = GPT(OmegaConf.create(payload['cfg']['model']))
    load_model_weights(clone, payload)
    assert payload['tokenizer_contract']['name'] == 'gpt2'
