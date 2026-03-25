from types import SimpleNamespace
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.gpt import GPT


def _build_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        dim=16,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        vocab_size=32,
        block_size=8,
        dropout=0.0,
        attn_dropout=0.0,
        attn_bias=True,
        causal=True,
        rope=0,
        norm_type="layernorm",
        attn_type="gqa",
        mlp_type="moe",
        moe=SimpleNamespace(
            hidden_dim=32,
            num_experts=4,
            shared_num_experts=0,
            top_k=2,
            aux_loss_weight=0.01,
            ffn="gelu",
        ),
    )


def test_eval_loss_can_exclude_moe_aux_term() -> None:
    torch.manual_seed(0)
    model = GPT(_build_cfg())
    model.eval()

    idx = torch.randint(0, model.cfg.vocab_size, (2, 6))
    targets = torch.randint(0, model.cfg.vocab_size, (2, 6))

    _, loss_with_aux, with_aux = model(
        idx,
        targets,
        include_aux_loss=True,
        return_loss_breakdown=True,
    )
    _, loss_without_aux, without_aux = model(
        idx,
        targets,
        include_aux_loss=False,
        return_loss_breakdown=True,
    )

    assert with_aux["aux_loss"] is not None
    torch.testing.assert_close(loss_without_aux, without_aux["ce_loss"], rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(without_aux["ce_loss"], with_aux["ce_loss"], rtol=1e-6, atol=1e-7)
    expected_total = with_aux["ce_loss"] + model.aux_loss_weight * with_aux["aux_loss"]
    torch.testing.assert_close(loss_with_aux, expected_total, rtol=1e-6, atol=1e-7)
