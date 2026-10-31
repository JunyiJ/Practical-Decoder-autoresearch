"""Read-only export regression on retained synthetic fixtures, never trained weights."""
import gc
import json
from pathlib import Path
import pytest
import torch
from omegaconf import OmegaConf
from research.block_norm_v1.runtime.train.weight_averaging import derived_checkpoint_payload
from research.block_norm_v1.runtime.utils.checkpoint import read_checkpoint,load_model_weights,save_checkpoint
from research.block_norm_v1.runtime.models.gpt import GPT
from research.block_norm_v1.test_resume_replay import same
from src.research.protocol import file_digest
ROOT=Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('kind',['rms_incumbent','layer_matched_epsilon'])
def test_export_remains_exact_but_incomplete_resume_rejected(kind):
    manifest=json.loads((ROOT/'research/tokenizer-evidence/block-norm-preparation-001/checkpoint-transport-fixture-manifest.json').read_text())
    stem='research_state/block-norm-checkpoint-fixture-001/'+kind
    paths=[ROOT/(stem+suffix) for suffix in ['-raw.pt','-h200.pt']]
    for p in paths:assert file_digest(p)==manifest['files'][str(p.relative_to(ROOT))]['sha256']
    rng=torch.get_rng_state().clone()
    derived=derived_checkpoint_payload(paths[0],200)
    assert torch.equal(rng,torch.get_rng_state())
    saved=read_checkpoint(paths[1]);assert same(derived,saved)
    del derived,saved;gc.collect()
    raw=read_checkpoint(paths[0]);cfg=OmegaConf.create(raw['cfg'])
    with torch.random.fork_rng(devices=[]):model=GPT(cfg.model)
    # These preserved fixtures intentionally contain empty optimizer slots and
    # minimal averaging metadata; export remains valid, full resume is refused.
    with pytest.raises(ValueError):load_model_weights(model,raw,for_resume=True)
    load_model_weights(model,raw,for_resume=False)
    for p in paths:assert file_digest(p)==manifest['files'][str(p.relative_to(ROOT))]['sha256']


def test_legacy_resume_behavior_unchanged():
    from research.block_norm_v1.test_model_policy import config
    cfg=config('rms_incumbent');cfg.model.tokenizer_architecture_policy='tokenizer_mla_quarter_rope_v1'
    del cfg.model.block_normalization
    with torch.random.fork_rng(devices=[]):model=GPT(cfg.model)
    captured=[]
    save_checkpoint(Path('not-written.pt'),model,cfg=cfg,payload_writer=lambda path,payload:captured.append(payload))
    assert 'optimizer_state' not in captured[0]
    load_model_weights(model,captured[0],for_resume=True)  # legacy low-level semantics preserved
