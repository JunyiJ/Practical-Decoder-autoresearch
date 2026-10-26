"""Actual KDA trainer gates, deferred while the decay study is live."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from src.attention.cache import KVCache
from src.models.gpt import GPT
from src.research.averaging_eval import validate_derivation
from src.research.protocol import file_digest
from src.train.train import run_training
from src.train.weight_averaging import export_averaged_checkpoint
from src.utils.checkpoint import read_checkpoint, load_model_weights
from tests.test_kda_scaling import config
from tests.test_schedule_training import equal_nested


@pytest.mark.parametrize('width', [64, 256])
def test_kda_actual_optimizer_resume_average_and_cache(tmp_path, width):
    torch.set_num_threads(1)
    corpus = tmp_path/'synthetic.txt'
    corpus.write_text('abcdefghijklmno\n'*1024)
    cfg = OmegaConf.create(dict(
        data=dict(path=str(corpus), train_fraction=.8, val_fraction=.1),
        model=vars(config(width)),
        training=dict(batch_layout='paired_blocks512', batch_size=8,
            cpu_threads=1, device='cpu', strict_device=True, seed=9361,
            eval_seed=2026, max_iters=8, eval_every=4, eval_iters=2,
            checkpoint_every=4, checkpoint_dir=str(tmp_path/'full'),
            keep_best_checkpoint=False, optimizer='muon', decay_mode='independent',
            learning_rate=.005, weight_decay=.00006, grad_clip=1.,
            lr_schedule=dict(kind='linear_cooldown', total_steps=8, start_step=4, min_ratio=.1),
            weight_averaging=dict(kind='spaced_tail_ema', total_steps=8,
                                 start_step=4, interval=2, half_lives=[200]))))
    result = run_training(copy.deepcopy(cfg))
    parent = Path(result['final_checkpoint']); saved = read_checkpoint(parent)
    assert saved['step'] == 8 and result['metrics'][-1]['tokens_processed'] == 8192
    assert 'dim_alpha' not in saved['cfg']['model']
    assert saved['optimizer_state']['independent_decay'] == .00006
    assert all(torch.isfinite(t).all() for t in saved['model_state'].values())
    model = GPT(OmegaConf.create(saved['cfg']['model']))
    load_model_weights(model, saved)
    assert all(block.attn.dim_alpha == width//4 for block in model.transformer.h)
    names = dict(model.named_parameters()); seen = set()
    for opt in saved['optimizer_state']['optimizers']:
        for group in opt['param_groups']:
            name, = group['parameter_names']; seen.add(name)
            slot = opt['state'][group['params'][0]]
            matrix = name.startswith('transformer.h.') and names[name].ndim == 2
            scale = 1/math.sqrt(width/64) if matrix else 1.
            assert group['width_lr_scale'] == scale
            assert group['lr'] == .005*.1*scale and group['weight_decay'] == 0.
            if matrix:
                assert set(slot) == {'momentum_buffer'}
                assert torch.count_nonzero(slot['momentum_buffer']) > 0
            else:
                assert set(slot) == {'step', 'exp_avg', 'exp_avg_sq'}
                assert slot['step'].item() == 8
                assert torch.count_nonzero(slot['exp_avg_sq']) > 0
            for key, value in slot.items():
                assert torch.isfinite(value).all()
                if key != 'step':
                    assert value.shape == names[name].shape
    assert seen == set(names)
    averages = saved['training_state']['weight_averaging_state']
    assert averages['snapshots'] == 3 and set(averages['averages']) == {'200'}
    assert set(averages['parameter_names']) == set(names)
    plain_cfg = copy.deepcopy(cfg)
    del plain_cfg.training.weight_averaging
    plain_cfg.training.checkpoint_dir = str(tmp_path/'plain')
    plain = read_checkpoint(run_training(plain_cfg)['final_checkpoint'])
    for key in ('model_state', 'optimizer_state'):
        equal_nested(saved[key], plain[key])
    for key in ('loader_rng', 'torch_rng'):
        equal_nested(saved['training_state'][key], plain['training_state'][key])
    middle, = (tmp_path/'full').glob('step_000004_*.pt')
    resumed_cfg = copy.deepcopy(cfg)
    resumed_cfg.training.resume_from = str(middle)
    resumed_cfg.training.checkpoint_dir = str(tmp_path/'resume')
    restored = read_checkpoint(run_training(resumed_cfg)['final_checkpoint'])
    for key in ('model_state', 'optimizer_state'):
        equal_nested(saved[key], restored[key])
    for key in ('loader_rng', 'torch_rng', 'weight_averaging_state'):
        equal_nested(saved['training_state'][key], restored['training_state'][key])
    wrong = copy.deepcopy(resumed_cfg); wrong.model.dim_alpha = width//8
    with pytest.raises(ValueError, match='Resume requires matching|KDA'):
        run_training(wrong)
    gen = torch.Generator().manual_seed(9361)
    for _ in range(8):
        torch.randint(int(.8*len(corpus.read_text()))-512, (2,), generator=gen)
    assert torch.equal(saved['training_state']['loader_rng']['train'], gen.get_state())
    assert torch.equal(saved['training_state']['loader_rng']['test'],
                       torch.Generator().manual_seed(2027).get_state())
    parent_sha = file_digest(parent); derived_path = tmp_path/'H200.pt'
    export_averaged_checkpoint(parent, derived_path, 200)
    derived = validate_derivation(dict(checkpoint=str(derived_path),
        checkpoint_sha256=file_digest(derived_path), parent_checkpoint=str(parent),
        parent_sha256=parent_sha, half_life=200, completed_steps=8))
    assert file_digest(parent) == parent_sha
    for name, value in derived['model_state'].items():
        target = averages['averages']['200'][name] if name in names else saved['model_state'][name]
        assert torch.equal(value, target)
    load_model_weights(model, derived)
    with pytest.raises(ValueError, match='cannot resume'):
        load_model_weights(model, derived, for_resume=True)
    model.eval()
    caches = [KVCache(SimpleNamespace(model=model.cfg), 'cpu', batch_size=2)
              for _ in range(4)]
    tokens = torch.arange(16).reshape(2, 8) % model.cfg.vocab_size
    with torch.no_grad():
        full = model(tokens)[0]
        chunked = torch.cat((model(tokens[:, :3], cache=caches)[0],
                             model(tokens[:, 3:], cache=caches, start_pos=3)[0]), dim=1)
    torch.testing.assert_close(chunked, full, atol=1e-4, rtol=2e-5)
