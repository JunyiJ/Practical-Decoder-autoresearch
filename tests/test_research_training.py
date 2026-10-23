import pytest
import torch
from omegaconf import OmegaConf

from src.data.loaders import TinyDataLoader, TextDataset
from src.train.train import _evaluate, evaluate_checkpoint, run_training
from src.utils.checkpoint import read_checkpoint


@pytest.fixture
def research_cfg(tmp_path):
    corpus = tmp_path / 'corpus.txt'
    corpus.write_text('abcdefghij' * 100)
    return OmegaConf.create({
        'data': {'path': str(corpus), 'train_fraction': .8, 'val_fraction': .1},
        'model': {'dim': 8, 'n_heads': 2, 'n_layers': 1, 'vocab_size': 10,
                  'block_size': 8, 'dropout': .1, 'rope': 1, 'norm_type': 'rmsnorm',
                  'attn_type': 'mha', 'mlp_type': 'dense'},
        'training': {'device': 'cpu', 'strict_device': True, 'seed': 42, 'eval_seed': 123,
                     'batch_size': 2, 'learning_rate': .001, 'weight_decay': .1,
                     'grad_clip': 1., 'max_iters': 2, 'eval_every': 1, 'eval_iters': 2,
                     'checkpoint_dir': str(tmp_path / 'checkpoints'),
                     'checkpoint_every': 0, 'keep_best_checkpoint': False},
    })


def test_splits_do_not_share_positions_or_cross_boundaries(tmp_path):
    corpus = tmp_path / 'unique.txt'
    corpus.write_text(''.join(chr(0x100 + i) for i in range(100)))
    loader = TinyDataLoader(corpus, 16, 4, 'cpu', train_fraction=.6, val_fraction=.2)
    for split, lower, upper in [('train', 0, 60), ('val', 60, 80), ('test', 80, 100)]:
        assert loader.splits[split].tolist() == list(range(lower, upper))
        x, y = loader.get_batch(split)
        assert int(x.min()) >= lower
        assert int(y.max()) < upper
        assert torch.equal(x[:, 1:], y[:, :-1])


@pytest.mark.parametrize('fractions', [(0, .1), (.9, 0), (.8, .3), (float('nan'), .1)])
def test_invalid_split_fractions_rejected(tmp_path, fractions):
    corpus = tmp_path / 'text.txt'
    corpus.write_text('abc' * 100)
    with pytest.raises(ValueError, match='fraction'):
        TinyDataLoader(corpus, 2, 4, 'cpu', train_fraction=fractions[0], val_fraction=fractions[1])


def test_legacy_split_has_no_test_and_preserves_rounding(tmp_path):
    corpus = tmp_path / 'text.txt'
    corpus.write_text('a' * 101)
    loader = TinyDataLoader(corpus, 2, 4, 'cpu')
    assert len(loader.train_data) == 90
    assert len(loader.val_data) == 11
    with pytest.raises(ValueError, match='split'):
        loader.get_batch('test')


def test_eval_seed_shares_windows_between_training_seeds(tmp_path):
    corpus = tmp_path / 'unique.txt'
    corpus.write_text(''.join(chr(0x100 + i) for i in range(200)))
    a = TinyDataLoader(corpus, 4, 4, 'cpu', seed=1, eval_seed=987,
                       train_fraction=.8, val_fraction=.1)
    b = TinyDataLoader(corpus, 4, 4, 'cpu', seed=2, eval_seed=987,
                       train_fraction=.8, val_fraction=.1)
    for split in ('val', 'test'):
        assert torch.equal(a.get_batch(split)[0], b.get_batch(split)[0])
    assert not torch.equal(a.get_batch('train')[0], b.get_batch('train')[0])


@pytest.mark.parametrize('device', ['mps', 'cuda'])
def test_strict_device_rejects_unavailable_accelerator(research_cfg, monkeypatch, device):
    backend = torch.backends.mps if device == 'mps' else torch.cuda
    monkeypatch.setattr(backend, 'is_available', lambda: False)
    research_cfg.training.device = device
    with pytest.raises(RuntimeError, match='strict_device'):
        run_training(research_cfg)


def test_training_never_reads_test_and_frozen_checkpoint_can_be_scored(research_cfg, monkeypatch):
    original = TinyDataLoader.get_batch
    seen = []
    def record_batch(loader, split='train'):
        seen.append(split)
        return original(loader, split)
    monkeypatch.setattr(TinyDataLoader, 'get_batch', record_batch)
    result = run_training(research_cfg)
    assert set(seen) == {'train', 'val'}
    assert result['best_metric'] is not None
    assert result['best_checkpoint'] is None
    from pathlib import Path
    assert len(list(Path(research_cfg.training.checkpoint_dir).glob('*.pt'))) == 1
    validation = evaluate_checkpoint(research_cfg, result['final_checkpoint'], 'val')
    assert validation['loss'] == result['final_metric']['val_loss']
    heldout = evaluate_checkpoint(research_cfg, result['final_checkpoint'])
    repeated = evaluate_checkpoint(research_cfg, result['final_checkpoint'])
    assert heldout == repeated
    assert heldout['split'] == 'test'
    assert heldout['eval_seed'] == 124
    assert heldout['tokens_evaluated'] == 32
    assert heldout['checkpoint_step'] == 2
    contract = read_checkpoint(result['final_checkpoint'])['training_state']['contract']
    assert contract['data_split'] == {'train_fraction': .8, 'val_fraction': .1}
    assert contract['eval_seed'] == 123


def test_checkpoint_eval_rejects_data_or_protocol_changes(research_cfg):
    result = run_training(research_cfg)
    research_cfg.training.eval_seed += 1
    with pytest.raises(ValueError, match='matching'):
        evaluate_checkpoint(research_cfg, result['final_checkpoint'])
    research_cfg.training.eval_seed -= 1
    research_cfg.data.train_fraction = .7
    with pytest.raises(ValueError, match='matching'):
        evaluate_checkpoint(research_cfg, result['final_checkpoint'])
    research_cfg.data.train_fraction = .8
    from pathlib import Path
    Path(research_cfg.data.path).write_text('abcdefghij' * 101)
    with pytest.raises(ValueError, match='matching'):
        evaluate_checkpoint(research_cfg, result['final_checkpoint'])


def test_nonfinite_evaluation_is_rejected(tmp_path):
    corpus = tmp_path / 'text.txt'
    corpus.write_text('a' * 100)
    loader = TinyDataLoader(corpus, 1, 2, 'cpu')
    class NonfiniteModel:
        def eval(self):
            return self
        def __call__(self, *args, **kwargs):
            return None, torch.tensor(float('nan')), {'aux_loss': None}
    with pytest.raises(FloatingPointError, match='Nonfinite val'):
        _evaluate(NonfiniteModel(), loader, 'val', 1)


def test_tokenized_loader_uses_the_same_split_protocol(tmp_path, monkeypatch):
    class FakeEncoding:
        n_vocab = 100
        def encode(self, text):
            return list(range(len(text)))
    monkeypatch.setattr('src.data.loaders.tiktoken.get_encoding', lambda _: FakeEncoding())
    corpus = tmp_path / 'text.txt'
    corpus.write_text('a' * 100)
    loader = TextDataset(corpus, 2, 4, 'cpu', train_fraction=.6, val_fraction=.2)
    assert [len(loader.splits[k]) for k in ('train', 'val', 'test')] == [60, 20, 20]
    assert int(loader.get_batch('test')[0].min()) >= 80


def test_legacy_contract_accepts_only_original_split_and_eval_seed():
    from src.train.train import _contracts_match
    old = {'model': {'dim': 8}, 'training': {'seed': 42}, 'data_sha256': 'fixture'}
    expected = {**old, 'data_split': {'train_fraction': .9, 'val_fraction': .1}, 'eval_seed': 43}
    assert _contracts_match(old, expected)
    assert not _contracts_match(old, {**expected, 'eval_seed': 123})
    assert not _contracts_match(old, {**expected, 'data_split': {'train_fraction': .8, 'val_fraction': .1}})


def test_nonstrict_cuda_does_not_silently_fall_back(research_cfg, monkeypatch):
    from src.train.train import _resolve_device
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    research_cfg.training.device = 'cuda'
    research_cfg.training.strict_device = False
    assert _resolve_device(research_cfg) == 'cuda'
