"""Bounded synthetic state integration; no corpus training or disk checkpoints.

Real torch serialization is held in RAM. Only file IO is adapted; production
checkpoint/EMA/model/optimizer/schedule functions execute unchanged. Filesystem
transport remains a separate prerequisite, never inferred from these checks.
"""
import gc
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch

from src.attention.cache import build_caches
from src.models.gpt import GPT
from src.train.lr_schedule import LearningRateSchedule
from src.train.optimizers import build_optimizer
from src.train.weight_averaging import TailWeightAverages, export_averaged_checkpoint
from src.utils import checkpoint as ck
from tests.test_tokenizer_activation import config


def fingerprint(value):
    h = hashlib.sha256()

    def visit(x):
        if isinstance(x, torch.Tensor):
            h.update(str((tuple(x.shape), x.dtype)).encode())
            h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=repr):
                visit(key); visit(x[key])
        elif isinstance(x, (tuple, list)):
            h.update(str(type(x)).encode())
            for item in x:
                visit(item)
        else:
            h.update(repr(x).encode())
    visit(value)
    return h.hexdigest()


class MemoryCheckpoints:
    def __init__(self, monkeypatch):
        self.blobs = {}
        self.original_read = ck.read_checkpoint
        original_open, original_exists = Path.open, Path.exists

        def opened(path, mode='r', *args, **kwargs):
            if path in self.blobs:
                assert mode == 'rb'
                return io.BytesIO(self.blobs[path])
            return original_open(path, mode, *args, **kwargs)

        def exists(path):
            return path in self.blobs or original_exists(path)

        monkeypatch.setattr(Path, 'open', opened)
        monkeypatch.setattr(Path, 'exists', exists)
        monkeypatch.setattr(ck, 'read_checkpoint', self.read)

    def write(self, path, payload):
        path = Path(path).resolve()
        assert path not in self.blobs
        with io.BytesIO() as stream:
            torch.save(payload, stream)
            self.blobs[path] = stream.getvalue()

    def read(self, path, map_location='cpu'):
        key = Path(path).resolve()
        if key not in self.blobs:
            return self.original_read(path, map_location=map_location)
        return self.original_read(io.BytesIO(self.blobs[key]), map_location=map_location)


def objects(cfg):
    model = GPT(cfg.model)
    opt = build_optimizer(model, cfg.training)
    schedule = LearningRateSchedule(opt, cfg.training)
    average = TailWeightAverages(model, cfg.training)
    loader = SimpleNamespace(generators={'train': torch.Generator().manual_seed(7351)})
    return model, opt, schedule, average, loader


def update(model, opt, schedule, average, loader, step):
    # Two independent RNG streams make restoring both global and loader RNG
    # necessary. These random IDs never sample or score a corpus split.
    ids = torch.randint(0, 50257, (1, 3), generator=loader.generators['train'])
    targets = torch.randint(0, 50257, (1, 3))
    schedule.apply(step)
    opt.zero_grad()
    _, loss, parts = model(ids, targets, return_loss_breakdown=True)
    assert parts['aux_loss'] is None
    torch.testing.assert_close(loss, parts['ce_loss'], rtol=0, atol=0)
    assert torch.isfinite(loss)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    opt.step()
    average.update(step + 1)
    return float(loss.detach())


def save(store, path, cfg, model, opt, average, loader, step):
    state = ck.capture_training_state(loader, 'cpu',
        contract={'weight_averaging': average.spec,
                  'tokenizer': OmegaConf.to_container(cfg.model.output_bias_prior.tokenizer)},
        weight_averaging_state=average.state_dict())
    ck.save_checkpoint(path, model, opt, step, cfg, state, payload_writer=store.write)


@pytest.mark.parametrize('kind', ['silu', 'gelu_exact'])
def test_serialized_resume_rng_schedule_h200_and_tokenizer(monkeypatch, kind):
    torch.set_num_threads(1)
    cfg = config(kind)
    cfg.training.max_iters = 4
    cfg.training.lr_schedule.update(total_steps=4, start_step=2)
    cfg.training.weight_averaging.update(total_steps=4, start_step=2, interval=1)
    store = MemoryCheckpoints(monkeypatch)
    root = (Path('/tmp/tokenizer-activation-ram-only') / kind).resolve()
    raw_path, final_path, avg_path = (root / n for n in ('step2.pt', 'raw.pt', 'h200.pt'))
    assert not root.exists()  # No file or directory is created by this test.
    torch.manual_seed(7350)
    model, opt, schedule, average, loader = objects(cfg)
    for step in range(2):
        update(model, opt, schedule, average, loader, step)
    save(store, raw_path, cfg, model, opt, average, loader, 2)
    expected_losses = [update(model, opt, schedule, average, loader, step) for step in (2, 3)]
    expected = dict(model=fingerprint(model.state_dict()), optimizer=fingerprint(opt.state_dict()),
                    average=fingerprint(average.state_dict()),
                    rng=fingerprint(ck.capture_training_state(loader, 'cpu')))
    del model, opt, schedule, average, loader
    gc.collect()

    model, opt, schedule, average, loader = objects(cfg)
    raw = ck.load_checkpoint(raw_path, model, opt)
    average.load_state_dict(raw['training_state']['weight_averaging_state'], 2)
    ck.restore_training_state(raw['training_state'], loader, 'cpu')
    del raw
    # Only volatile serialized bytes are released, never a filesystem artifact.
    del store.blobs[raw_path]
    actual_losses = [update(model, opt, schedule, average, loader, step) for step in (2, 3)]
    assert actual_losses == expected_losses
    assert fingerprint(model.state_dict()) == expected['model']
    assert fingerprint(opt.state_dict()) == expected['optimizer']
    assert fingerprint(average.state_dict()) == expected['average']
    assert fingerprint(ck.capture_training_state(loader, 'cpu')) == expected['rng']
    expected_average = fingerprint(average.averages['200'])
    save(store, final_path, cfg, model, opt, average, loader, 4)
    del model, opt, schedule, average, loader
    gc.collect()

    raw_sha = hashlib.sha256(store.blobs[final_path]).hexdigest()
    rng = torch.get_rng_state().clone()
    receipt = export_averaged_checkpoint(final_path, avg_path, 200, payload_writer=store.write)
    assert torch.equal(rng, torch.get_rng_state())
    assert hashlib.sha256(store.blobs[final_path]).hexdigest() == raw_sha
    assert receipt['parent_checkpoint']['sha256'] == raw_sha
    assert receipt['checkpoint_sha256'] == hashlib.sha256(store.blobs[avg_path]).hexdigest()
    exported = ck.read_checkpoint(avg_path)
    assert 'optimizer_state' not in exported
    assert exported['averaged_weights']['snapshots'] == 3
    assert exported['tokenizer_contract'] == OmegaConf.to_container(cfg.model.output_bias_prior.tokenizer)
    model = GPT(cfg.model)
    ck.load_model_weights(model, exported)
    assert fingerprint(dict(model.named_parameters())) == expected_average
    with pytest.raises(ValueError, match='Averaged inference weights'):
        ck.load_model_weights(model, exported, for_resume=True)
    with pytest.raises(ValueError, match='new output path'):
        export_averaged_checkpoint(final_path, avg_path, 200, payload_writer=store.write)
    # Production loader verifies actual local tokenizer/data identity; no split
    # is sampled and no validation/test loss is computed.
    loaded = ck.checkpoint_loader(cfg, exported, cfg.data.path, 'cpu')
    assert loaded.tokenizer_contract == exported['tokenizer_contract']
    wrong = dict(exported, tokenizer_contract=dict(exported['tokenizer_contract'], version='wrong'))
    with pytest.raises(ValueError, match='tokenizer identity differs'):
        ck.checkpoint_loader(cfg, wrong, cfg.data.path, 'cpu')
    del exported, model, loaded, store
    gc.collect()


@pytest.mark.parametrize('kind', ['silu', 'gelu_exact'])
def test_cache_interleaving_reset_and_ce_only(kind):
    cfg = config(kind)
    model = GPT(cfg.model).eval()
    first = torch.tensor([[11, 22, 33, 44, 55, 66, 77]])
    second = first + 100
    a = build_caches(cfg, 'cpu', batch_size=1, dtype=torch.float32)
    b = build_caches(cfg, 'cpu', batch_size=1, dtype=torch.float32)
    with torch.no_grad():
        expected_a = model(first)[0]; expected_b = model(second)[0]
        a0 = model(first[:, :2], cache=a, start_pos=0)[0]
        b0 = model(second[:, :4], cache=b, start_pos=0)[0]
        a1 = model(first[:, 2:], cache=a, start_pos=2)[0]
        b1 = model(second[:, 4:], cache=b, start_pos=4)[0]
        torch.testing.assert_close(torch.cat([a0, a1], 1), expected_a, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(torch.cat([b0, b1], 1), expected_b, rtol=1e-5, atol=1e-6)
        for cache in a:
            cache.reset()
        torch.testing.assert_close(model(second, cache=a)[0], expected_b, rtol=1e-5, atol=1e-6)
        for aux in (False, True):
            logits, loss, parts = model(first, second, include_aux_loss=aux,
                                       return_loss_breakdown=True)
            expected = torch.nn.functional.cross_entropy(logits.flatten(0, 1), second.flatten())
            torch.testing.assert_close(loss, expected, rtol=0, atol=0)
            assert parts['aux_loss'] is None
        with pytest.raises(ValueError, match='Training objective requires explicit contract'):
            model(first, second, include_training_objective=True)
