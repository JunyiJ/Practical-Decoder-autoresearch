"""Two fixed optimizer updates per homogeneous MPS control; no quality selection."""
import gc
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.attention.cache import build_caches
from src.models.gpt import GPT
from src.research.memory_guard import start_memory_guard
from src.research.protocol import file_digest
from src.research.runner import atomic_json, source_fingerprint
from src.research.tokenizer_attention_bench import check_cached, buffer_bytes
from src.research.tokenizer_hybrid_study import config, PROTOCOL
from src.research.tokenizer_study import ROOT, runtime_identity
from src.train.optimizers import build_optimizer
from src.train.train import _make_loader

out = ROOT/'research/tokenizer-evidence/hybrid-preparation-001/control-preflight-v2.json'
if out.exists():
    raise FileExistsError(out)
if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
    raise RuntimeError('MPS required without fallback')
source = source_fingerprint()
runtime = runtime_identity()
protocol = file_digest(PROTOCOL)
torch.set_num_threads(4)
start_memory_guard()
rows = []
for family, seed in [('gated_mla', 2992), ('kda', 2993)]:
    torch.manual_seed(seed)
    cfg = OmegaConf.create(config(Path('/tmp/no-checkpoint-control-preflight'), family, seed, .00125))
    loader = _make_loader(cfg, 'cpu')
    ids = loader.train_data[:129].unsqueeze(0).repeat(8, 1).to('mps')
    model = GPT(cfg.model).to('mps')
    optimizer = build_optimizer(model, cfg.training)
    losses = []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(ids[:, :-1].contiguous(), ids[:, 1:].contiguous())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        losses.append(float(loss.detach()))
    for opt in optimizer.state_dict()['optimizers']:
        for state in opt['state'].values():
            assert all(torch.isfinite(v).all() for v in state.values() if isinstance(v, torch.Tensor))
    model.eval()
    checks = []
    for batch in (1, 8):
        error = check_cached(model, cfg, ids[:batch, :12])
        caches = build_caches(cfg, 'mps', batch_size=batch)
        size = buffer_bytes(caches)
        assert size == batch*(98304 if family == 'gated_mla' else 41984)
        checks.append(dict(batch=batch, max_absolute_error=error, state_bytes=size))
        del caches
    rows.append(dict(family=family, seed=seed, updates=2, diagnostic_losses=losses, cache_checks=checks,
                     parameters=sum(p.numel() for p in model.parameters())))
    print(json.dumps(rows[-1]), flush=True)
    del loss, ids, optimizer, model, loader
    gc.collect(); torch.mps.empty_cache()
assert source_fingerprint() == source and file_digest(PROTOCOL) == protocol
atomic_json(out, dict(status='passed', source=source, runtime=runtime, protocol_sha256=protocol,
                     helper_sha256=file_digest(Path(__file__)), records=rows, holdout_consumed=False,
                     scope='Two repeated fixed-training-batch feasibility updates per control; no quality ranking or saved checkpoint.'))
