"""Audit saved pilot artifacts without scoring any data or rerunning training."""
import importlib
import json
import math
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.research.protocol import digest, file_digest
from src.train.train import _make_loader, _training_contract
from src.utils.checkpoint import checkpoint_loader, read_checkpoint

pilot = importlib.import_module('research.tokenizer-evidence.hybrid-preparation-001.run_pilot')
root = Path('research_runs/tokenizer-hybrid-pilot-001').resolve()
out = Path('research/tokenizer-evidence/hybrid-pilot-001')
plan = pilot.verify(root)
state = json.loads((root/'state.json').read_text())
r = json.loads((root/'result.json').read_text())
assert state['status'] == r['status'] == 'completed' and state['plan_sha256'] == digest(plan)
assert not r['holdout_consumed'] and r['training']['completed_steps'] == 20
raw = read_checkpoint(r['raw_checkpoint']); avg = read_checkpoint(r['h200_checkpoint'])
assert raw['cfg'] == avg['cfg'] == plan['config']
assert raw['step'] == avg['step'] == 20
assert avg['averaged_weights']['snapshots'] == 3
assert raw['training_state']['weight_averaging_state']['snapshots'] == 3
assert avg['parent_checkpoint']['sha256'] == r['raw_sha256'] == file_digest(Path(r['raw_checkpoint']))
assert r['h200_sha256'] == file_digest(Path(r['h200_checkpoint']))
assert 'optimizer_state' not in avg
assert raw['optimizer_state']['independent_decay'] == .00006
for opt in raw['optimizer_state']['optimizers']:
    for values in opt['state'].values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                assert torch.isfinite(value).all()
            if key == 'step':
                assert int(value) == 20
for c in (raw, avg):
    assert all(torch.isfinite(v).all() for v in c['model_state'].values())
cfg = OmegaConf.create(plan['config']); loader = _make_loader(cfg, 'cpu')
assert raw['training_state']['contract'] == _training_contract(cfg, 'mps', loader)
assert torch.equal(raw['training_state']['loader_rng']['test'], loader.generators['test'].get_state())
for _ in range(20):
    loader.get_batch('train')
assert torch.equal(raw['training_state']['loader_rng']['train'], loader.generators['train'].get_state())
assert raw['training_state']['contract']['tokenizer'] == avg['tokenizer_contract'] == loader.tokenizer_contract
checkpoint_loader(cfg, avg, cfg.data.path, 'cpu')
assert [v['batch'] for v in r['cache_checks']] == [1, 8]
for v in r['cache_checks']:
    assert math.isfinite(v['max_absolute_error'])
    assert v['per_layer_cache_bytes'] == [v['batch']*10496]*3 + [v['batch']*24576]
    assert v['allocated_cache_bytes'] == v['batch']*56064
size = sum(p.stat().st_size for p in root.rglob('*') if p.is_file())
assert size <= plan['max_output_bytes'] and shutil.disk_usage(root).free > plan['min_free_bytes']
review = dict(status='passed', steps=20, ema_snapshots=3,
              result_sha256=file_digest(root/'result.json'), raw_sha256=r['raw_sha256'],
              h200_sha256=r['h200_sha256'], training_seconds=r['training']['training_time_sec'],
              checkpoint_pair_bytes=Path(r['raw_checkpoint']).stat().st_size+Path(r['h200_checkpoint']).stat().st_size,
              output_bytes=size, cache_checks=r['cache_checks'],
              train_sampler_replayed=True, test_rng_untouched=True,
              scope='Feasibility only, no quality ranking or adoption; source/runtime/recipe/protocol/corpus/helper verified.')
for name in ('plan.json', 'state.json', 'result.json', 'worker.log', 'metrics.jsonl'):
    shutil.copyfile(root/name, out/name)
(out/'review.json').write_text(json.dumps(review, indent=2)+'\n')
print(json.dumps(review, indent=2))
