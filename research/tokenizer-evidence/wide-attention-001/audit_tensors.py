"""Bounded CPU audit of retained raw/EMA/export tensors; no evaluation or test access."""
import gc
import signal
signal.alarm(600)
import json
import time
from pathlib import Path
import torch
from src.research.memory_guard import start_memory_guard
from src.research.protocol import file_digest
from src.research.runner import atomic_json

root=Path('research_runs/tokenizer-wide-attention-001');out=Path('research/tokenizer-evidence/wide-attention-001')
state=json.loads((root/'state.json').read_text());assert state['status']=='completed' and len(state['records'])==12
torch.set_num_threads(2);start_memory_guard();started=time.monotonic();rows=[]
def finite(x):
    if torch.is_tensor(x):
        if x.is_floating_point() or x.is_complex():assert torch.isfinite(x).all()
    elif isinstance(x,dict):
        for v in x.values():finite(v)
    elif isinstance(x,(list,tuple)):
        for v in x:finite(v)
for r in state['records']:
    assert time.monotonic()-started<600
    raw_path=Path(r['raw']['checkpoint']);avg_path=Path(r['h200']['checkpoint'])
    assert file_digest(raw_path)==r['raw']['checkpoint_sha256'] and file_digest(avg_path)==r['h200']['checkpoint_sha256']
    raw=torch.load(raw_path,map_location='cpu',weights_only=False);avg=torch.load(avg_path,map_location='cpu',weights_only=False)
    finite(raw);finite(avg)
    ema=raw['training_state']['weight_averaging_state'];names=ema['parameter_names']
    assert ema['snapshots']==41 and set(ema['averages']['200'])==set(names)
    assert set(raw['model_state'])==set(avg['model_state'])
    for name,value in avg['model_state'].items():
        expected=ema['averages']['200'][name] if name in names else raw['model_state'][name]
        assert torch.equal(value,expected),name
    assert 'lm_head.weight' in names and 'lm_head.weight' in avg['model_state']
    expected_params={'gqa':62877696,'mla':62424192}[r['family']]
    assert sum(avg['model_state'][n].numel() for n in names)==expected_params
    rows.append(dict(id=r['id'],raw_sha256=r['raw']['checkpoint_sha256'],h200_sha256=r['h200']['checkpoint_sha256'],
        all_saved_float_tensors_finite=True,export_exactly_matches_saved_ema_and_raw_buffers=True,parameter_count=expected_params))
    del raw,avg,ema,expected,value;gc.collect()
    print('PASS '+r['id'],flush=True)
atomic_json(out/'tensor-audit.json',dict(status='passed',records=rows,wall_seconds=time.monotonic()-started,
    helper_sha256=file_digest(Path(__file__)),quality_state_sha256=file_digest(root/'state.json'),
    scope='All24 retained raw/H200 files: finite saved model/optimizer/EMA tensors, exact H200-to-saved-EMA and buffer equality, canonical parameter layout. Does not replay optimizer updates or recompute validation losses.',holdout_consumed=False))
