"""Exact prospective activation requests; constructing a config never launches."""
import copy
import json
from pathlib import Path

from ..models.tokenizer_activation import POLICY
from ..models.tokenizer_architecture import architecture_contract
from .protocol import ROOT, file_digest
from . import tokenizer_activation_rules as rules

BASELINE = ROOT / 'research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json'
BASELINE_SHA256 = '14936864a1ef15aaa1e97c763755f0b090c0e31babdb5c27edc657417aa4f3d3'
LIMITS = dict(reserve_bytes=8*1024**3, rss_limit_bytes=4*1024**3,
              driver_limit_bytes=6*1024**3, pilot_seconds=600, quality_seconds=7200,
              cpu_threads=4, device='mps', cpu_fallback=False)


def config(output, spec):
    candidates = rules.pilot_specs() + rules.screen_specs()
    for a in rules.RATES:
        for b in rules.RATES:
            candidates += rules.confirmation_specs(dict(zip(rules.FAMILIES, (a, b))))
    if not any(set(spec) == set(s) and all(type(spec[k]) is type(v) and spec[k] == v
                                         for k, v in s.items()) for s in candidates):
        raise ValueError('Request is outside the finite activation design')
    if file_digest(BASELINE) != BASELINE_SHA256:
        raise ValueError('Adopted MLA baseline changed')
    cfg = copy.deepcopy(json.loads(BASELINE.read_text())['config'])
    cfg['model'].update(tokenizer_architecture_policy=POLICY,
                        gated_activation_kind=spec['family'], gated_activation_width_policy=POLICY)
    architecture_contract(cfg['model'])
    output = Path(output).resolve()
    t = cfg['training']
    t.update(seed=spec['seed'], learning_rate=spec['lr'], max_iters=spec['steps'],
             device='mps', strict_device=True, resume_from=None, max_train_seconds=None,
             checkpoint_dir=str(output/'checkpoints'), metrics_file=str(output/'metrics.jsonl'),
             summary_file=str(output/'training-summary.json'))
    pilot = spec['stage'] == 'pilot'
    t['eval_every'] = 20 if pilot else 300
    start = 13 if pilot else 800
    t['lr_schedule'].update(total_steps=spec['steps'], start_step=start)
    t['weight_averaging'].update(total_steps=spec['steps'], start_step=start,
                                interval=1 if pilot else 10, half_lives=[200])
    return cfg
