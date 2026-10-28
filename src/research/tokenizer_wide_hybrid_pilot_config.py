"""Prospective resource-pilot contract only; no freeze, dispatch or retry API.

Pilots never select recipes. Numerical prerequisites, source/data binding and a
full fitted/frozen campaign remain necessary before any real model is run.
"""
import json
from pathlib import Path

from .tokenizer_wide_hybrid_config import FAMILIES, rank_config
from .protocol import ROOT
from .tokenizer_serial_process import RESERVE

METADATA = 4*1024**2
PROJECTION = ROOT/'research/tokenizer-evidence/wide-hybrid-preparation-001/rank-storage-projection.json'

CONTRACT = dict(version=1, width=512, seed=3491, divisor=2, lr=.0025,
                steps=20, device='mps', ema_start=13, ema_interval=1,
                ema_half_life=200, ema_snapshots=8, cooldown_start=13,
                job_seconds=600, rss_limit_bytes=4*1024**3,
                driver_limit_bytes=6*1024**3, reserve_bytes=RESERVE,
                log_limit_bytes=METADATA, selection_eligible=False)


def specs():
    return [dict(stage='pilot', family=f, width=512, seed=3491,
                 divisor=2, lr=.0025, steps=20) for f in FAMILIES]


def config(output, family):
    cfg = rank_config(Path(output), family, width=512, seed=3491,
                      lr=.0025, steps=20, device='mps', bottleneck_divisor=2)
    t = cfg['training']
    t['eval_every'] = 20
    t['lr_schedule'].update(total_steps=20, start_step=13)
    t['weight_averaging'].update(total_steps=20, start_step=13,
                               interval=1, half_lives=[200])
    return cfg


def storage_contract(completed=()):
    """Reserve ALL quality runs plus remaining pilots, never just the next job.

    completed is an already audited ordered prefix, not proof of successful
    pilots. A future controller must verify retained files before supplying it.
    The metadata allowance is deliberately not discounted after a pilot.
    """
    if (not isinstance(completed, (tuple, list)) or len(completed) > 3
            or list(completed) != list(FAMILIES[:len(completed)])):
        raise ValueError('Completed pilots must be the exact ordered family prefix')
    projection = json.loads(PROJECTION.read_text())
    rows = [r for r in projection['records'] if r['width'] == 512]
    keys = [(r['family'], r['bottleneck_divisor']) for r in rows]
    expected = {(f, d) for f in FAMILIES for d in (2, 4, 8)}
    if len(keys) != 9 or set(keys) != expected:
        raise ValueError('Incomplete or duplicate width512 allocation records')
    budgets = dict(zip(keys, (r['per_trajectory_allowance_bytes'] for r in rows)))
    if any(type(n) is not int or n <= 0 for n in budgets.values()):
        raise ValueError('Trajectory allowances must be positive integer bytes')
    if any(budgets[f, 2] < max(budgets[f, d] for d in (4, 8)) for f in FAMILIES):
        raise ValueError('D/2 no longer bounds the proposed pilot family')
    quality = sum(3*budgets[f, 4] + budgets[f, 8] + budgets[f, 2]
                  + 3*max(budgets[f, d] for d in (2, 4, 8)) for f in FAMILIES)
    pilots = {f: budgets[f, 2] for f in FAMILIES}
    full = RESERVE + METADATA + quality + sum(pilots.values())
    if (projection['reserve_bytes'] != RESERVE
            or projection['panels']['512']['trajectories'] != 27
            or projection['panels']['512']['required_free_bytes'] != full):
        raise ValueError('Full campaign accounting differs from retained projection')
    remaining = sum(pilots[f] for f in FAMILIES[len(completed):])
    return dict(pilot_allowances=pilots, quality_allowance_bytes=quality,
                remaining_pilot_allowance_bytes=remaining,
                metadata_allowance_bytes=METADATA, reserve_bytes=RESERVE,
                full_panel_required_free_bytes=full,
                required_free_bytes=RESERVE + METADATA + quality + remaining)


def require_storage(free_bytes, completed=()):
    if type(free_bytes) is not int or free_bytes < 0:
        raise ValueError('Free space must be a nonnegative integer byte observation')
    contract = storage_contract(completed)
    if free_bytes < contract['required_free_bytes']:
        raise ValueError('Complete retained panel does not fit; no partial pilot launch')
    return contract
