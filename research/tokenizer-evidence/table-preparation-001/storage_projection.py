"""Prospective storage accounting only: no training, evaluation, or checkpoint writes."""
import copy
import datetime
import gc
import json
import math
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.models.tokenizer_tables import POLICY, KINDS
from src.research.runner import source_fingerprint


def project():
    torch.set_num_threads(1)
    recipe = json.loads(Path('research/tokenizer-evidence/horizon-001/adopted-width256-gqa-recipe.json').read_text())['config']
    reserve = 8 * 1024**3
    free = shutil.disk_usage('.').free
    panels = []
    for width in (64, 256):
        arms = {}
        for kind in KINDS:
            cfg = copy.deepcopy(recipe['model'])
            cfg.update(dim=width, n_heads=width//32, tokenizer_architecture_policy=POLICY,
                       weight_tying_kind=kind, weight_tying_width_policy=POLICY)
            model = GPT(OmegaConf.create(cfg))
            count = sum(p.numel() for p in model.parameters())
            # Raw: model + one EMA + at most two optimizer tensors per parameter.
            # H200 export: one model. Conservatively charge Adam's two tensors even
            # for Muon's one-buffer matrices; metadata covers buffers/scalars/RNG.
            pair_bound = 20 * count
            allowance = math.ceil(pair_bound * 1.15) + 2 * 1024**2
            arms[kind] = dict(parameters=count, fp32_parameter_bytes=4*count,
                              raw_h200_tensor_bound_bytes=pair_bound,
                              per_trial_allowance_bytes=allowance)
            del model
            gc.collect()
        # Three retained pilots plus 6 screening and 9 fresh confirmation trials.
        # Each arm gets two LR screens and three fresh confirmation seeds.
        output = 6 * sum(a['per_trial_allowance_bytes'] for a in arms.values()) + 4*1024**2
        panels.append(dict(width=width, arms=arms, pilot_trials=3, quality_trials=15,
                           output_allowance_bytes=output, required_free_bytes=reserve+output,
                           available_margin_bytes=free-reserve-output,
                           fits_at_observation=free >= reserve+output))
    return dict(created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                free_bytes=free, reserve_bytes=reserve, source=source_fingerprint(), panels=panels,
                status='prospective_accounting_not_launch_authorization',
                assumptions='FP32; one raw and one H200 file per trajectory; one EMA half-life; no periodic/best checkpoints; all artifacts retained. Pilots must validate actual costs, and remaining-panel guard must pass again before every launch.',
                quality_claim=None, training_launched=False)


if __name__ == '__main__':
    print(json.dumps(project(), indent=2))
