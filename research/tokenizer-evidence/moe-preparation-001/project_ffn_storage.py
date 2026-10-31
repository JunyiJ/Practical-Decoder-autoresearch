"""Read-only meta-shape accounting. No training, tokenization or checkpoint IO.

MoE/wider-dense totals are hypothetical component replacements, not constructed
or validated full models. All optimizer states are charged even for idle experts.
This report cannot authorize a launch or deletion.
"""
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil

import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.models.mlp import SwiGLU
from src.models.tokenizer_architecture import validate_architecture_model
from src.train.optimizers import build_optimizer


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project():
    torch.set_num_threads(1)
    recipe_path = 'research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json'
    cfg = OmegaConf.create(json.loads(Path(recipe_path).read_text())['config'])
    with torch.device('meta'):
        model = GPT(cfg.model)
        validate_architecture_model(model)
        optimizer = build_optimizer(model, cfg.training)
        parameters = dict(model.named_parameters())
        p = sum(t.numel() for t in parameters.values())
        ffn = sum(t.numel() for n, t in parameters.items() if '.mlp.' in n)
        assert ffn == 4 * 3 * 256 * 672
        states = {}
        for child in optimizer.optimizers:
            name = type(child).__name__
            assert name in ('Muon', 'AdamW')
            states[name] = dict(parameters=sum(t.numel() for g in child.param_groups
                                              for t in g['params']),
                                tensors=sum(len(g['params']) for g in child.param_groups))
        assert states['Muon']['parameters'] + states['AdamW']['parameters'] == p
        persistent_buffers = sum(t.numel() * t.element_size()
                                 for n, t in model.named_buffers() if n in model.state_dict())
        # Independent shape counts for proposed FFN replacements, without
        # bypassing the full model's currently unsupported architecture policy.
        wide = sum(t.numel() for t in SwiGLU(256, 1344, 0.).parameters()) * 4
        experts = sum(t.numel() for t in SwiGLU(256, 336, 0.).parameters()) * 4 * 4
        routers = sum(t.numel() for t in torch.nn.Linear(256, 4, bias=False).parameters()) * 4
        assert wide == experts == 2 * ffn and routers == 4096
        for kind in ('silu', 'gelu_exact'):
            assert sum(t.numel() for t in SwiGLU(256, 672, 0., gate_activation=kind).parameters()) * 4 == ffn
    arms = {}
    for name, extra_hidden, extra_adam, extra_adam_tensors in (
            ('dense_silu', 0, 0, 0), ('dense_gelu_exact', 0, 0, 0),
            ('wider_dense_silu', wide - ffn, 0, 0),
            ('moe_top2of4_silu', experts - ffn, routers, 4)):
        total = p + extra_hidden + extra_adam
        hidden = states['Muon']['parameters'] + extra_hidden
        adam = states['AdamW']['parameters'] + extra_adam
        adam_tensors = states['AdamW']['tensors'] + extra_adam_tensors
        # FP32 model + FP32 EMA + H200; Muon momentum; Adam first/second
        # moments and one FP32 step scalar per Adam tensor. Excludes RNG and
        # serialization headers, so this is payload arithmetic, not file size.
        opt = 4 * hidden + 8 * adam + 4 * adam_tensors
        pair = 12 * total + opt + 2 * persistent_buffers
        # Same conservative 24 B/parameter + 2 MiB, then 15% method used by
        # wide-hybrid preparation; an extra model-sized overlap is included
        # beyond raw(model+EMA+two moments)+H200. No intended cleanup credit.
        allowance = math.ceil((24 * total + 2 * persistent_buffers + 2 * 1024**2) * 1.15)
        assert pair + 4 * total <= 24 * total + 2 * persistent_buffers
        arms[name] = dict(parameters=total, muon_parameters=hidden, adam_parameters=adam,
                          optimizer_tensor_bytes=opt, persistent_buffer_bytes=persistent_buffers,
                          raw_h200_payload_bytes=pair, per_trajectory_allowance_bytes=allowance)
    free = shutil.disk_usage('.').free
    reserve, metadata = 8 * 1024**3, 256 * 1024**2
    panels = {}
    for name, kinds in (
            ('moe_three_arm', ('dense_silu', 'wider_dense_silu', 'moe_top2of4_silu')),
            ('activation_two_arm', ('dense_silu', 'dense_gelu_exact'))):
        per_family = sum(arms[k]['per_trajectory_allowance_bytes'] for k in kinds)
        phases = dict(pilots=per_family, lr_screen=3 * per_family,
                      fresh_confirmation=4 * per_family)
        required = reserve + metadata + max(phases.values())
        panels[name] = dict(arms=list(kinds), pilot_trajectories=len(kinds),
                           screening_trajectories=3 * len(kinds), fresh_trajectories=3 * len(kinds),
                           protected_selected_screen_pairs=len(kinds),
                           protected_fresh_pairs=3 * len(kinds),
                           phase_allowance_bytes=phases, required_free_bytes=required,
                           margin_bytes=free-required, fits_projection_at_observation=free >= required,
                           final_payload_plus_reserve_bytes=reserve + 4 * sum(
                               arms[k]['raw_h200_payload_bytes'] for k in kinds))
    refs = (recipe_path, 'src/models/gpt.py', 'src/models/mlp.py',
            'src/models/tokenizer_architecture.py', 'src/models/gated_activation.py',
            'src/train/optimizers.py', 'src/train/weight_averaging.py', 'src/utils/checkpoint.py',
            'research/tokenizer-evidence/wide-hybrid-preparation-001/project_rank_storage.py')
    return dict(status='prospective_shape_accounting_not_launch_authorization',
                observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                torch_version=torch.__version__, helper_sha256=sha(__file__),
                source_refs={p: sha(p) for p in refs}, free_bytes=free,
                reserve_bytes=reserve, metadata_allowance_bytes=metadata,
                baseline_optimizer_groups=states, arms=arms, panels=panels,
                assumptions=[
                    'FP32; one raw/H200 pair, one EMA half-life, all optimizer states charged.',
                    'Pilots cleaned only after complete review; losing screens only after frozen selection and audited cleanup.',
                    'One selected screening pair plus all three fresh pairs per arm remain protected.',
                    'Existing protected files already consume disk; observed free space gives no credit for deleting them.',
                    'Raw/export/audit overlap and metadata allowance retained; no periodic or best checkpoints.',
                    'No learned checkpoint weights read, written, removed or compressed.',
                    'Hypothetical replacement shapes do not establish model validity, numerical correctness or runtime resource fit.',
                    'Require exact implementation projection, current free-space recheck and guarded pilots before any quality run.'],
                training_launched=False, holdout_consumed=False, deletion_authorized=False)


if __name__ == '__main__':
    print(json.dumps(project(), indent=2))
