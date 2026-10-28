"""Meta-only complete anchor geometry/optimizer accounting, not a launch plan."""
import datetime
import json
import math
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf

from src.attention.cache import build_caches
from src.models.gpt import GPT
from src.models.tokenizer_architecture import validate_architecture_model
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_config import FAMILIES, config
from src.train.optimizers import build_optimizer


def tensor_bytes(values):
    return sum(t.numel()*t.element_size() for t in values)


rows = []
for width in (512, 1024):
    for family in FAMILIES:
        cfg = OmegaConf.create(config(Path('/tmp/wide-hybrid-meta'), family, width))
        with torch.device('meta'):
            model = GPT(cfg.model)
            validate_architecture_model(model)
            optimizer = build_optimizer(model, cfg.training)
            caches = build_caches(cfg, 'meta', batch_size=1, dtype=torch.float32)
            # Initialize the installed optimizer's actual state shapes. There is
            # no numerical optimizer step, forward pass, token loading or MPS use.
            # These private APIs are intentionally pinned by the recorded runtime.
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            for child in optimizer.optimizers:
                for group in child.param_groups:
                    if type(child) is torch.optim.Muon:
                        child._init_group(group, [], [], [])
                    elif type(child) is torch.optim.AdamW:
                        child._init_group(group, [], [], [], [], [], [])
                    else:
                        raise ValueError('Unaccounted optimizer type')
        parameter_bytes = tensor_bytes(model.parameters())
        saved_keys = set(model.state_dict())
        buffers_bytes = tensor_bytes(b for name, b in model.named_buffers() if name in saved_keys)
        optimizer_bytes = sum(tensor_bytes(s.values()) for child in optimizer.optimizers
                              for s in child.state.values())
        # Raw model + raw EMA + H200 model, with buffers in each model export.
        pair_tensor_bytes = 3*parameter_bytes + optimizer_bytes + 2*buffers_bytes
        conservative_tensor_bytes = 6*parameter_bytes + 2*buffers_bytes
        assert pair_tensor_bytes <= conservative_tensor_bytes
        rows.append(dict(width=width, family=family,
            parameters=sum(p.numel() for p in model.parameters()),
            parameter_bytes=parameter_bytes, persistent_buffer_bytes=buffers_bytes,
            optimizer_tensor_bytes=optimizer_bytes,
            raw_h200_model_optimizer_ema_tensor_bytes=pair_tensor_bytes,
            cache_bytes_batch1=tensor_bytes(b for c in caches for b in c.buffers()),
            per_trajectory_allowance_bytes=math.ceil((conservative_tensor_bytes+2*1024**2)*1.15)))

reserve = 8*1024**3
result = dict(status='anchor_accounting_only_not_frozen_HP_panel_or_launch_plan',
    at=datetime.datetime.now(datetime.timezone.utc).isoformat(), torch_version=torch.__version__,
    free_bytes=shutil.disk_usage('.').free, reserve_bytes=reserve, records=rows,
    anchor_panels={str(w):reserve+4*1024**2+9*sum(r['per_trajectory_allowance_bytes']
        for r in rows if r['width']==w) for w in (512,1024)},
    panel_description='Nine retained trajectories per family: one pilot, three LR screens, two stage2 probes, three fresh confirmations. Anchor geometry substituted for all probes; changed rank/geometry probes require recalculation before freeze.',
    accounting='Exact meta model/optimizer/EMA tensor counts; excludes RNG/serialization metadata. Guard projection uses conservative24bytes/parameter plus buffers and2MiB metadata per trajectory, then15% allowance,4MiB round metadata and8GiB reserve. Not measured export size or training RSS.',
    helper_sha256=file_digest(Path(__file__)))
Path(__file__).with_name('anchor-storage-projection.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
