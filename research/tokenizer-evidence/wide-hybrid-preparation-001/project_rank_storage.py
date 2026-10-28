"""Full draft rank-panel meta accounting, never a dispatcher or launch approval."""
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
from src.research.tokenizer_wide_hybrid_config import FAMILIES, rank_config
from src.train.optimizers import build_optimizer


def tensor_bytes(values):
    return sum(t.numel()*t.element_size() for t in values)


records = []
for width in (512, 1024):
    for family in FAMILIES:
        for divisor in (8, 4, 2):
            cfg = OmegaConf.create(rank_config(Path('/tmp/wide-hybrid-rank-meta'), family,
                width, bottleneck_divisor=divisor))
            with torch.device('meta'):
                model = GPT(cfg.model)
                validate_architecture_model(model)
                optimizer = build_optimizer(model, cfg.training)
                caches = build_caches(cfg, 'meta', batch_size=1, dtype=torch.float32)
                for p in model.parameters(): p.grad = torch.zeros_like(p)
                for child in optimizer.optimizers:
                    for group in child.param_groups:
                        if type(child) is torch.optim.Muon:
                            child._init_group(group, [], [], [])
                        elif type(child) is torch.optim.AdamW:
                            child._init_group(group, [], [], [], [], [], [])
                        else:
                            raise ValueError('Unaccounted optimizer type')
            parameters = sum(p.numel() for p in model.parameters())
            parameter_bytes = tensor_bytes(model.parameters())
            saved_keys = set(model.state_dict())
            buffer_bytes = tensor_bytes(b for name,b in model.named_buffers() if name in saved_keys)
            optimizer_bytes = sum(tensor_bytes(s.values()) for child in optimizer.optimizers
                                  for s in child.state.values())
            pair_bytes = 3*parameter_bytes + optimizer_bytes + 2*buffer_bytes
            conservative_tensor_bytes = 6*parameter_bytes + 2*buffer_bytes
            assert pair_bytes <= conservative_tensor_bytes
            records.append(dict(width=width, family=family, bottleneck_divisor=divisor,
                parameters=parameters, parameter_bytes=parameter_bytes,
                optimizer_tensor_bytes=optimizer_bytes, persistent_buffer_bytes=buffer_bytes,
                raw_h200_model_optimizer_ema_tensor_bytes=pair_bytes,
                cache_bytes_batch1=tensor_bytes(b for c in caches for b in c.buffers()),
                per_trajectory_allowance_bytes=math.ceil((conservative_tensor_bytes+2*1024**2)*1.15)))

reserve = 8*1024**3
panels = {}
for width in (512, 1024):
    families = []
    for family in FAMILIES:
        rows = {r['bottleneck_divisor']:r for r in records if r['width']==width and r['family']==family}
        allowance = {k:r['per_trajectory_allowance_bytes'] for k,r in rows.items()}
        largest = max(allowance, key=allowance.get)
        assert largest == 2
        # One largest-rank pilot, three anchor LR screens, both rank probes,
        # then three confirmations reserved at the most costly possible selection.
        total = allowance[2] + 3*allowance[4] + allowance[8]+allowance[2] + 3*max(allowance.values())
        families.append(dict(family=family, trajectories=9, allowance_bytes=total,
                             worst_confirmation_divisor=largest))
    panels[str(width)] = dict(families=families, trajectories=27,
        required_free_bytes=reserve+4*1024**2+sum(f['allowance_bytes'] for f in families))

result = dict(status='draft_rank_panel_accounting_not_launch_plan',
    at=datetime.datetime.now(datetime.timezone.utc).isoformat(), torch_version=torch.__version__,
    free_bytes=shutil.disk_usage('.').free, reserve_bytes=reserve, records=records, panels=panels,
    accounting='Exact constructed meta model/optimizer/EMA tensor shapes; 24bytes/parameter plus persistent buffers and2MiB metadata, then15% allowance per retained trajectory. Additional4MiB round metadata and8GiB reserve. No measured checkpoint size, MPS RSS, time or quality claim. Numerical/model continuation prerequisites and fitted/frozen plans remain required.',
    helper_sha256=file_digest(Path(__file__)))
Path(__file__).with_name('rank-storage-projection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
