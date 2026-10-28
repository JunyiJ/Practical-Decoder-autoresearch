"""Meta-device parameter/cache accounting, no training or corpus scoring."""
import datetime
import json
import math
import shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from src.research.tokenizer_wide_config import config
from src.models.gpt import GPT
from src.models.tokenizer_architecture import validate_architecture_model
from src.train.optimizers import build_optimizer
from src.attention.cache import KVCache
from src.research.protocol import file_digest
rows=[]
for width in (512,1024):
    for family in ('gqa','mla'):
        cfg=OmegaConf.create(config(Path('/tmp/wide-meta-accounting'),family,width,device='cpu'))
        with torch.device('meta'):
            model=GPT(cfg.model);validate_architecture_model(model);build_optimizer(model,cfg.training)
            caches=[KVCache(cfg,'meta',batch_size=1) for _ in range(4)]
        count=sum(p.numel() for p in model.parameters())
        cache_bytes=sum(b.numel()*b.element_size() for c in caches for b in c.buffers())
        rows.append(dict(width=width,family=family,parameters=count,parameter_bytes=4*count,
            cache_bytes_batch1=cache_bytes,raw_h200_tensor_upper_bytes=24*count,
            per_trial_allowance_bytes=math.ceil((24*count+2*1024**2)*1.15)))
result=dict(status='prospective_accounting_not_launch_plan',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    free_bytes=shutil.disk_usage('.').free,reserve_bytes=8*1024**3,records=rows,
    panels={str(w):8*1024**3+4*1024**2+7*sum(r['per_trial_allowance_bytes'] for r in rows if r['width']==w) for w in (512,1024)},
    panel_description='Two retained pilots, six LR screens, six fresh confirmations: seven trajectories per family. CPU continuation fixtures serialized only in memory. No MPS measurement or quality claim.',
    accounting='24 bytes/parameter conservative tensor bound, plus2MiB metadata per trajectory, then15% headroom; additional4MiB round metadata and8GiB reserve.',
    helper_sha256=file_digest(Path(__file__)))
Path(__file__).with_name('storage-projection.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
