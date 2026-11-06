"""Bind old immutable sources and only this additive implementation."""
import hashlib,json,os
from pathlib import Path
import torch
from .contract import ROOT,require,fingerprint

def reference(path):
    path=Path(path);return dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def source_inventory():
    old=json.loads((ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json').read_text())['sources']
    require(len(old)==918,'Exact predecessor inventory required')
    for n,h in old.items():require(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,'Frozen source changed: '+n)
    result=dict(old)
    for p in sorted((ROOT/'research/body_allocation_model_v1').glob('*.py')):result[str(p.relative_to(ROOT))]=reference(p)['sha256']
    return result

def environment():
    flags={k:os.environ.get(k) for k in ('PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')}
    require(list(flags.values())==['0','0','4','4','4'] and torch.get_num_threads()==4,'Strict environment required')
    return dict(flags=flags,torch_version=str(torch.__version__),threads=4)
