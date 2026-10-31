"""New tiny MPS copy diagnostic; no model, training, fixture rewrite or retry."""
import json,hashlib
from pathlib import Path
import torch
from src.research.protocol import ROOT
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from .mps_guard import StrictMpsGuard
base=ROOT/'research/tokenizer-evidence/block-norm-preparation-001'
with session_lock(GLOBAL_LOCK.parent),StrictMpsGuard(base/'validation-transfer-diagnostic-001-guard.json',ROOT):
    idle();torch.set_num_threads(4);rows=[]
    values=[.5,1.,2.,10.75]
    for name,gpu in [('literal',torch.tensor(values,device='mps',dtype=torch.float32)),
                     ('cross_entropy',torch.nn.functional.cross_entropy(torch.zeros(4,7,device='mps'),torch.tensor([0,1,2,3],device='mps'),reduction='none'))]:
        torch.mps.synchronize()
        fp32=gpu.cpu();combined=gpu.to('cpu',dtype=torch.float64);separate=gpu.cpu().double()
        rows.append(dict(case=name,fp32_cpu=fp32.tolist(),combined_transfer_fp64=combined.tolist(),separate_transfer_fp64=separate.tolist(),combined_equals_separate=torch.equal(combined,separate)))
    idle()
with (base/'validation-transfer-diagnostic-001.json').open('x') as f:json.dump(dict(torch_version=torch.__version__,rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Tiny FP32 transfer diagnostic; no model/optimizer/quality work'),f,indent=2);f.write('\n')
print(json.dumps(rows))
