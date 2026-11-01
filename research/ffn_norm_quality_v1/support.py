"""Existing bounded file and cache checks through the score runtime."""
import gc,os,subprocess
from pathlib import Path
import torch
from . import campaign_contract as c
from research.ffn_norm_v1 import validation as v
from research.ffn_norm_v1.runtime.utils.checkpoint import load_model_weights,read_checkpoint
from research.ffn_norm_v1.runtime.attention.cache import build_caches
from research.block_norm_v1.pilot_execution import exclusive as _exclusive
from research.ffn_norm_campaign_v1.pilot import idle as previous_idle
from src.research.tokenizer_wide_hybrid_measurements import check_kernel
from src.research.tokenizer_wide_hybrid_timing import allocated_bytes,token_digest
require=c.require

def file_ref(path,weight=False):return dict(path=str(path),sha256=c.file_sha(path,root=c.ROOT,max_bytes=(512 if weight else 2)*1024**2))
def idle(allowed=()):
    previous_idle(allowed)
    for line in subprocess.check_output(['ps','-axo','pid=,command='],text=True).splitlines():
        row=line.strip().split(None,1)
        if len(row)==2 and 'research.ffn_norm_quality_v1' in row[1] and Path(row[1].split()[0]).name.lower().startswith('python'):
            require(int(row[0]) in {os.getpid(),*allowed},'Other score quality worker is live')

def cache_audit(request):
    binding={k:request['files'][k] for k in ('config','parent','h200')}
    model,data,cfg=v.load(binding,'mps');ids=data.train_data[:96].tolist();rows=[]
    try:
        for endpoint in ('parent','h200'):
            load_model_weights(model,read_checkpoint(binding[endpoint]['path']))
            for batch in (1,8):
                fixed=torch.tensor(ids,device='mps',dtype=torch.int64).unsqueeze(0).repeat(batch,1)
                factory=lambda:build_caches(cfg,'mps',batch_size=batch,dtype=torch.float32)
                result=check_kernel(model,fixed,factory);cache=factory()
                state=allocated_bytes(b for layer in cache for b in layer.buffers());require(state==196608*batch,'Pilot cache storage differs')
                rows.append(dict(endpoint=endpoint,batch=batch,state_bytes=state,**result));del cache,fixed
        return dict(training96_sha256=token_digest(ids),rows=rows)
    finally:del model,data,cfg;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()



def exclusive(path,value):
    # Immutable external anchors/reviews/retirement shards use <=32MiB in total;
    # the finite twelve-slot campaign's declared groups plus this fit256MiB.
    path=Path(path);base=c.ROOT/'research/tokenizer-evidence/ffn-norm-preparation-001'
    if path.is_relative_to(base) and path.relative_to(base).parts[0].startswith('quality001'):
        used=sum(p.stat().st_size for p in base.rglob('*') if p.is_file() and p.relative_to(base).parts[0].startswith('quality001'))
        require(used+len((c.canonical(value)+'\n').encode())<=32*1024**2,'Complete external quality evidence allowance exceeded')
    return _exclusive(path,value)
