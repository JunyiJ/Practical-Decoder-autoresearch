"""Read-only full-validation scoring with checkpoint/data/target provenance."""
import copy
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from research.block_norm_v1 import campaign_contract as c
from .runtime.models.block_norm import FFN_NORM_POLICY, contract
from .runtime.models.gpt import GPT
from .runtime.models.output_bias import digest
from .runtime.utils.checkpoint import load_model_weights,read_checkpoint
from .validation_data import load as validation_data

require=c.require


def artifacts(binding):
    require(type(binding) is dict and set(binding)=={'config','h200','parent'},'Exact artifact bindings required')
    for kind,v in binding.items():
        require(type(v) is dict and set(v)=={'path','sha256'},'Exact artifact path/hash required')
        limit=2*1024**2 if kind=='config' else 512*1024**2
        require(c.file_sha(Path(v['path']),root=c.ROOT,max_bytes=limit)==v['sha256'],'Artifact hash differs')


def load(binding,device):
    import json
    artifacts(binding)
    cfg=OmegaConf.create(json.loads(Path(binding['config']['path']).read_text()))
    saved=read_checkpoint(binding['h200']['path'])
    require(c.canonical(saved['cfg'])==c.canonical(OmegaConf.to_container(cfg,resolve=True)),'Saved config differs')
    require(cfg.model.tokenizer_architecture_policy==FFN_NORM_POLICY,'Explicit SwiGLU product normalization policy required')
    contract(cfg.model)
    require(saved.get('parent_checkpoint')==binding['parent'],'H200 parent binding differs')
    require(saved.get('averaged_weights',{}).get('half_life')==200,'H200 weights required')
    data=validation_data(cfg,saved)
    require(set(data.splits)=={'train','val'},'Closed test boundary required')
    # Constructor RNG consumption must not affect subsequent training/inference.
    with torch.random.fork_rng(devices=[]):
        model=GPT(cfg.model).to(device).eval();load_model_weights(model,saved)
    del saved
    return model,data,cfg


def target_contract(data):
    ids=data.val_data
    require(ids.device.type=='cpu' and ids.dtype==torch.int64 and ids.ndim==1 and 2<=len(ids)<=1024*1024,'Bounded CPU validation tokens required')
    values=ids.tolist();n=len(values)-1
    # GPT2 token byte lengths sum exactly even across UTF8 token boundaries.
    import tiktoken
    enc=tiktoken.get_encoding('gpt2')
    require(data.tokenizer_contract['name']=='gpt2','GPT2 target byte accounting required')
    return dict(validation_tokens=len(values),targets=n,ids_sha256=digest(values),target_ids_sha256=digest(values[1:]),
        target_bytes=sum(len(enc.decode_single_token_bytes(t)) for t in values[1:]),
        window_counts=[min(128,n-i) for i in range(0,n,128)],block_counts=[min(2048,n-i) for i in range(0,n,2048)],
        tokenizer=data.tokenizer_contract,context=128,first_target_offset=1,tail_retained=True,holdout_scored=False)


def score(model,data):
    require(not model.training and next(model.parameters()).dtype==torch.float32,'Eval FP32 model required')
    require(model.cfg.block_size==128,'Context128 required')
    target=target_contract(data);ids=data.val_data;device=next(model.parameters()).device
    losses=[]
    with torch.no_grad():
        for start in range(0,len(ids)-1,128):
            count=min(128,len(ids)-1-start)
            x=ids[start:start+count].unsqueeze(0).to(device);y=ids[start+1:start+count+1].to(device)
            logits,_=model(x)
            require(logits.shape==(1,count,50257) and logits.dtype==torch.float32,'Exact GPT2 logit layout required')
            # Preserve FP32 bytes during MPS transfer; cast only after CPU arrival.
            values=F.cross_entropy(logits[0],y,reduction='none').to('cpu').to(dtype=torch.float64)
            require(torch.isfinite(values).all().item() and (values>=0).all().item(),'Finite nonnegative target losses required')
            losses.append(values)
    loss=torch.cat(losses);require(loss.numel()==target['targets'],'Target coverage differs')
    blocks=[float(loss[i:i+2048].sum()) for i in range(0,len(loss),2048)]
    windows=[float(v.sum()) for v in losses];total=float(loss.sum())
    return dict(target=target,window_loss_sums=windows,block_loss_sums=blocks,total_loss=total,ce=total/target['targets'],
                bits_per_byte=total/math.log(2)/target['target_bytes'],loss_dtype='float32_cross_entropy_float64_aggregation')


def validate_report(report,binding,data):
    require(type(report) is dict and set(report)=={'binding','target','window_loss_sums','block_loss_sums','total_loss','ce','bits_per_byte','loss_dtype'},'Exact report required')
    require(c.canonical(report['binding'])==c.canonical(binding),'Report checkpoint binding differs')
    target=target_contract(data);require(c.canonical(report['target'])==c.canonical(target),'Report target coverage/identity differs')
    require(report['loss_dtype']=='float32_cross_entropy_float64_aggregation','Loss arithmetic differs')
    for name,counts in [('window_loss_sums',target['window_counts']),('block_loss_sums',target['block_counts'])]:
        values=report[name];require(type(values) is list and len(values)==len(counts),'Loss block/window count differs')
        require(all(type(v) is float and math.isfinite(v) and v>=0 for v in values),'Finite FP64 loss sums required')
        require(math.isclose(sum(values),report['total_loss'],rel_tol=0,abs_tol=1e-8),'Loss sums differ')
    require(type(report['total_loss']) is float and math.isfinite(report['total_loss']),'Finite total required')
    require(type(report['ce']) is float and report['ce']==report['total_loss']/target['targets'],'CE differs')
    require(type(report['bits_per_byte']) is float and report['bits_per_byte']==report['total_loss']/math.log(2)/target['target_bytes'],'Byte score differs')


def evaluate(binding,device='mps'):
    require(device in ('cpu','mps'),'Declared CPU/MPS device required')
    model,data,cfg=load(binding,device)
    try:
        report=dict(binding=copy.deepcopy(binding),**score(model,data));validate_report(report,binding,data);artifacts(binding)
        return report
    finally:del model,data,cfg
