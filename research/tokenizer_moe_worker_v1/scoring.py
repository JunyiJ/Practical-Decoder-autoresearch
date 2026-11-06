"""Own model-policy scoring, exact original34278 native targets, router aux excluded."""
import copy
import math
import tiktoken
import torch
from torch.nn import functional as F
from research.body_allocation_model_v1.contract import require
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest
from research.tokenizer_moe_model_v1.model import GPT
from research.tokenizer_moe_resume_v1.data import architecture,load


def target_contract(model,data,cfg):
    require(type(model) is GPT and model.signature==architecture(cfg),'Genuine own MoE/dense model/config required');model.validate()
    require(not model.training and model.signature['context']==128 and next(model.parameters()).dtype==torch.float32,'Eval FP32/context128 required')
    expected=load(cfg,'cpu');ids=data.val_data
    require(data.data_identity==expected.data_identity and data.tokenizer_contract==expected.tokenizer_contract and set(data.splits)=={'train','val'},'Original data/split identity required')
    require(ids.device.type=='cpu' and ids.dtype==torch.int64 and ids.ndim==1 and len(ids)==34279,'Complete original native validation IDs required')
    values=ids.tolist();require(digest(values)==expected.data_identity['validation_ids_sha256'] and digest(values[1:])==expected.data_identity['validation_target_ids_sha256'],'Validation target ID bytes differ')
    enc=tiktoken.get_encoding('gpt2');n=34278
    return dict(policy='tokenizer_moe_native_validation_v1',validation_tokens=34279,targets=n,ids_sha256=digest(values),target_ids_sha256=digest(values[1:]),
                target_bytes=sum(len(enc.decode_single_token_bytes(i)) for i in values[1:]),window_counts=[min(128,n-i) for i in range(0,n,128)],
                block_counts=[min(2048,n-i) for i in range(0,n,2048)],tokenizer=copy.deepcopy(data.tokenizer_contract),context=128,first_target_offset=1,tail_retained=True,holdout_scored=False)


def score(model,data,cfg):
    target=target_contract(model,data,cfg);ids=data.val_data;device=next(model.parameters()).device;losses=[]
    with torch.no_grad():
        for start in range(0,34278,128):
            n=min(128,34278-start);x=ids[start:start+n].unsqueeze(0).to(device);y=ids[start+1:start+n+1].to(device)
            logits,_=model(x,include_aux_loss=False)
            require(logits.shape==(1,n,50257) and logits.dtype==torch.float32,'Native FP32 vocabulary logits required')
            values=F.cross_entropy(logits[0],y,reduction='none').to('cpu').to(torch.float64)
            require(bool(torch.isfinite(values).all()) and bool((values>=0).all()),'Finite nonnegative token losses required');losses.append(values)
    loss=torch.cat(losses);require(loss.numel()==34278,'Complete target coverage required')
    total=float(loss.sum());report=dict(target=target,window_loss_sums=[float(v.sum()) for v in losses],block_loss_sums=[float(loss[i:i+2048].sum()) for i in range(0,34278,2048)],
        total_loss=total,ce=total/34278,bits_per_byte=total/math.log(2)/target['target_bytes'],loss_dtype='FP32_cross_entropy_then_CPU_FP64_aggregation',holdout_scored=False,aux_loss_included=False)
    validate_report(report,target);return report


def validate_report(report,target):
    require(type(report) is dict and set(report)=={'target','window_loss_sums','block_loss_sums','total_loss','ce','bits_per_byte','loss_dtype','holdout_scored','aux_loss_included'},'Exact own native score report required')
    require(report['target']==target and target['targets']==34278 and report['holdout_scored'] is False and report['aux_loss_included'] is False,'Complete native CE-only target required')
    require(report['loss_dtype']=='FP32_cross_entropy_then_CPU_FP64_aggregation','Exact score arithmetic required')
    require(type(report['total_loss']) is float and math.isfinite(report['total_loss']) and report['total_loss']>=0,'Finite total loss required')
    for name,counts in [('window_loss_sums',target['window_counts']),('block_loss_sums',target['block_counts'])]:
        values=report[name];require(type(values) is list and len(values)==len(counts) and all(type(v) is float and math.isfinite(v) and v>=0 for v in values),'Complete finite windows/blocks required')
        require(abs(math.fsum(values)-report['total_loss'])<1e-8,'Conserved target loss required')
    require(type(report['ce']) is float and report['ce']==report['total_loss']/34278,'CE normalization differs')
    require(type(report['bits_per_byte']) is float and report['bits_per_byte']==report['total_loss']/math.log(2)/target['target_bytes'],'Native byte normalization differs')
