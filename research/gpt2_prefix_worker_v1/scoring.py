"""Own-policy FP32 token loss over an identical bound raw-byte target suffix."""
import copy,math
import torch
import torch.nn.functional as F
from research.body_allocation_model_v1.contract import require,canonical,fingerprint
from research.gpt2_prefix_data_v1.data import load,architecture
from research.gpt2_prefix_model_v2.model import GPT
from research.tokenizer_common_bytes_v1.metric import windows,loss_summary

def target_contract(model,data,cfg):
    require(type(model) is GPT and model.signature==architecture(cfg),'Genuine own-model/config contract required');model.validate()
    require(not model.training and next(model.parameters()).dtype==torch.float32,'Own-policy eval FP32 required')
    expected=load(cfg,'cpu')
    require(data.data_identity==expected.data_identity and canonical(data.common_byte_plan)==canonical(expected.common_byte_plan)
        and data.vocab_size==model.signature['vocab_size'],'Actual corpus/codec/common-byte target identity differs')
    p=expected.common_byte_plan
    require(p['target_bytes']==111475 and p['target_sha256']=='ad7da8cdcacc24b5312461c23f2d00aead59d544fcc2fad4fb2de40da4964fe8','Original common byte sequence required')
    return dict(policy=p['policy'],plan_sha256=fingerprint(p),tokenizer=copy.deepcopy(p['codec']),target_sha256=p['target_sha256'],
                target_bytes=p['target_bytes'],target_tokens=p['target_tokens'],prefix_bytes=p['prefix_bytes'],context=p['context'])

def score(model,data,cfg):
    target=target_contract(model,data,cfg);plan=data.common_byte_plan;device=next(model.parameters()).device;losses=[];sums=[];offset=0
    with torch.no_grad():
        for x,y,start in windows(plan):
            require(start==offset and len(x)==len(y) and 0<len(y)<=128,'Disjoint complete target windows required')
            indices=torch.tensor([x],dtype=torch.int64,device=device);targets=torch.tensor(y,dtype=torch.int64,device=device)
            logits,_=model(indices);require(logits.shape==(1,len(y),model.signature['vocab_size']) and logits.dtype==torch.float32,'Effective vocabulary/FP32 logit layout differs')
            row=F.cross_entropy(logits[0],targets,reduction='none').to('cpu').to(torch.float64)
            require(bool(torch.isfinite(row).all()) and bool((row>=0).all()),'Finite nonnegative target losses required')
            values=row.tolist();losses.extend(values);sums.append(math.fsum(values));offset+=len(values)
    require(offset==target['target_tokens'],'Complete common-byte token target coverage required')
    result=dict(target=target,window_loss_sums=sums,**loss_summary(plan,losses),loss_dtype='FP32_cross_entropy_then_CPU_FP64_aggregation',holdout_scored=False)
    validate_report(result,target,plan);return result

def validate_report(report,target,plan):
    require(report['target']==target and report['target_sha256']==target['target_sha256'] and report['target_bytes']==target['target_bytes']
        and report['target_tokens']==target['target_tokens'] and report['holdout_scored'] is False,'Score coverage/identity differs')
    require(len(report['window_loss_sums'])==(target['target_tokens']+127)//128,'Complete window sum coverage required')
    require(all(type(x) in (float,int) and math.isfinite(x) and x>=0 for x in report['window_loss_sums']),'Finite window losses required')
    total=math.fsum(report['window_loss_sums']);require(abs(total-report['total_nats'])<max(1e-8,total*1e-12),'Complete window loss conservation differs')
    require(math.isfinite(report['bits_per_byte']) and abs(report['bits_per_byte']-total/(math.log(2)*target['target_bytes']))<1e-12,'Common-byte normalization differs')
    require(set(report['raw_byte_blocks'])=={'4096','8192','16384'},'Exact common raw-byte block inventory required')
    for width,block in report['raw_byte_blocks'].items():
        counts=[min(int(width),target['target_bytes']-i) for i in range(0,target['target_bytes'],int(width))]
        require(block['byte_counts']==counts and len(block['loss_sums'])==len(counts),'Byte block coverage differs')
        require(all(type(x) in (float,int) and math.isfinite(x) and x>=0 for x in block['loss_sums']) and abs(math.fsum(block['loss_sums'])-total)<max(1e-8,total*1e-12),'Byte block loss conservation differs')
    return report
