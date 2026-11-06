"""Same incumbent parameterization; one explicit training-only input dropout site."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.contract import model_config,signature as base_signature,canonical,require,fingerprint
from research.body_allocation_model_v1.model import GPT as Original,Embedding,Block,Attention,Cache
from research.ffn_structure_v1.runtime.models.mlp import SwiGLU
from research.final_ln_scale_v1.component import FinalLayerNorm
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis

POLICY='tokenizer_embedding_input_feature_dropout_v1'
def config(probability):
    require(type(probability) is float and probability in (0.,.1),'Exactly zero/.1 input dropout required')
    return dict(policy=POLICY,probability=probability,base_model=model_config('incumbent'),
                mask_policy='independent_per_batch_token_channel_inverted_bernoulli',location='after_lookup_before_first_residual_block')
def signature(cfg):
    require(type(cfg) is dict and canonical(cfg)==canonical(config(cfg.get('probability'))),'Exact explicit input-dropout contract required')
    return dict(base_signature(cfg['base_model']),policy=POLICY,input_dropout=cfg['probability'],
                input_dropout_fingerprint=fingerprint(cfg))

class InputDropout(nn.Module):
    def __init__(self,p):
        super().__init__();require(type(p) is float and p in (0.,.1),'Finite input dropout probability required');self.p=p
    def forward(self,x):
        require(type(self.p) is float and self.p in (0.,.1) and x.dtype==torch.float32,'Exact FP32 input dropout required')
        # These identity paths perform no stochastic operation/RNG advance.
        if not self.training or self.p==0.:return x
        return F.dropout(x,p=self.p,training=True,inplace=False)

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=copy.deepcopy(cfg);self.signature=signature(cfg)
        # Construct a genuine original model with its genuine old configuration.
        # Reuse its exact modules; do not pass the new policy through old guards.
        base=Original(cfg['base_model']);self.transformer=base.transformer;self.lm_head=base.lm_head
        self.width_mult=base.width_mult;self.input_dropout=InputDropout(cfg['probability']);self.validate()
    def validate(self):
        s=signature(self.cfg);require(self.signature==s,'Input dropout signature changed')
        require(type(self.input_dropout) is InputDropout and self.input_dropout.p==s['input_dropout']
                and self.input_dropout.training==self.training,'Input dropout site/mode differs')
        p=list(self.parameters());require(len(p)==88 and sum(x.numel() for x in p)==28492416
            and all(x.dtype==torch.float32 and x.requires_grad for x in p) and len({x.device for x in p})==1,'Original FP32 learned inventory required')
        require(len({x.untyped_storage().data_ptr() for x in p})==88,'Parameter alias forbidden')
        require(type(self.transformer.wte) is Embedding and self.transformer.wte.weight.shape==(50257,256)
                and self.lm_head.weight.shape==(50257,256) and self.lm_head.bias is None and self.width_mult==4.,'Original tables/readout required')
        require(len(self.transformer.h)==4,'Original four blocks required')
        f=self.transformer.ln_f;require(type(f) is FinalLayerNorm and f.width==256 and f.kind=='learned' and f.eps==1e-5,'Original final norm required')
        for i,b in enumerate(self.transformer.h):
            require(type(b) is Block and b.layer_index==i and b.training==self.training,'Original serial block/mode required')
            for norm in (b.ln1,b.ln2):require(type(norm) is nn.RMSNorm and norm.normalized_shape==(256,) and norm.eps==2**-23,'Original RMSNorm required')
            a=b.attn;require(type(a) is Attention and a.dim==256 and a.dimc_q==a.dimc_kv==64 and a.n_heads==8
                and a.head_dim==a.head_dim_r==32 and a.attention_logit_scale==.5 and a.causal and a.max_seq_len==128
                and a.attn_dropout.p==a.out_dropout.p==0.,'Original unperturbed attention required')
            for n,shape in dict(dkv=(64,256),uk=(256,64),uv=(256,64),dq=(64,256),uq=(256,64),qr=(256,64),kr=(32,256),out=(256,256)).items():
                layer=getattr(a,n);require(type(layer) is nn.Linear and tuple(layer.weight.shape)==shape and layer.bias is not None and tuple(layer.bias.shape)==(shape[0],),'Original affine attention required')
            require(type(b.mlp) is SwiGLU and b.mlp.dim==256 and b.mlp.hidden_dim==672 and b.mlp.gate_activation=='silu'
                and b.mlp.hidden_norm is None and b.mlp.dropout.p==0.,'Original unperturbed SwiGLU required')
            for n,shape in dict(linear_gate=(672,256),linear_up=(672,256),linear_down=(256,672)).items():
                layer=getattr(b.mlp,n);require(type(layer) is nn.Linear and tuple(layer.weight.shape)==shape and layer.bias is None,'Original FFN projection required')
            require(all(tuple(t.shape)==(128,16) and t.dtype==torch.float32 and t.device==p[0].device for t in (b.freq_cos,b.freq_sin)),'Original RoPE layout required')
    def caches(self,batch,device=None):
        self.validate();return [Cache(self.signature,i,batch,device or self.lm_head.weight.device) for i in range(4)]
    def validate_caches(self,caches,batch,start):
        require(type(caches) in (list,tuple) and len(caches)==4,'Four own-policy caches required');tensors=[]
        for i,c in enumerate(caches):
            require(type(c) is Cache and c.signature==self.signature and c.layer_index==i and type(c.position) is int and c.position==start,'Foreign policy/cache position')
            require(tuple(c.c_cache.shape)==(batch,128,64) and tuple(c.r_cache.shape)==(batch,128,32),'Original cache shape required')
            for t in (c.c_cache,c.r_cache):require(t.dtype==torch.float32 and t.device==self.lm_head.weight.device and t.is_contiguous() and not t.requires_grad,'Cache storage differs');tensors.append(t)
        ptrs=[t.untyped_storage().data_ptr() for t in tensors];forbidden={t.untyped_storage().data_ptr() for t in (*self.parameters(),*self.buffers())}
        require(len(set(ptrs))==8 and not set(ptrs).intersection(forbidden),'Cache aliases forbidden')
    def forward(self,idx,targets=None,cache=None,start_pos=0,return_loss_breakdown=False):
        self.validate();require(not torch.is_autocast_enabled(self.lm_head.weight.device.type),'Autocast forbidden')
        require(idx.dtype==torch.int64 and idx.ndim==2 and min(idx.shape)>0 and idx.numel()<=1024 and idx.device==self.lm_head.weight.device,'Original bounded token layout required')
        require(type(start_pos) is int and start_pos>=0 and start_pos+idx.shape[1]<=128,'Original context bound required')
        require(not bool((idx<0).any()) and not bool((idx>=50257).any()),'Token outside vocabulary')
        if cache is None:require(start_pos==0,'Offset requires own-policy caches')
        else:
            require(not self.training and not torch.is_grad_enabled(),'Cached use is eval/no-grad only')
            self.validate_caches(cache,idx.shape[0],start_pos)
        x=self.input_dropout(self.transformer.wte(idx))
        for i,b in enumerate(self.transformer.h):x=b(x,None if cache is None else cache[i],start_pos)
        logits=self.lm_head(self.transformer.ln_f(x))/self.width_mult;loss=None
        if targets is not None:
            require(targets.shape==idx.shape and targets.dtype==torch.int64 and targets.device==idx.device,'Targets differ')
            loss=F.cross_entropy(logits.reshape(-1,50257),targets.reshape(-1))
        if return_loss_breakdown:return logits,loss,dict(ce_loss=loss,aux_loss=None,total_loss=loss)
        return logits,loss

def load_weights(model,saved):
    model.validate();require(saved.get('kind') in ('raw_embedding_input_dropout_v1','h200_embedding_input_dropout_v1')
        and saved.get('architecture')==model.signature,'Foreign checkpoint policy rejected')
    old=saved.get('model_state');current=model.state_dict();require(type(old) in (dict,type(current)) and old.keys()==current.keys(),'Exact state inventory required')
    for n,t in old.items():
        require(type(t) is torch.Tensor and t.shape==current[n].shape and t.dtype==current[n].dtype and bool(torch.isfinite(t).all()),'Invalid state tensor '+n)
        if n.endswith(('freq_cos','freq_sin')):
            cos,sin=precompute_freqs_cis(32,128);require(torch.equal(t.cpu(),cos if n.endswith('freq_cos') else sin),'RoPE values differ')
    model.load_state_dict(old)
