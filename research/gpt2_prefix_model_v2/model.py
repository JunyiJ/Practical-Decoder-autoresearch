"""Explicit effective vocabulary, genuine native construction, matched row subset."""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.contract import model_config,signature as base_signature,canonical,require,fingerprint
from research.body_allocation_model_v1.model import GPT as Original,Embedding,Block,Attention,Cache
from research.ffn_structure_v1.runtime.models.mlp import SwiGLU
from research.final_ln_scale_v1.component import FinalLayerNorm
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis

from .contract import config,signature
from torch.autograd.function import once_differentiable

class _PrefixGrouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weight,indices):
        require(weight.dtype==torch.float32 and tuple(weight.shape)==(8192,256),'Explicit8KFP32embedding required')
        require(indices.dtype==torch.int64 and indices.ndim==2 and 0<indices.numel()<=1024,'Bounded int64 prefix indices required')
        require(weight.device.type=='mps' and indices.device==weight.device,'One MPS device required')
        require(not torch.is_autocast_enabled('mps'),'Autocast forbidden')
        require(not bool((indices<0).any()) and not bool((indices>=8192).any()),'Prefix ID outside vocabulary')
        ctx.save_for_backward(indices);ctx.shape=tuple(weight.shape)
        return F.embedding(indices,weight)
    @staticmethod
    @once_differentiable
    def backward(ctx,upstream):
        (indices,)=ctx.saved_tensors
        require(upstream.dtype==torch.float32 and upstream.device==indices.device,'FP32 device-local upstream required')
        tokens=indices.reshape(-1);unique=torch.unique(tokens,sorted=True)
        membership=(unique[:,None]==tokens[None,:]).to(torch.float32)
        summed=membership@upstream.reshape(-1,ctx.shape[1])
        gradient=torch.zeros(ctx.shape,dtype=torch.float32,device=upstream.device)
        gradient.index_copy_(0,unique,summed)
        require(all(x.device==upstream.device for x in (unique,membership,summed,gradient)),'Prefix reduction left MPS')
        return gradient,None

class PrefixEmbedding(nn.Embedding):
    def forward(self,indices):
        if self.weight.device.type=='mps':return _PrefixGrouped.apply(self.weight,indices)
        return F.embedding(indices,self.weight)

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=copy.deepcopy(cfg);self.signature=signature(cfg)
        # The old constructor sees only its genuine original50257 configuration.
        base=Original(cfg['backbone_template']);v=cfg['effective_vocab_size']
        if v<50257:
            # Additional module construction must not consume native live RNG.
            with torch.random.fork_rng(devices=[]):
                embedding=PrefixEmbedding(v,256);readout=nn.Linear(256,v,bias=False)
            with torch.no_grad():
                embedding.weight.copy_(base.transformer.wte.weight[:v]);readout.weight.copy_(base.lm_head.weight[:v])
            base.transformer.wte=embedding;base.lm_head=readout
        self.transformer=base.transformer;self.lm_head=base.lm_head;self.width_mult=base.width_mult;self.validate()
    def validate(self):
        s=signature(self.cfg);v=s['vocab_size'];require(self.signature==s,'Effective vocabulary signature changed')
        p=list(self.parameters());require(len(p)==88 and sum(x.numel() for x in p)==s['parameters']
            and all(x.dtype==torch.float32 and x.requires_grad for x in p) and len({x.device for x in p})==1,'Original FP32 learned inventory required')
        require(len({x.untyped_storage().data_ptr() for x in p})==88,'Parameter alias forbidden')
        require(type(self.transformer.wte) is (Embedding if v==50257 else PrefixEmbedding) and self.transformer.wte.weight.shape==(v,256)
                and self.lm_head.weight.shape==(v,256) and self.lm_head.bias is None and self.width_mult==4.,'Original tables/readout required')
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
        require(not bool((idx<0).any()) and not bool((idx>=self.signature['vocab_size']).any()),'Token outside vocabulary')
        if cache is None:require(start_pos==0,'Offset requires own-policy caches')
        else:
            require(not self.training and not torch.is_grad_enabled(),'Cached use is eval/no-grad only')
            self.validate_caches(cache,idx.shape[0],start_pos)
        x=self.transformer.wte(idx)
        for i,b in enumerate(self.transformer.h):x=b(x,None if cache is None else cache[i],start_pos)
        logits=self.lm_head(self.transformer.ln_f(x))/self.width_mult;loss=None
        if targets is not None:
            require(targets.shape==idx.shape and targets.dtype==torch.int64 and targets.device==idx.device,'Targets differ')
            require(not bool((targets<0).any()) and not bool((targets>=self.signature['vocab_size']).any()),'Target outside effective vocabulary')
            loss=F.cross_entropy(logits.reshape(-1,self.signature['vocab_size']),targets.reshape(-1))
        if return_loss_breakdown:return logits,loss,dict(ce_loss=loss,aux_loss=None,total_loss=loss)
        return logits,loss

def load_weights(model,saved):
    model.validate();require(saved.get('kind') in ('raw_gpt2_prefix_model_v2','h200_gpt2_prefix_model_v2')
        and saved.get('architecture')==model.signature,'Foreign checkpoint policy rejected')
    old=saved.get('model_state');current=model.state_dict();require(type(old) in (dict,type(current)) and old.keys()==current.keys(),'Exact state inventory required')
    for n,t in old.items():
        require(type(t) is torch.Tensor and t.shape==current[n].shape and t.dtype==current[n].dtype and bool(torch.isfinite(t).all()),'Invalid state tensor '+n)
        if n.endswith(('freq_cos','freq_sin')):
            cos,sin=precompute_freqs_cis(32,128);require(torch.equal(t.cpu(),cos if n.endswith('freq_cos') else sin),'RoPE values differ')
    model.load_state_dict(old)
