"""Same equations and initialization as incumbent, with explicit new geometry."""
import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd.function import once_differentiable
from research.block_norm_v1.runtime.attention.mla import MLA as FrozenMLA
from research.ffn_structure_v1.runtime.models.mlp import SwiGLU
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from research.final_ln_scale_v1.component import FinalLayerNorm
from .contract import signature,require

class _Grouped(torch.autograd.Function):
    @staticmethod
    def forward(ctx,weight,indices):
        require(weight.dtype==torch.float32 and weight.ndim==2 and weight.shape[0]==50257
            and weight.shape[1] in (192,256),'Explicit FP32 allocation embedding required')
        require(indices.dtype==torch.int64 and indices.ndim==2 and 0<indices.numel()<=1024,'Bounded int64 embedding indices required')
        require(weight.device.type=='mps' and indices.device==weight.device,'One MPS device required')
        require(not torch.is_autocast_enabled('mps'),'Autocast forbidden')
        require(not bool((indices<0).any()) and not bool((indices>=50257).any()),'Token outside vocabulary')
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
        require(all(x.device==upstream.device for x in (unique,membership,summed,gradient)),'Embedding reduction left MPS')
        return gradient,None

class Embedding(nn.Embedding):
    def forward(self,indices):
        if self.weight.device.type=='mps':return _Grouped.apply(self.weight,indices)
        return F.embedding(indices,self.weight)

class Attention(FrozenMLA):
    def __init__(self,spec):
        # An explicitly new constructor; never call an old validator with fake cfg.
        nn.Module.__init__(self)
        d,r=spec['width'],spec['latent'];self.dim=d;self.dimc_q=self.dimc_kv=r
        self.n_heads=spec['heads'];self.head_dim=self.head_dim_r=32
        self.attention_logit_scale=.5;self.causal=True;self.max_seq_len=128
        self.dkv=nn.Linear(d,r);self.uk=nn.Linear(r,d);self.uv=nn.Linear(r,d)
        self.dq=nn.Linear(d,r);self.uq=nn.Linear(r,d);self.qr=nn.Linear(r,d)
        self.kr=nn.Linear(d,32);self.out=nn.Linear(d,d)
        self.attn_dropout=nn.Dropout(0.);self.out_dropout=nn.Dropout(0.)
        self.register_buffer('causal_mask',torch.tril(torch.ones(128,128,dtype=torch.bool)),persistent=False)

class Block(nn.Module):
    def __init__(self,spec,index):
        super().__init__();d=spec['width'];self.layer_index=index
        self.ln1=nn.RMSNorm(d,eps=2**-23);self.ln2=nn.RMSNorm(d,eps=2**-23)
        self.attn=Attention(spec);self.mlp=SwiGLU(d,spec['hidden'],0.)
        cos,sin=precompute_freqs_cis(32,128)
        self.register_buffer('freq_cos',cos);self.register_buffer('freq_sin',sin)
    def forward(self,x,cache,start):
        y,_=self.attn(self.ln1(x),cache=cache,start_pos=start,freq_cos=self.freq_cos,freq_sin=self.freq_sin)
        x=x+y
        return x+self.mlp(self.ln2(x))

class Cache(nn.Module):
    def __init__(self,spec,index,batch,device,dtype=torch.float32):
        super().__init__();require(type(batch) is int and batch>0,'Positive cache batch required')
        require(dtype==torch.float32,'Only FP32 cache supported')
        self.signature=copy.deepcopy(spec);self.layer_index=index;self.position=0
        self.register_buffer('c_cache',torch.zeros(batch,128,spec['latent'],device=device,dtype=dtype))
        self.register_buffer('r_cache',torch.zeros(batch,128,32,device=device,dtype=dtype))
    def update(self,c,r,start):
        require(start==self.position and c.ndim==r.ndim==3 and c.shape[:2]==r.shape[:2]
            and c.shape[0]==self.c_cache.shape[0] and c.shape[2]==self.c_cache.shape[2] and r.shape[2]==32
            and c.dtype==r.dtype==self.c_cache.dtype and c.device==r.device==self.c_cache.device,'Cache update differs')
        end=start+c.shape[1];require(end<=128,'Cache overflow')
        self.c_cache[:,start:end,:]=c;self.r_cache[:,start:end,:]=r;self.position=end
        return self.c_cache[:,:end,:],self.r_cache[:,:end,:]
    def reset(self):self.c_cache.zero_();self.r_cache.zero_();self.position=0

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__();require(torch.get_default_dtype()==torch.float32,'FP32 construction required')
        self.cfg=copy.deepcopy(cfg);self.signature=signature(cfg);s=self.signature;d=s['width']
        self.transformer=nn.ModuleDict(dict(wte=Embedding(50257,d),
            h=nn.ModuleList([Block(s,i) for i in range(s['layers'])]),ln_f=FinalLayerNorm(d,'learned')))
        self.lm_head=nn.Linear(d,50257,bias=False);self.width_mult=d/64
        with torch.no_grad():
            self.lm_head.weight.mul_(math.sqrt(self.width_mult))
            for m in self.transformer.h.modules():
                if isinstance(m,nn.Linear) and m.bias is not None:m.bias.mul_(math.sqrt(self.width_mult))
        self.validate()
    def validate(self):
        s=self.signature;require(signature(self.cfg)==s,'Mutable geometry differs')
        p=list(self.parameters());require(all(x.dtype==torch.float32 and x.requires_grad for x in p) and len({x.device for x in p})==1,'FP32 learned single-device parameters required')
        require(sum(x.numel() for x in p)==s['parameters'] and len(p)==s['muon_groups']+s['adam_groups'],'Parameter inventory differs')
        require(len({id(x) for x in p})==len(p) and len({x.untyped_storage().data_ptr() for x in p})==len(p),'Aliased parameters forbidden')
        require(len(self.transformer.h)==s['layers'] and self.width_mult==s['width']/64,'Depth or readout scale differs')
        require(type(self.transformer.wte) is Embedding and tuple(self.transformer.wte.weight.shape)==(50257,s['width']),'Embedding geometry differs')
        require(tuple(self.lm_head.weight.shape)==(50257,s['width']) and self.lm_head.bias is None,'Readout geometry differs')
        f=self.transformer.ln_f;require(type(f) is FinalLayerNorm and f.width==s['width'] and f.kind=='learned' and f.eps==1e-5,'Final norm differs')
        for i,b in enumerate(self.transformer.h):
            require(type(b) is Block and b.layer_index==i and type(b.attn) is Attention and type(b.mlp) is SwiGLU,'Block layout differs')
            for n in (b.ln1,b.ln2):require(type(n) is nn.RMSNorm and n.normalized_shape==(s['width'],) and n.eps==2**-23,'Block norm differs')
            a=b.attn;require(a.dim==s['width'] and a.dimc_q==a.dimc_kv==s['latent'] and a.n_heads==s['heads']
                and a.head_dim==a.head_dim_r==32 and a.attention_logit_scale==.5 and a.causal
                and a.attn_dropout.p==a.out_dropout.p==0.,'Attention geometry/semantics differ')
            d,r=s['width'],s['latent']
            for name,shape in dict(dkv=(r,d),uk=(d,r),uv=(d,r),dq=(r,d),uq=(d,r),qr=(d,r),kr=(32,d),out=(d,d)).items():
                m=getattr(a,name);require(type(m) is nn.Linear and tuple(m.weight.shape)==shape
                    and m.bias is not None and tuple(m.bias.shape)==(shape[0],),'Attention projection differs')
            require(b.mlp.dim==s['width'] and b.mlp.hidden_dim==s['hidden'] and b.mlp.gate_activation=='silu'
                and b.mlp.hidden_norm is None and b.mlp.dropout.p==0.,'FFN geometry/semantics differ')
            for name,shape in dict(linear_gate=(s['hidden'],d),linear_up=(s['hidden'],d),linear_down=(d,s['hidden'])).items():
                m=getattr(b.mlp,name);require(type(m) is nn.Linear and tuple(m.weight.shape)==shape and m.bias is None,'FFN projection differs')
            require(all(tuple(t.shape)==(128,16) and t.dtype==torch.float32 and t.device==p[0].device
                for t in (b.freq_cos,b.freq_sin)),'RoPE buffers differ')
    def caches(self,batch,device=None):
        self.validate();return [Cache(self.signature,i,batch,device or self.lm_head.weight.device) for i in range(self.signature['layers'])]
    def validate_caches(self,caches,batch,start):
        s=self.signature;require(type(caches) in (tuple,list) and len(caches)==s['layers'],'Exact cache layer inventory required')
        tensors=[]
        for i,c in enumerate(caches):
            require(type(c) is Cache and c.signature==s and type(c.layer_index) is int and c.layer_index==i
                and type(c.position) is int and c.position==start,'Foreign cache signature/position')
            require(tuple(c.c_cache.shape)==(batch,128,s['latent']) and tuple(c.r_cache.shape)==(batch,128,32),'Cache shape differs')
            for t in (c.c_cache,c.r_cache):require(t.dtype==torch.float32 and t.device==self.lm_head.weight.device
                and t.is_contiguous() and not t.requires_grad,'Cache dtype/device/layout differs');tensors.append(t)
        require(len({id(x) for x in caches})==len(caches) and len({x.untyped_storage().data_ptr() for x in tensors})==len(tensors),'Aliased caches forbidden')
        forbidden={x.untyped_storage().data_ptr() for x in self.parameters()}
        require(not any(x.untyped_storage().data_ptr() in forbidden for x in tensors),'Cache aliases model parameters')
    def forward(self,idx,targets=None,cache=None,start_pos=0,include_aux_loss=True,return_loss_breakdown=False,include_training_objective=False):
        self.validate();require(not torch.is_autocast_enabled(self.lm_head.weight.device.type),'Autocast forbidden')
        require(idx.dtype==torch.int64 and idx.ndim==2 and min(idx.shape)>0 and idx.device==self.lm_head.weight.device,'Nonempty device-local int64 tokens required')
        require(type(start_pos) is int and 0<=start_pos and start_pos+idx.shape[1]<=128,'Invalid token position')
        require(not bool((idx<0).any()) and not bool((idx>=50257).any()),'Token outside vocabulary')
        if cache is None:require(start_pos==0,'Offset requires caches')
        else:
            require(not torch.is_grad_enabled(),'Cache is inference only')
            self.validate_caches(cache,idx.shape[0],start_pos)
        x=self.transformer.wte(idx)
        for i,b in enumerate(self.transformer.h):x=b(x,None if cache is None else cache[i],start_pos)
        logits=self.lm_head(self.transformer.ln_f(x))/self.width_mult
        loss=None
        if targets is not None:
            require(targets.shape==idx.shape and targets.dtype==torch.int64 and targets.device==idx.device,'Targets differ')
            loss=F.cross_entropy(logits.reshape(-1,50257),targets.reshape(-1))
        if return_loss_breakdown:return logits,loss,dict(ce_loss=loss,aux_loss=None,total_loss=loss)
        return logits,loss

def load_weights(model,payload):
    model.validate();require(type(payload) is dict and payload.get('architecture')==model.signature,'Foreign checkpoint architecture')
    current=model.state_dict();saved=payload.get('model_state')
    require(type(saved) in (dict,type(current)) and saved.keys()==current.keys(),'Checkpoint tensor inventory differs')
    for n,t in saved.items():
        require(type(t) is torch.Tensor and t.shape==current[n].shape and t.dtype==current[n].dtype
            and bool(torch.isfinite(t).all()),'Checkpoint tensor differs: '+n)
        if n.endswith(('freq_cos','freq_sin')):
            cos,sin=precompute_freqs_cis(32,128)
            require(torch.equal(t.cpu(),cos if n.endswith('freq_cos') else sin),'Checkpoint RoPE values differ')
    model.load_state_dict(saved)
