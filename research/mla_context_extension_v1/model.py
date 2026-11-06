"""Same learned incumbent equations; new explicit256-context inference contract."""
import copy,hashlib,json,math
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.contract import model_config,signature as original_signature,canonical,require
from research.block_norm_v1.runtime.attention.mla import MLA as FrozenMLA
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from research.ffn_structure_v1.runtime.models.mlp import SwiGLU
from research.final_ln_scale_v1.component import FinalLayerNorm

POLICY='incumbent_mla_context256_inference_only_v1'
def config():
    c=model_config('incumbent');c.update(tokenizer_architecture_policy=POLICY,block_size=256,
        inference_only=True,inference_context_policy=POLICY)
    c['embedding_backward']=dict(policy='inference_only_native_lookup_v1',width=256,max_tokens=2048)
    return c
def signature(cfg):
    require(canonical(cfg)==canonical(config()),'Exact new inference-only policy required')
    return dict(policy=POLICY,width=256,layers=4,hidden=672,heads=8,latent=64,head_dim=32,vocab=50257,
        max_context=256,parameters=28492416,parameter_tensors=88,inference_only=True,
        cfg_sha256=hashlib.sha256(canonical(cfg).encode()).hexdigest())

class Attention(FrozenMLA):
    def __init__(self):
        # A new explicit constructor, not a fake cfg sent through an old guard.
        nn.Module.__init__(self);self.dim=256;self.dimc_q=self.dimc_kv=64
        self.n_heads=8;self.head_dim=self.head_dim_r=32;self.attention_logit_scale=.5;self.causal=True;self.max_seq_len=256
        self.dkv=nn.Linear(256,64);self.uk=nn.Linear(64,256);self.uv=nn.Linear(64,256)
        self.dq=nn.Linear(256,64);self.uq=nn.Linear(64,256);self.qr=nn.Linear(64,256)
        self.kr=nn.Linear(256,32);self.out=nn.Linear(256,256)
        self.attn_dropout=nn.Dropout(0.);self.out_dropout=nn.Dropout(0.)
        self.register_buffer('causal_mask',torch.tril(torch.ones(256,256,dtype=torch.bool)),persistent=False)

class Block(nn.Module):
    def __init__(self,index):
        super().__init__();self.layer_index=index
        self.ln1=nn.RMSNorm(256,eps=2**-23);self.ln2=nn.RMSNorm(256,eps=2**-23)
        self.attn=Attention();self.mlp=SwiGLU(256,672,0.)
        cos,sin=precompute_freqs_cis(32,256);self.register_buffer('freq_cos',cos);self.register_buffer('freq_sin',sin)
    def forward(self,x,cache,start):
        y,_=self.attn(self.ln1(x),cache=cache,start_pos=start,freq_cos=self.freq_cos,freq_sin=self.freq_sin)
        x=x+y;return x+self.mlp(self.ln2(x))

class Cache(nn.Module):
    def __init__(self,spec,index,batch,device):
        super().__init__();require(type(batch) is int and batch in (1,8),'Declared inference batches required')
        self.signature=copy.deepcopy(spec);self.layer_index=index;self.position=0
        self.register_buffer('c_cache',torch.zeros(batch,256,64,dtype=torch.float32,device=device))
        self.register_buffer('r_cache',torch.zeros(batch,256,32,dtype=torch.float32,device=device))
    def update(self,c,r,start):
        require(start==self.position and c.ndim==r.ndim==3 and c.shape[:2]==r.shape[:2]
            and tuple(c.shape[::2])==(self.c_cache.shape[0],64) and r.shape[2]==32
            and c.dtype==r.dtype==torch.float32 and c.device==r.device==self.c_cache.device,'Cache update differs')
        end=start+c.shape[1];require(end<=256,'Cache overflow')
        self.c_cache[:,start:end]=c;self.r_cache[:,start:end]=r;self.position=end
        return self.c_cache[:,:end],self.r_cache[:,:end]
    def reset(self):self.c_cache.zero_();self.r_cache.zero_();self.position=0

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__();require(torch.get_default_dtype()==torch.float32,'FP32 required');self.cfg=copy.deepcopy(cfg);self.spec=signature(cfg)
        self.transformer=nn.ModuleDict(dict(wte=nn.Embedding(50257,256),h=nn.ModuleList([Block(i) for i in range(4)]),ln_f=FinalLayerNorm(256,'learned')))
        self.lm_head=nn.Linear(256,50257,bias=False);self.width_mult=4.;self.checkpoint_sha=None;self._parameter_stamp=None;self._buffer_stamp=None
        with torch.no_grad():
            self.lm_head.weight.mul_(2.)
            for m in self.transformer.h.modules():
                if isinstance(m,nn.Linear) and m.bias is not None:m.bias.mul_(2.)
        nn.Module.train(self,False);self.validate()
    def train(self,mode=True):
        require(mode is False,'Inference-only model cannot train');return nn.Module.train(self,False)
    def stamp(self):return tuple((n,id(p),p.untyped_storage().data_ptr(),p._version,p.dtype,p.device,tuple(p.shape)) for n,p in self.named_parameters())
    def buffer_stamp(self):return tuple((n,id(p),p.untyped_storage().data_ptr(),p._version,p.dtype,p.device,tuple(p.shape)) for n,p in self.named_buffers())
    def validate(self):
        require(signature(self.cfg)==self.spec and not self.training,'Immutable inference cfg/eval required')
        p=list(self.parameters());require(len(p)==88 and sum(v.numel() for v in p)==28492416
            and all(v.dtype==torch.float32 for v in p) and len({v.device for v in p})==1,'Exact original FP32 learned inventory required')
        require(len({v.untyped_storage().data_ptr() for v in p})==88,'Parameter alias forbidden')
        if self._parameter_stamp is not None:require(self.stamp()==self._parameter_stamp,'Loaded inference weights were mutated or moved')
        if self._buffer_stamp is not None:require(self.buffer_stamp()==self._buffer_stamp,'Loaded inference buffers were mutated or moved')
        require(self.width_mult==4. and type(self.transformer.wte) is nn.Embedding and tuple(self.transformer.wte.weight.shape)==(50257,256)
            and self.lm_head.bias is None and tuple(self.lm_head.weight.shape)==(50257,256),'Embedding/readout differs')
        norm=self.transformer.ln_f;require(type(norm) is FinalLayerNorm and norm.width==256 and norm.kind=='learned' and norm.eps==1e-5,'Final learned norm differs')
        require(len(self.transformer.h)==4,'Exact incumbent depth required')
        for i,b in enumerate(self.transformer.h):
            require(type(b) is Block and b.layer_index==i,'Block layout differs')
            for norm in (b.ln1,b.ln2):require(type(norm) is nn.RMSNorm and norm.eps==2**-23 and norm.normalized_shape==(256,),'Original block RMSNorm required')
            a=b.attn;require(type(a) is Attention and a.dim==256 and a.dimc_q==a.dimc_kv==64 and a.n_heads==8
                and a.head_dim==a.head_dim_r==32 and a.attention_logit_scale==.5 and a.causal and a.max_seq_len==256
                and a.attn_dropout.p==a.out_dropout.p==0. and a.causal_mask.shape==(256,256),'Explicit extended attention differs')
            for n,shape in dict(dkv=(64,256),uk=(256,64),uv=(256,64),dq=(64,256),uq=(256,64),qr=(256,64),kr=(32,256),out=(256,256)).items():
                layer=getattr(a,n);require(type(layer) is nn.Linear and tuple(layer.weight.shape)==shape and layer.bias is not None and tuple(layer.bias.shape)==(shape[0],),'Attention projection differs')
            require(type(b.mlp) is SwiGLU and b.mlp.dim==256 and b.mlp.hidden_dim==672 and b.mlp.gate_activation=='silu' and b.mlp.hidden_norm is None and b.mlp.dropout.p==0.,'Original SwiGLU required')
            for name,shape in dict(linear_gate=(672,256),linear_up=(672,256),linear_down=(256,672)).items():
                layer=getattr(b.mlp,name);require(type(layer) is nn.Linear and tuple(layer.weight.shape)==shape and layer.bias is None,'FFN projection differs')
            require(all(tuple(t.shape)==(256,16) and t.dtype==torch.float32 and t.device==p[0].device for t in (b.freq_cos,b.freq_sin)),'Analytic extended RoPE layout differs')
    def cache_signature(self):
        require(self.checkpoint_sha is not None,'Verified fixed H200 required before cache construction');return dict(**self.spec,checkpoint_sha256=self.checkpoint_sha)
    def caches(self,batch):
        self.validate();return [Cache(self.cache_signature(),i,batch,self.lm_head.weight.device) for i in range(4)]
    def validate_caches(self,caches,batch,start):
        require(type(caches) in (list,tuple) and len(caches)==4,'Exact new cache layers required');tensors=[]
        for i,c in enumerate(caches):
            require(type(c) is Cache and c.signature==self.cache_signature() and c.layer_index==i and type(c.position) is int and c.position==start,'Foreign checkpoint/context cache rejected')
            require(tuple(c.c_cache.shape)==(batch,256,64) and tuple(c.r_cache.shape)==(batch,256,32),'Cache geometry differs')
            for t in (c.c_cache,c.r_cache):require(t.dtype==torch.float32 and t.device==self.lm_head.weight.device and t.is_contiguous() and not t.requires_grad,'Cache dtype/device/layout differs');tensors.append(t)
        ptrs=[t.untyped_storage().data_ptr() for t in tensors];forbidden={p.untyped_storage().data_ptr() for p in (*self.parameters(),*self.buffers())}
        require(len(set(ptrs))==8 and not set(ptrs).intersection(forbidden),'Cache aliases forbidden')
    def forward(self,idx,*,cache=None,start_pos=0):
        self.validate();require(not torch.is_grad_enabled() and self.checkpoint_sha is not None,'Verified inference only, gradients forbidden')
        require(not torch.is_autocast_enabled(self.lm_head.weight.device.type),'Autocast forbidden')
        require(idx.dtype==torch.int64 and idx.ndim==2 and idx.shape[0] in (1,8) and 0<idx.shape[1]<=256
            and idx.device==self.lm_head.weight.device and type(start_pos) is int and 0<=start_pos and start_pos+idx.shape[1]<=256,'Bounded inference token layout required')
        require(not bool((idx<0).any()) and not bool((idx>=50257).any()),'Token outside vocabulary')
        if cache is None:require(start_pos==0,'Offset requires caches')
        else:self.validate_caches(cache,idx.shape[0],start_pos)
        x=self.transformer.wte(idx)
        for i,b in enumerate(self.transformer.h):x=b(x,None if cache is None else cache[i],start_pos)
        return self.lm_head(self.transformer.ln_f(x))/self.width_mult

def import_original_h200(model,saved,checkpoint_sha):
    require(model.checkpoint_sha is None and isinstance(checkpoint_sha,str) and len(checkpoint_sha)==64 and all(c in '0123456789abcdef' for c in checkpoint_sha),'Single verified checkpoint import required')
    require(saved.get('kind')=='h200_body_allocation_v1','Only original inference-only H200 can import; raw/optimizer resume forbidden')
    spec=original_signature(saved['cfg']['model']);require(spec['arm']=='incumbent' and saved['architecture']==spec and saved['step']==1200,'Original accepted incumbent H200 required')
    current=model.state_dict();old=saved['model_state'];require(old.keys()==current.keys(),'Exact original tensor inventory required')
    parameters=dict(model.named_parameters());cos,sin=precompute_freqs_cis(32,256)
    avg=saved.get('averaged_weights',{});require(avg.get('half_life')==200 and avg.get('snapshots')==41 and avg.get('purpose')=='inference_only_not_optimizer_resume'
        and avg.get('parameter_names')==list(parameters) and avg.get('spec')==dict(kind='spaced_tail_ema',total_steps=1200,start_step=800,interval=10,half_lives=[200]),'Original complete H200 provenance required')
    for n,t in old.items():
        require(type(t) is torch.Tensor and t.dtype==torch.float32 and bool(torch.isfinite(t).all()),'Invalid original FP32 state')
        if n in parameters:require(t.shape==parameters[n].shape,'Learned parameter shape differs')
        else:
            require(n.endswith(('.freq_cos','.freq_sin')) and t.shape==(128,16),'Only analytic RoPE buffers may extend')
            expected=cos if n.endswith('.freq_cos') else sin;require(torch.equal(t.cpu(),expected[:128]),'Original analytic RoPE prefix differs')
            require(torch.equal(current[n].cpu(),expected),'New analytic extended RoPE differs')
    with torch.no_grad():
        for n,p in parameters.items():p.copy_(old[n].to(p.device))
    model.checkpoint_sha=checkpoint_sha;model._parameter_stamp=model.stamp();model._buffer_stamp=model.buffer_stamp();model.validate()
