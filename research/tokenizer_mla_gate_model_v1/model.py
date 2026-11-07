import copy,math
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.contract import model_config as native_config
from research.body_allocation_model_v1.model import GPT as Native,Embedding,Attention as NativeAttention,Cache as NativeCache
from research.ffn_structure_v1.runtime.models.mlp import SwiGLU
from research.final_ln_scale_v1.component import FinalLayerNorm
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis,apply_rotary_emb
from research.tokenizer_mla_gate_v1.component import ElementwiseOutputGate
from .contract import signature,require

class Attention(NativeAttention):
    def __init__(self,original,kind):
        nn.Module.__init__(self)
        for name in ('dim','dimc_q','dimc_kv','n_heads','head_dim','head_dim_r','attention_logit_scale','causal','max_seq_len','dkv','uk','uv','dq','uq','qr','kr','out','attn_dropout','out_dropout'):setattr(self,name,getattr(original,name))
        self.register_buffer('causal_mask',original.causal_mask,persistent=False)
        self.output_gate=ElementwiseOutputGate(kind,256,8,32)

    def forward(self,x,cache=None,start_pos=0,freq_cos=None,freq_sin=None):
        b,s,d=x.size();c_q=self.dq(x);c_kv=self.dkv(x)
        c_q_r=self.qr(c_q).view(b,s,self.n_heads,-1).transpose(1,2);k_r=self.kr(x).unsqueeze(1)
        if freq_cos is not None and freq_sin is not None:
            cos=freq_cos[start_pos:start_pos+s].to(device=c_q.device,dtype=c_q.dtype);sin=freq_sin[start_pos:start_pos+s].to(device=c_q.device,dtype=c_q.dtype)
            r_q=apply_rotary_emb(c_q_r,cos,sin);r_k=apply_rotary_emb(k_r,cos,sin)
        else:r_q=c_q_r;r_k=k_r
        r_k=r_k.squeeze(1)
        if cache is not None:c_kv,r_k=cache.update(c_kv,r_k,start_pos)
        total=c_kv.size(1);q_base=self.uq(c_q).view(b,s,self.n_heads,self.head_dim).transpose(1,2);k_base=self.uk(c_kv).view(b,total,self.n_heads,self.head_dim).transpose(1,2);v=self.uv(c_kv).view(b,total,self.n_heads,self.head_dim).transpose(1,2)
        q=torch.cat([q_base,r_q],dim=-1);k_rotary=r_k.unsqueeze(1).expand(-1,self.n_heads,-1,-1);k=torch.cat([k_base,k_rotary],dim=-1)
        att=(q@k.transpose(-2,-1))/math.sqrt(self.head_dim+self.head_dim_r)
        if self.attention_logit_scale!=1.:att=att*self.attention_logit_scale
        if self.causal:
            mask=self._get_causal_mask(k.size(-2),x.device);mask=mask[start_pos:start_pos+s,:k.size(-2)];att=att.masked_fill(~mask,float('-inf'))
        att=self.attn_dropout(torch.softmax(att,dim=-1));y=att@v
        y=self.output_gate(x,y)
        y=y.transpose(1,2).contiguous().view(b,s,d)
        return self.out_dropout(self.out(y)),None

class Block(nn.Module):
    def __init__(self,original,arm):
        super().__init__();self.layer_index=original.layer_index;self.ln1=original.ln1;self.ln2=original.ln2
        self.attn=original.attn if arm=='none' else Attention(original.attn,arm)
        self.mlp=original.mlp
        self.register_buffer('freq_cos',original.freq_cos);self.register_buffer('freq_sin',original.freq_sin)
    def forward(self,x,cache,start):
        y,_=self.attn(self.ln1(x),cache=cache,start_pos=start,freq_cos=self.freq_cos,freq_sin=self.freq_sin);x=x+y
        return x+self.mlp(self.ln2(x))

class Cache(NativeCache):
    """Own signature/type,same native compressed-KV and rotary-buffer equations."""

class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=copy.deepcopy(cfg);self.signature=signature(cfg);arm=self.signature['arm']
        base=Native(native_config('incumbent'))
        self.transformer=nn.ModuleDict(dict(wte=base.transformer.wte,h=nn.ModuleList([Block(b,arm) for b in base.transformer.h]),ln_f=base.transformer.ln_f));self.lm_head=base.lm_head;self.width_mult=4.;self.validate()
    def validate(self):
        s=self.signature;require(signature(self.cfg)==s,'Mutable own gate model policy')
        params=list(self.parameters());device=params[0].device
        require(all(p.dtype==torch.float32 and p.requires_grad and p.device==device for p in params),'FP32 learned single-device parameters required')
        require(sum(p.numel() for p in params)==s['parameters'] and len(params)==s['muon_groups']+s['adam_groups'],'Own gate parameter inventory differs')
        require(len({id(p) for p in params})==len(params) and len({p.untyped_storage().data_ptr() for p in params})==len(params),'Aliased learned parameters')
        require(self.width_mult==4. and len(self.transformer.h)==4,'Body/readout scale differs')
        require(type(self.transformer.wte) is Embedding and self.transformer.wte.weight.shape==(50257,256),'Native embedding differs')
        require(type(self.lm_head) is nn.Linear and self.lm_head.weight.shape==(50257,256) and self.lm_head.bias is None,'Native readout differs')
        norm=self.transformer.ln_f;require(type(norm) is FinalLayerNorm and norm.width==256 and norm.kind=='learned' and norm.eps==1e-5 and norm.weight.shape==norm.bias.shape==(256,),'Final normalization differs')
        for i,b in enumerate(self.transformer.h):
            require(type(b) is Block and b.layer_index==i,'Own block layout differs')
            for n in (b.ln1,b.ln2):require(type(n) is nn.RMSNorm and n.normalized_shape==(256,) and n.eps==2**-23 and n.elementwise_affine and n.weight.shape==(256,),'Native block norm differs')
            a=b.attn;require(type(a) is (NativeAttention if s['arm']=='none' else Attention),'Own attention type differs')
            require(a.dim==256 and a.dimc_q==a.dimc_kv==64 and a.n_heads==8 and a.head_dim==a.head_dim_r==32 and a.attention_logit_scale==.5 and a.causal and a.max_seq_len==128 and a.attn_dropout.p==a.out_dropout.p==0.,'Native MLA geometry differs')
            for name,shape in dict(dkv=(64,256),uk=(256,64),uv=(256,64),dq=(64,256),uq=(256,64),qr=(256,64),kr=(32,256),out=(256,256)).items():
                m=getattr(a,name);require(type(m) is nn.Linear and m.weight.shape==shape and m.bias is not None and m.bias.shape==(shape[0],),'Native MLA projection differs')
            if s['arm']!='none':
                g=a.output_gate;require(type(g) is ElementwiseOutputGate and (g.kind,g.width,g.heads,g.head_dim)==g._signature==(s['arm'],256,8,32),'Own gate policy differs')
                require((type(g.weight) is nn.Parameter and g.weight.shape==(256,256)) if s['arm']=='sigmoid_elementwise_zero' else g.weight is None,'Elementwise gate parameter differs')
            m=b.mlp;require(type(m) is SwiGLU and m.dim==256 and m.hidden_dim==672 and m.gate_activation=='silu' and m.hidden_norm is None and m.dropout.p==0.,'Native SwiGLU differs')
            for name,shape in [('linear_gate',(672,256)),('linear_up',(672,256)),('linear_down',(256,672))]:require(type(getattr(m,name)) is nn.Linear and getattr(m,name).weight.shape==shape and getattr(m,name).bias is None,'Native FFN projection differs')
            require(all(t.shape==(128,16) and t.dtype==torch.float32 and t.device==device for t in (b.freq_cos,b.freq_sin)),'RoPE buffers differ')
    def caches(self,batch,device=None):
        self.validate();return [Cache(self.signature,i,batch,device or self.lm_head.weight.device) for i in range(4)]
    def validate_caches(self,caches,batch,start):
        require(type(caches) in (list,tuple) and len(caches)==4,'Exact own cache inventory required');tensors=[]
        for i,c in enumerate(caches):
            require(type(c) is Cache and c.signature==self.signature and type(c.layer_index) is int and c.layer_index==i and type(c.position) is int and c.position==start,'Foreign cache signature or position')
            require(c.c_cache.shape==(batch,128,64) and c.r_cache.shape==(batch,128,32),'Cache geometry differs')
            for t in (c.c_cache,c.r_cache):require(t.dtype==torch.float32 and t.device==self.lm_head.weight.device and t.is_contiguous() and not t.requires_grad,'Cache device/dtype differs');tensors.append(t)
        require(len({id(c) for c in caches})==4 and len({t.untyped_storage().data_ptr() for t in tensors})==8,'Aliased cache')
        forbidden={p.untyped_storage().data_ptr() for p in self.parameters()};require(not any(t.untyped_storage().data_ptr() in forbidden for t in tensors),'Cache aliases weights')
    def forward(self,idx,targets=None,cache=None,start_pos=0,include_aux_loss=True,return_loss_breakdown=False,include_training_objective=False):
        self.validate();require(not torch.is_autocast_enabled(self.lm_head.weight.device.type),'Autocast forbidden')
        require(type(idx) is torch.Tensor and idx.dtype==torch.int64 and idx.ndim==2 and 0<idx.shape[0]<=8 and 0<idx.shape[1]<=128 and idx.device==self.lm_head.weight.device,'Bounded device-local native tokens required')
        require(type(start_pos) is int and 0<=start_pos and start_pos+idx.shape[1]<=128 and all(type(v) is bool for v in (include_aux_loss,return_loss_breakdown,include_training_objective)),'Explicit positions/objective flags required')
        require(not bool((idx<0).any()) and not bool((idx>=50257).any()),'Token outside native vocabulary')
        if targets is not None:require(type(targets) is torch.Tensor and targets.shape==idx.shape and targets.dtype==torch.int64 and targets.device==idx.device and not bool((targets<0).any()) and not bool((targets>=50257).any()),'Targets differ')
        if cache is None:require(start_pos==0,'Offset requires caches')
        else:require(not torch.is_grad_enabled(),'Cache is inference only');self.validate_caches(cache,idx.shape[0],start_pos)
        x=self.transformer.wte(idx)
        for i,b in enumerate(self.transformer.h):x=b(x,None if cache is None else cache[i],start_pos)
        logits=self.lm_head(self.transformer.ln_f(x))/4.;loss=F.cross_entropy(logits.reshape(-1,50257),targets.reshape(-1)) if targets is not None else None
        if return_loss_breakdown:return logits,loss,dict(ce_loss=loss,aux_loss=None,total_loss=loss)
        return logits,loss

def load_weights(model,payload):
    model.validate();require(type(payload) is dict and set(payload)=={'kind','architecture','model_state'} and payload['kind']=='tokenizer_mla_gate_weights_v1' and payload['architecture']==model.signature,'Only own exact gate weights may load')
    current=model.state_dict();saved=payload['model_state'];require(type(saved) in (dict,type(current)) and saved.keys()==current.keys(),'Exact learned/buffer inventory required')
    for name,t in saved.items():
        require(type(t) is torch.Tensor and t.shape==current[name].shape and t.dtype==current[name].dtype and bool(torch.isfinite(t).all()),'Finite exact weight tensor required '+name)
        if name.endswith(('freq_cos','freq_sin')):
            cos,sin=precompute_freqs_cis(32,128);require(torch.equal(t.cpu(),cos if name.endswith('freq_cos') else sin),'Exact native RoPE required')
    model.load_state_dict(saved)
