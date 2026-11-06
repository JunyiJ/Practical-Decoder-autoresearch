import copy
import torch
from torch import nn
from torch.nn import functional as F
from research.body_allocation_model_v1.model import GPT as Native, Embedding, Attention, Cache as NativeCache
from research.body_allocation_model_v1.contract import model_config as native_config
from research.ffn_structure_v1.runtime.models.mlp import SwiGLU
from research.final_ln_scale_v1.component import FinalLayerNorm
from research.block_norm_v1.runtime.models.rope import precompute_freqs_cis
from research.tokenizer_moe_component_v1.component import RoutedSwiGLU, Expert
from .contract import signature,require


class Block(nn.Module):
    def __init__(self,original,arm):
        super().__init__()
        self.layer_index=original.layer_index
        self.ln1,self.ln2,self.attn=original.ln1,original.ln2,original.attn
        if arm=='dense':self.mlp=original.mlp
        elif arm=='wide':self.mlp=SwiGLU(256,1344,0.)
        else:
            self.mlp=RoutedSwiGLU(256,336,4.)
            with torch.no_grad():self.mlp.router.weight.mul_(2.)
        self.register_buffer('freq_cos',original.freq_cos)
        self.register_buffer('freq_sin',original.freq_sin)

    def forward(self,x,cache,start,record):
        value,_=self.attn(self.ln1(x),cache=cache,start_pos=start,freq_cos=self.freq_cos,freq_sin=self.freq_sin)
        x=x+value
        if type(self.mlp) is RoutedSwiGLU:
            value,aux,_=self.mlp(self.ln2(x),record_routes=record)
        else:value,aux=self.mlp(self.ln2(x)),None
        return x+value,aux


class Cache(NativeCache):
    """Own signature/type; same proved attention-buffer update equations."""


class GPT(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=copy.deepcopy(cfg);self.signature=signature(cfg);arm=self.signature['arm']
        # Genuine native construction and validation, never a fake legacy cfg.
        base=Native(native_config('incumbent'))
        blocks=nn.ModuleList([Block(b,arm) for b in base.transformer.h])
        self.transformer=nn.ModuleDict(dict(wte=base.transformer.wte,h=blocks,ln_f=base.transformer.ln_f))
        self.lm_head=base.lm_head;self.width_mult=4.;self.validate()

    def validate(self):
        s=self.signature;require(signature(self.cfg)==s,'Mutable own model policy')
        params=list(self.parameters());device=params[0].device
        require(all(p.dtype==torch.float32 and p.requires_grad and p.device==device for p in params),'FP32 learned single-device parameters required')
        require(sum(p.numel() for p in params)==s['parameters'] and len(params)==s['muon_groups']+s['adam_groups'],'Own parameter inventory differs')
        require(len({id(p) for p in params})==len(params) and len({p.untyped_storage().data_ptr() for p in params})==len(params),'Aliased model parameters')
        require(self.width_mult==4. and len(self.transformer.h)==4,'Body/readout scale differs')
        require(type(self.transformer.wte) is Embedding and tuple(self.transformer.wte.weight.shape)==(50257,256),'Native embedding differs')
        require(type(self.lm_head) is nn.Linear and tuple(self.lm_head.weight.shape)==(50257,256) and self.lm_head.bias is None,'Native readout differs')
        norm=self.transformer.ln_f
        require(type(norm) is FinalLayerNorm and norm.width==256 and norm.kind=='learned' and norm.eps==1e-5 and tuple(norm.weight.shape)==tuple(norm.bias.shape)==(256,),'Final norm differs')
        for i,b in enumerate(self.transformer.h):
            require(type(b) is Block and b.layer_index==i,'Own block layout differs')
            for n in (b.ln1,b.ln2):require(type(n) is nn.RMSNorm and n.normalized_shape==(256,) and n.eps==2**-23 and n.elementwise_affine and tuple(n.weight.shape)==(256,),'Native block norm differs')
            a=b.attn
            require(type(a) is Attention and a.dim==256 and a.dimc_q==a.dimc_kv==64 and a.n_heads==8 and a.head_dim==a.head_dim_r==32 and a.attention_logit_scale==.5 and a.causal and a.max_seq_len==128 and a.attn_dropout.p==a.out_dropout.p==0.,'Native MLA differs')
            for name,shape in dict(dkv=(64,256),uk=(256,64),uv=(256,64),dq=(64,256),uq=(256,64),qr=(256,64),kr=(32,256),out=(256,256)).items():
                m=getattr(a,name);require(type(m) is nn.Linear and tuple(m.weight.shape)==shape and m.bias is not None and tuple(m.bias.shape)==(shape[0],),'MLA projection differs')
            if s['arm']=='moe':
                m=b.mlp;require(type(m) is RoutedSwiGLU and m.dim==256 and m.hidden==336 and m.router_scale==4. and len(m.experts)==4,'Own routed FFN differs')
                require(tuple(m.router.weight.shape)==(4,256) and m.router.bias is None,'Router differs')
                for expert in m.experts:
                    require(type(expert) is Expert,'Foreign expert')
                    for name,shape in [('gate',(336,256)),('up',(336,256)),('down',(256,336))]:
                        v=getattr(expert,name);require(type(v) is nn.Linear and tuple(v.weight.shape)==shape and v.bias is None,'Expert projection differs')
                require(m.assignment_counts.dtype==m.routed_tokens.dtype==torch.int64 and m.assignment_counts.device==m.routed_tokens.device==device and tuple(m.assignment_counts.shape)==(4,) and m.routed_tokens.ndim==0,'Counter state differs')
                require(bool((m.assignment_counts>=0).all()) and int(m.assignment_counts.sum())==2*int(m.routed_tokens),'Counter coverage differs')
            else:
                m=b.mlp;require(type(m) is SwiGLU and m.dim==256 and m.hidden_dim==s['hidden'] and m.gate_activation=='silu' and m.hidden_norm is None and m.dropout.p==0.,'Dense FFN differs')
                for name,shape in [('linear_gate',(s['hidden'],256)),('linear_up',(s['hidden'],256)),('linear_down',(256,s['hidden']))]:
                    v=getattr(m,name);require(type(v) is nn.Linear and tuple(v.weight.shape)==shape and v.bias is None,'Dense projection differs')
            require(all(tuple(t.shape)==(128,16) and t.dtype==torch.float32 and t.device==device for t in (b.freq_cos,b.freq_sin)),'Native RoPE buffers differ')

    def caches(self,batch,device=None):
        self.validate();return [Cache(self.signature,i,batch,device or self.lm_head.weight.device) for i in range(4)]

    def validate_caches(self,caches,batch,start):
        require(type(caches) in (list,tuple) and len(caches)==4,'Exact own cache inventory required')
        tensors=[]
        for i,c in enumerate(caches):
            require(type(c) is Cache and c.signature==self.signature and type(c.layer_index) is int and c.layer_index==i and type(c.position) is int and c.position==start,'Foreign cache signature/position')
            require(tuple(c.c_cache.shape)==(batch,128,64) and tuple(c.r_cache.shape)==(batch,128,32),'Cache shape differs')
            for t in (c.c_cache,c.r_cache):
                require(t.dtype==torch.float32 and t.device==self.lm_head.weight.device and t.is_contiguous() and not t.requires_grad,'Cache layout/device differs');tensors.append(t)
        require(len({id(c) for c in caches})==4 and len({t.untyped_storage().data_ptr() for t in tensors})==8,'Aliased cache')
        forbidden={p.untyped_storage().data_ptr() for p in self.parameters()}
        require(not any(t.untyped_storage().data_ptr() in forbidden for t in tensors),'Cache aliases weights')

    def forward(self,idx,targets=None,cache=None,start_pos=0,include_aux_loss=True,return_loss_breakdown=False,include_training_objective=False):
        self.validate();require(not torch.is_autocast_enabled(self.lm_head.weight.device.type),'Autocast forbidden')
        require(idx.dtype==torch.int64 and idx.ndim==2 and min(idx.shape)>0 and idx.device==self.lm_head.weight.device,'Device-local token matrix required')
        require(type(start_pos) is int and 0<=start_pos and start_pos+idx.shape[1]<=128,'Invalid position')
        require(not bool((idx<0).any()) and not bool((idx>=50257).any()),'Token outside native vocabulary')
        require(all(type(v) is bool for v in (include_aux_loss,return_loss_breakdown,include_training_objective)),'Explicit boolean objective flags required')
        if targets is not None:
            require(targets.shape==idx.shape and targets.dtype==torch.int64 and targets.device==idx.device,'Target layout differs')
            require(not bool((targets<0).any()) and not bool((targets>=50257).any()),'Target outside native vocabulary')
        if cache is None:require(start_pos==0,'Offset requires caches')
        else:
            require(not torch.is_grad_enabled(),'Cache is inference only');self.validate_caches(cache,idx.shape[0],start_pos)
        record=bool(self.training and targets is not None and include_training_objective)
        x=self.transformer.wte(idx);aux=[]
        for i,b in enumerate(self.transformer.h):
            x,a=b(x,None if cache is None else cache[i],start_pos,record)
            if a is not None:aux.append(a)
        logits=self.lm_head(self.transformer.ln_f(x))/4.
        ce=None
        if targets is not None:
            ce=F.cross_entropy(logits.reshape(-1,50257),targets.reshape(-1))
        router_aux=sum(aux)/4 if aux else None
        total=ce+.01*router_aux if ce is not None and router_aux is not None and include_aux_loss else ce
        if return_loss_breakdown:return logits,total,dict(ce_loss=ce,aux_loss=router_aux,total_loss=total)
        return logits,total


def load_weights(model,payload):
    model.validate();require(type(payload) is dict and payload.get('kind')=='tokenizer_moe_weights_v1' and payload.get('architecture')==model.signature,'Own weights signature required')
    current=model.state_dict();saved=payload.get('model_state')
    require(type(saved) in (dict,type(current)) and saved.keys()==current.keys(),'Weights tensor inventory differs')
    for n,t in saved.items():
        require(type(t) is torch.Tensor and t.shape==current[n].shape and t.dtype==current[n].dtype and bool(torch.isfinite(t).all()),'Weight tensor differs: '+n)
        if n.endswith(('freq_cos','freq_sin')):
            cos,sin=precompute_freqs_cis(32,128);require(torch.equal(t.cpu(),cos if n.endswith('freq_cos') else sin),'Foreign RoPE buffer')
    for i in range(4):
        key=f'transformer.h.{i}.mlp.'
        if model.signature['arm']=='moe':
            counts=saved[key+'assignment_counts'];tokens=saved[key+'routed_tokens']
            require(bool((counts>=0).all()) and int(tokens)>=0 and int(counts.sum())==2*int(tokens),'Invalid saved router counter coverage')
    model.load_state_dict(saved)
