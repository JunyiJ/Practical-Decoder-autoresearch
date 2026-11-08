"""Read-only cached inference adapter; does not change frozen model-v3 APIs.

One session owns four native MLA caches and one immutable eval model. Callers
must serialize access and discard a session after an execution/driver failure.
"""
import torch
from omegaconf import OmegaConf
from research.byte_input_model_v5 import ByteInputModel
from research.byte_input_model_v3.contract import require, native_model, same_typed
from research.final_ln_scale_model_v1.runtime.attention.cache import build_caches, KVCache
from research.final_ln_scale_model_v1.runtime.models.block_norm import contract

KIND='byte_input_cached_inference_v1'


class Session:
    def __init__(self, model, batch):
        require(type(model) is ByteInputModel and not model.training
                and type(batch) is int and batch in (1,8), 'Own eval model and B1/B8 cache')
        self.model=model;self.batch=batch;self.device=model.validate();self.position=0
        cfg=OmegaConf.create(dict(model=native_model()))
        self.signature=contract(cfg.model)
        self.caches=build_caches(cfg,self.device,batch_size=batch,dtype=torch.float32)
        self._layers=tuple(self.caches)
        self._tensors=tuple(t for c in self.caches for t in (c.c_cache,c.r_cache))
        self._model_state=self._versions()
        self.check()

    def _versions(self):
        return tuple((name,id(t),t.untyped_storage().data_ptr(),t._version,t.device,t.dtype,tuple(t.shape))
                     for name,t in (*self.model.ordered_parameters(),*self.model.named_buffers()))

    def check(self):
        require(not self.model.training and all(not m.training for m in self.model.modules())
                and self._versions()==self._model_state, 'Unchanged eval model for cache lifetime')
        require(type(self.position) is int and 0<=self.position<=128 and type(self.caches) is list
                and len(self.caches)==4 and all(c is old for c,old in zip(self.caches,self._layers)),
                'Four original session-owned cache layers')
        pointers=[]
        for i,c in enumerate(self.caches):
            require(type(c) is KVCache and c.layer_index==i and c.attn_type=='mla' and c.is_mla is True
                    and same_typed(c.block_norm_signature,self.signature) and c.hybrid_signature is None
                    and type(c.position) is int and c.position==self.position, 'Aligned native cache policy and positions')
            for j,(t,width) in enumerate(((c.c_cache,64),(c.r_cache,32))):
                require(t is self._tensors[2*i+j] and type(t) is torch.Tensor
                        and tuple(t.shape)==(self.batch,128,width) and t.dtype==torch.float32
                        and t.device==self.device and not t.requires_grad, 'Original cache tensor layout')
                pointers.append(t.untyped_storage().data_ptr())
        require(len(set(pointers))==8, 'Nonaliasing caches')

    @torch.no_grad()
    def reset(self):
        self.check();self.model.validate()
        for c in self.caches:c.reset()
        self.position=0;self.check()

    @torch.no_grad()
    def logits(self, inputs, start_pos):
        self.check()
        require(type(start_pos) is int and start_pos==self.position, 'Contiguous integer cache position')
        self.model._ids(inputs,self.device)
        require(inputs.shape[0]==self.batch and self.position+inputs.shape[1]<=128,
                'Bound cache batch and complete context')
        inputs=inputs.contiguous()
        if self.model.spec.variant=='native':
            logits,_=self.model.native(inputs,cache=self.caches,start_pos=start_pos)
        else:
            ids=inputs.reshape(-1)
            codes=self.model.code_ids.index_select(0,ids)
            lengths=self.model.code_lengths.index_select(0,ids)
            hidden=self.model.composer(codes,lengths).reshape(self.batch,inputs.shape[1],256)
            for block,cache in zip(self.model.blocks,self.caches):
                hidden,auxiliary=block(hidden,cache,start_pos)
                require(auxiliary is None, 'No auxiliary objective')
            logits=self.model.readout(self.model.final_norm(hidden))/4.
        require(logits.dtype==torch.float32 and tuple(logits.shape)==(*inputs.shape,50257)
                and bool(torch.isfinite(logits).all()), 'Finite ordinary cached primary logits')
        self.position+=inputs.shape[1];self.check()
        return logits

    @property
    def storage_bytes(self):
        self.check()
        return sum(t.numel()*t.element_size() for t in self._tensors)
