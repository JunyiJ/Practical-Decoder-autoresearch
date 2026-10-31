"""Standalone proposed FP32 block-normalizer contract, not a model policy.

No training/model/optimizer/checkpoint entry is registered by this component.
The incumbent calls the exact native RMSNorm path with eps=None.
"""
from dataclasses import asdict,dataclass
from collections.abc import Mapping
import torch
from torch import nn

POLICY='tokenizer_block_norm_fp32_v1'
EPSILON=2.0**-23
KINDS=('rms_incumbent','layer_matched_epsilon')


def require(condition,message):
    if not condition:raise ValueError(message)


@dataclass(frozen=True)
class Spec:
    kind:str
    policy:str=POLICY
    width:int=256
    dtype:str='float32'
    epsilon:float=EPSILON
    affine:bool=True
    placement:str='pre_attention_and_pre_ffn'

    def __post_init__(self):
        require(type(self.kind) is str and self.kind in KINDS,'Unsupported normalizer kind')
        require(type(self.policy) is str and self.policy==POLICY,'Normalizer policy differs')
        require(type(self.width) is int and self.width==256,'Normalizer width must be256')
        require(type(self.dtype) is str and self.dtype=='float32','Only explicit FP32 normalizers supported')
        require(type(self.epsilon) is float and self.epsilon==EPSILON,'Matched FP32 epsilon required')
        require(type(self.affine) is bool and self.affine is True,'Learned affine gain required')
        require(type(self.placement) is str and self.placement=='pre_attention_and_pre_ffn','Normalizer placement differs')

    @property
    def bias(self):return self.kind=='layer_matched_epsilon'


def encode(spec):
    require(type(spec) is Spec,'Exact normalizer Spec required')
    return dict(**asdict(spec),bias=spec.bias)


def decode(value):
    require(isinstance(value,Mapping),'Normalizer signature must be a mapping')
    fields={'kind','policy','width','dtype','epsilon','affine','placement','bias'}
    require(set(value)==fields,'Exact normalizer signature fields required')
    spec=Spec(**{key:value[key] for key in fields-{'bias'}})
    require(type(value['bias']) is bool and value['bias']==spec.bias,'Normalizer bias signature differs')
    return spec


def require_same_signature(saved,current):
    require(decode(saved)==current,'Saved normalizer signature differs despite matching shape')


def make(spec):
    require(type(spec) is Spec,'Exact normalizer Spec required')
    # Explicit dtype preserves behavior under a changed global default dtype.
    if spec.kind=='rms_incumbent':module=nn.RMSNorm(256,eps=None,elementwise_affine=True,dtype=torch.float32)
    else:module=nn.LayerNorm(256,eps=EPSILON,elementwise_affine=True,bias=True,dtype=torch.float32)
    validate(spec,module)
    return module


def validate(spec,module):
    require(type(spec) is Spec,'Exact normalizer Spec required')
    expected=nn.LayerNorm if spec.bias else nn.RMSNorm
    require(type(module) is expected,'Constructed normalizer type differs')
    require(module.normalized_shape==(256,) and module.elementwise_affine is True,'Normalizer shape/affine differs')
    require(module.eps==(EPSILON if spec.bias else None),'Constructed normalizer epsilon differs')
    parameters=dict(module.named_parameters())
    require(set(parameters)==({'weight','bias'} if spec.bias else {'weight'}),'Normalizer parameters differ')
    require(all(type(p) is nn.Parameter and p.shape==(256,) and p.dtype==torch.float32
        and p.requires_grad for p in parameters.values()),'Normalizer FP32 parameter contract differs')
    require(not list(module.buffers()),'Unexpected normalizer buffers')
    require(len({p.device for p in parameters.values()})==1,'Mixed normalizer parameter devices')


def checked_forward(spec,module,x):
    validate(spec,module)
    require(type(x) is torch.Tensor and x.dtype==torch.float32,'Only FP32 input allowed')
    require(x.ndim>=2 and x.shape[-1]==256 and all(n>0 for n in x.shape),'Nonempty token-by-width256 input required')
    require(x.device==module.weight.device,'Input/normalizer device mismatch')
    require(not torch.is_autocast_enabled(x.device.type),'Autocast not supported by FP32 contract')
    return module(x)


def validate_eight(spec,normalizers):
    require(type(normalizers) in (tuple,list) and len(normalizers)==8,'Exactly eight pre-branch normalizers required')
    require(len({id(n) for n in normalizers})==8,'Shared block normalizers forbidden')
    for module in normalizers:validate(spec,module)
    parameters=[p for module in normalizers for p in module.parameters()]
    require(len({id(p) for p in parameters})==len(parameters),'Shared block normalizer parameters forbidden')
    require(len({p.untyped_storage().data_ptr() for p in parameters})==len(parameters),'Aliased block normalizer storage forbidden')
