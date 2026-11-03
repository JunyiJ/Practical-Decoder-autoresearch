"""Explicit final-scale construction and identity; historical guards unchanged."""
from torch import nn
from research.final_ln_scale_v1.component import FinalLayerNorm
from research.adamw_epsilon_model_v2.runtime.models.readout_norm import readout_norm_contract as historical_contract,build_readout_norm as historical_build
from .block_norm import READOUT_POLICY,contract,get,require

def readout_norm_contract(cfg):
 if get(cfg,'tokenizer_architecture_policy')!=READOUT_POLICY:return historical_contract(cfg)
 v=contract(cfg)['final_normalization']
 return dict(readout_norm_kind='layer_affine',readout_norm_epsilon=1e-5,readout_norm_width_policy='mla4_final_ln_scale_v1',final_ln_bias_kind='learned',final_ln_scale_kind=v['scale_kind'])
def readout_norm_signature(cfg):
 v=readout_norm_contract(cfg);return ('layer_affine',1e-5) if v is None else tuple(sorted(v.items()))
def build_readout_norm(cfg):
 if get(cfg,'tokenizer_architecture_policy')!=READOUT_POLICY:return historical_build(cfg)
 return FinalLayerNorm(256,contract(cfg)['final_normalization']['scale_kind'])
def validate_final(model):
 require(get(model.cfg,'tokenizer_architecture_policy')==READOUT_POLICY,'Explicit final scale policy required')
 v=contract(model.cfg)['final_normalization'];n=model.transformer.ln_f
 require(type(n) is FinalLayerNorm and n.width==256 and n.eps==1e-5 and n.kind==v['scale_kind'],'Constructed final scale semantics differ')
 require(set(n._parameters)=={'weight','bias'} and not n._buffers and not n._modules,'Exact final norm structure required')
 require(set(n.state_dict())==({'weight','bias'} if v['scale'] else {'bias'}),'Final scale inventory differs')
 require(isinstance(n.bias,nn.Parameter) and n.bias.shape==(256,) and n.bias.requires_grad,'Learned final bias required')
 if v['scale']:require(isinstance(n.weight,nn.Parameter) and n.weight.shape==(256,) and n.weight.requires_grad and n.weight is not n.bias,'Learned scale required')
 else:require(n.weight is None and n._parameters['weight'] is None,'Fixed unit scale cannot be a parameter or buffer')
 require(model.readout_norm_signature==readout_norm_signature(model.cfg),'Constructed scale identity differs')
def validate_readout_norm_load(model,checkpoint):
 require(readout_norm_contract(model.cfg)==readout_norm_contract(get(checkpoint.get('cfg',{}),'model',{})),'Checkpoint final scale policy differs');validate_final(model)
