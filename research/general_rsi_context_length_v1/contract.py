"""Explicit native context identity; unchanged parameter and optimizer policy."""
from dataclasses import dataclass
from research.byte_input_model_v3.contract import require, same_typed, primitive, counts
from research.byte_input_model_v3.contract import native_model as original_native_model
POLICY='general_rsi_context_length_v1'
@dataclass(frozen=True)
class Spec:
    context: int
    variant: str='native'
    def __post_init__(self): signature(self)
def signature(spec):
    require(type(spec) is Spec and type(spec.context) is int and spec.context in (128,256) and spec.variant=='native', 'Own native context128/256 spec')
    return (POLICY, 'native', 256, 50257, spec.context, 'uncached')
def native_model(context):
    require(type(context) is int and context in (128,256), 'Own context')
    config=original_native_model();config['block_size']=context
    return config
def native_config(context):
    from omegaconf import OmegaConf
    return OmegaConf.create(native_model(context))
def model_config(spec):
    signature(spec)
    return dict(policy=POLICY, context=spec.context, native_initializer_recipe=native_model(spec.context), primary_objective='ordinary_next_token_cross_entropy', cache='unsupported')
