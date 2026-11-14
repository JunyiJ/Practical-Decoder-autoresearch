"""Own native D256 depth4/8/16 and context128/256 identity."""
from dataclasses import dataclass
from research.byte_input_model_v3.contract import require, same_typed, primitive
from research.general_rsi_context_length_v1.contract import native_model, native_config
POLICY='general_rsi_depth_context_v1'
@dataclass(frozen=True)
class Spec:
    depth: int
    context: int
    extra_group_seeds: tuple
    variant: str='native'
    composer_seed: object=None
    codebook_identity: object=None
    def __post_init__(self): signature(self)
def signature(spec):
    require(type(spec) is Spec and spec.variant=='native' and spec.composer_seed is None and spec.codebook_identity is None, 'Own native-only spec')
    require(type(spec.depth) is int and spec.depth in (4,8,16), 'Depth4/8/16 only')
    require(type(spec.context) is int and spec.context in (128,256), 'Context128/256 only')
    require(type(spec.extra_group_seeds) is tuple and len(spec.extra_group_seeds)==3 and len(set(spec.extra_group_seeds))==3 and all(type(s) is int and 0<=s<2**63 for s in spec.extra_group_seeds), 'Three distinct private block-group seeds')
    return (POLICY,'native',256,50257,spec.depth,spec.context,spec.extra_group_seeds,'uncached')
def counts(spec):
    signature(spec);d=spec.depth
    return 21*d+4,25732096+690080*d,11*d,10*d+4
def model_config(spec):
    signature(spec)
    return dict(policy=POLICY,variant='native',depth=spec.depth,width=256,context=spec.context,extra_group_seeds=spec.extra_group_seeds,base_four_block_initializer=native_model(spec.context),initialization='Original first-group/input/output draw; additional four-block groups use independent private draws with discarded tables and norm.',primary_objective='ordinary_next_token_cross_entropy',cache='unsupported')
