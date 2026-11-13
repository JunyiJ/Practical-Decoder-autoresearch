"""Explicit depth identity; frozen four-block initialization is a component only."""
from dataclasses import dataclass
from research.byte_input_model_v3.contract import require, same_typed, primitive, native_model, native_config
POLICY='general_rsi_cross_size_depth_v1'
@dataclass(frozen=True)
class Spec:
    variant: str
    depth: int
    extra_group_seeds: tuple
    composer_seed: object=None
    codebook_identity: object=None
    def __post_init__(self): signature(self)
def signature(spec):
    require(type(spec) is Spec and spec.variant in ('native','raw_bytes'), 'Own native/raw-byte spec')
    require(type(spec.depth) is int and spec.depth in (4,8,16), 'Depth4/8/16 only')
    require(type(spec.extra_group_seeds) is tuple and len(spec.extra_group_seeds)==3 and len(set(spec.extra_group_seeds))==3 and all(type(s) is int and 0<=s<2**63 for s in spec.extra_group_seeds), 'Three distinct private extra-group seeds')
    if spec.variant=='native':require(spec.composer_seed is None and spec.codebook_identity is None, 'Native has no composer')
    else:
        require(type(spec.composer_seed) is int and 0<=spec.composer_seed<2**63 and spec.composer_seed not in spec.extra_group_seeds, 'Private composer seed')
        require(type(spec.codebook_identity) is tuple and len(spec.codebook_identity)==10 and spec.codebook_identity[0]=='rsi_byte_codebook_v1' and spec.codebook_identity[1]=='raw_bytes' and spec.codebook_identity[4:6]==(50257,130), 'Qualified raw-byte codebook')
    return (POLICY,spec.variant,spec.depth,spec.extra_group_seeds,spec.composer_seed,spec.codebook_identity,256,50257,128)
def counts(spec):
    signature(spec);d=spec.depth
    return (21*d+4,25732096+690080*d,11*d,10*d+4) if spec.variant=='native' else (21*d+21,13591539+690080*d,11*d+9,10*d+12)
def model_config(spec):
    signature(spec)
    return dict(policy=POLICY,variant=spec.variant,depth=spec.depth,width=256,context=128,extra_group_seeds=spec.extra_group_seeds,composer_seed=spec.composer_seed,codebook_identity=spec.codebook_identity,base_four_block_initializer=native_model(),initialization='First group and input/output use original native initializer. Additional four-block groups use independent private native-initializer draws, discarding their tables and final norm. No shared or cloned blocks.',primary_objective='ordinary_next_token_cross_entropy',cache='unsupported')
