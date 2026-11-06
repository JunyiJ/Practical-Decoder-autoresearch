"""Native and8Kcomponent geometries with an exact per-ID initialization bridge."""
import copy,functools
from research.body_allocation_model_v1.contract import model_config,signature as backbone_signature,canonical,require,fingerprint
from research.gpt2_prefix_tokenizer_v1.codec import Codec
POLICY='tokenizer_gpt2_prefix_model_v2';KINDS={'native':50257,'prefix8192':8192}
@functools.lru_cache(maxsize=2)
def _template(kind):
 require(kind in KINDS,'Declared component kind required')
 return dict(policy=POLICY,kind=kind,effective_vocab_size=KINDS[kind],tokenizer=Codec(KINDS[kind]).descriptor(),backbone_template=model_config('incumbent'),initialization='Construct genuine original50257-table model with native RNG;clone identical low-ID rows for smaller tables,keep body/readout scaling and post-construction CPU RNG exactly native.')
def config(kind):return copy.deepcopy(_template(kind))
def signature(cfg):
 require(type(cfg) is dict and cfg.get('kind') in KINDS and canonical(cfg)==canonical(config(cfg['kind'])),'Exact explicit codec/model policy required')
 v=cfg['effective_vocab_size'];return dict(backbone_signature(cfg['backbone_template']),policy=POLICY,kind=cfg['kind'],vocab_size=v,parameters=2760832+2*v*256,tokenizer_fingerprint=fingerprint(cfg['tokenizer']),model_fingerprint=fingerprint(cfg))
