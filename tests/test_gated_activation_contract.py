"""Pure metadata checks; standalone execution blocks all tensor imports."""
import builtins, copy, unittest
from types import SimpleNamespace as NS
original_import=builtins.__import__
def no_torch(name,*a,**k):
    if name=='torch' or name.startswith('torch.'):raise AssertionError('Actual model work deferred')
    return original_import(name,*a,**k)
if __name__=='__main__':builtins.__import__=no_torch
from tests.test_attention_scale_contract import config as inherited
from src.models.gated_activation import (POLICY,KINDS,FIELDS,gated_activation_contract,
    gated_activation_signature,resolve_gated_activation,validate_gated_activation_model,validate_gated_activation_load)
def config(kind='silu',width=64,scale=1.,topology='serial',table='legacy_untied'):
    c=inherited(scale,width,topology,table)
    c.update(gated_activation_kind=kind,gated_activation_width_policy=POLICY);return c
def fake(c):
    def mlp():return NS(gate_activation=resolve_gated_activation(c),hidden_dim=21*c['dim']//8,hidden_norm=None,
        linear_gate=NS(bias=None),linear_up=NS(bias=None),linear_down=NS(bias=None))
    return NS(cfg=c,gated_activation_signature=gated_activation_signature(c),transformer=NS(h=[NS(mlp=mlp()) for _ in range(4)]))
def cp(m):return dict(cfg=dict(model=copy.deepcopy(m.cfg)))
class Contract(unittest.TestCase):
    def test_legacy(self):
        self.assertIsNone(gated_activation_contract({}));self.assertEqual(resolve_gated_activation({}),'silu')
        validate_gated_activation_load(NS(cfg={}),dict(cfg=dict(model={})))
    def test_partial_contract(self):
        for f in FIELDS:
            c=config();del c[f]
            with self.assertRaises(ValueError):gated_activation_contract(c)
    def test_same_shape_load(self):
        for k in KINDS:
            for other in KINDS:
                if k!=other:
                    with self.assertRaises(ValueError):validate_gated_activation_load(fake(config(k)),cp(fake(config(other))))
    def test_mutation_deletion(self):
        for k in KINDS:
            m=fake(config(k));m.cfg['gated_activation_kind']='relu' if k!='relu' else 'silu'
            with self.assertRaises(ValueError):validate_gated_activation_load(m,cp(m))
            m=fake(config(k))
            for f in FIELDS:del m.cfg[f]
            with self.assertRaises(ValueError):validate_gated_activation_load(m,cp(m))
    def test_missing_inherited_attention(self):
        c=config();del c['attention_logit_scale'];del c['attention_logit_scale_width_policy']
        with self.assertRaises(ValueError):gated_activation_contract(c)
    def test_block_semantics_shapes_bias(self):
        for attr,value in [('gate_activation','relu'),('hidden_dim',169),('hidden_norm',NS())]:
            m=fake(config());setattr(m.transformer.h[2].mlp,attr,value)
            with self.assertRaises(ValueError):validate_gated_activation_model(m)
        for name in ('linear_gate','linear_up','linear_down'):
            m=fake(config());getattr(m.transformer.h[2].mlp,name).bias=0
            with self.assertRaises(ValueError):validate_gated_activation_model(m)
        m=fake(config());m.transformer.h.pop()
        with self.assertRaises(ValueError):validate_gated_activation_model(m)
def valid(k,w,s,t,table):
    def test(self):
        m=fake(config(k,w,s,t,table));validate_gated_activation_load(m,cp(m));self.assertEqual(resolve_gated_activation(m.cfg),k)
    return test
for k in KINDS:
 for w in (64,256):
  for s in (1.,.5,2.):
   for t in ('serial','parallel_separate_norms'):
    for table in ('legacy_untied','matched_untied','scaled_tied'):
     setattr(Contract,f'test_valid_{k}_{w}_{s}_{t}_{table}',valid(k,w,s,t,table))
def invalid(k,v):
    def test(self):
        c=config();c[k]=v
        with self.assertRaises(ValueError):gated_activation_contract(c)
    return test
bad=[('gated_activation_kind',v) for v in (True,False,None,1,'gelu','tanh','')]
bad+=list(dict(gated_activation_width_policy='other',activation='gelu',gate_activation='relu',swiglu_beta=1.,gelu_approximate='tanh',gate_multiplier=1.,ffn_activation_scale=1.,ffn_hidden_norm_policy='none',ffn_hidden_norm='none',dim=96,n_layers=6,base_width=128,ffn='gelu',ffn_bias=True,ffn_width_policy='dense_4x',dropout=.1,attention_logit_scale=3.,attn_type='kda').items())
for i,(k,v) in enumerate(bad):setattr(Contract,f'test_invalid_{i}_{k}',invalid(k,v))
if __name__=='__main__':unittest.main(verbosity=2)
