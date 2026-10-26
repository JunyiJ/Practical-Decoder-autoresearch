"""Standalone pure metadata checks; Torch imports blocked when run directly."""
import builtins
import copy
import unittest
from types import SimpleNamespace as NS
original_import = builtins.__import__
def no_torch(name,*args,**kwargs):
    if name=='torch' or name.startswith('torch.'):
        raise AssertionError('Model work is deferred while research runs')
    return original_import(name,*args,**kwargs)
if __name__=='__main__':builtins.__import__=no_torch
from tests.test_residual_topology_contract import config as residual_config
from src.models.attention_scale import (POLICY,SCALES,FIELDS,attention_scale_contract,
    attention_scale_signature,resolve_attention_scale,validate_attention_scale_model,validate_attention_scale_load)

def config(scale=1.,width=64,topology='serial',table='legacy_untied'):
    c=residual_config(topology,width,table)
    c.update(attention_logit_scale=scale,attention_logit_scale_width_policy=POLICY)
    return c

def fake(c):
    return NS(cfg=c,attention_scale_signature=attention_scale_signature(c),
        transformer=NS(h=[NS(attn=NS(attention_logit_scale=resolve_attention_scale(c))) for _ in range(4)]))
def cp(m):return dict(cfg=dict(model=copy.deepcopy(m.cfg)))

class Contract(unittest.TestCase):
    def test_legacy_absence(self):
        self.assertIsNone(attention_scale_contract({}))
        self.assertEqual(resolve_attention_scale({}),1.)
        validate_attention_scale_load(NS(cfg={}),dict(cfg=dict(model={})))
    def test_partial_contract(self):
        for f in FIELDS:
            c=config();del c[f]
            with self.assertRaises(ValueError):attention_scale_contract(c)
    def test_same_shape_mismatch(self):
        for s in SCALES:
            m=fake(config(s))
            for t in SCALES:
                if s!=t:
                    with self.assertRaises(ValueError):validate_attention_scale_load(m,cp(fake(config(t))))
    def test_mutation_and_deletion(self):
        for s in SCALES:
            m=fake(config(s));m.cfg['attention_logit_scale']=2. if s!=2. else .5
            with self.assertRaises(ValueError):validate_attention_scale_load(m,cp(m))
            m=fake(config(s))
            for f in FIELDS:del m.cfg[f]
            with self.assertRaises(ValueError):validate_attention_scale_load(m,cp(m))
    def test_block_scalar_and_count(self):
        m=fake(config());m.transformer.h[2].attn.attention_logit_scale=.5
        with self.assertRaises(ValueError):validate_attention_scale_model(m)
        m=fake(config());m.transformer.h.pop()
        with self.assertRaises(ValueError):validate_attention_scale_model(m)
    def test_missing_inherited_topology(self):
        c=config();del c['residual_topology_kind'];del c['residual_topology_width_policy']
        with self.assertRaises(ValueError):attention_scale_contract(c)

def valid(s,w,t):
    def test(self):
        m=fake(config(s,w,t));validate_attention_scale_load(m,cp(m));self.assertEqual(resolve_attention_scale(m.cfg),s)
    return test
for s in SCALES:
 for w in (64,128,256):
  for t in ('serial','parallel_separate_norms'):
   setattr(Contract,f'test_valid_{s}_{w}_{t}',valid(s,w,t))
def invalid(k,v):
    def test(self):
        c=config();c[k]=v
        with self.assertRaises(ValueError):attention_scale_contract(c)
    return test
BAD=[('attention_logit_scale',v) for v in (0.,-1.,3.,True,False,'1',float('nan'),float('inf'),None)]
BAD += [(k,v) for k,v in dict(attention_logit_scale_width_policy='other',attention_temperature=1.,attn_scale=1.,softmax_scale=1.,qk_scale=1.,logit_softcap=None,learned_temperature=False,dim=96,n_layers=6,base_width=128,attn_type='kda',qk_norm='rms',residual_topology_kind='shared',hidden_dim=512,dropout=.1).items()]
for i,(k,v) in enumerate(BAD):setattr(Contract,f'test_invalid_{i}_{k}',invalid(k,v))
if __name__=='__main__':unittest.main(verbosity=2)
