"""Standalone metadata checks block all Torch imports while research runs."""
import builtins
import copy
from types import SimpleNamespace as NS
import unittest

original_import = builtins.__import__
def no_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Actual model work is deferred until active research finishes')
    return original_import(name, *args, **kwargs)
if __name__ == '__main__':
    builtins.__import__ = no_torch

from tests.test_weight_tying_contract import config as table_config
from src.models.residual_topology import (POLICY, KINDS, FIELDS,
    residual_topology_contract, residual_topology_signature, resolve_residual_topology,
    validate_residual_topology_model, validate_residual_topology_load)

def config(kind='serial', width=64, table='legacy_untied'):
    c = table_config(table, width)
    c.update(residual_topology_kind=kind, residual_topology_width_policy=POLICY)
    return c

def fake(c):
    return NS(cfg=c,residual_topology_signature=residual_topology_signature(c),
              transformer=NS(h=[NS(residual_topology_kind=resolve_residual_topology(c),
                  ln1=object(),ln2=object()) for _ in range(4)]))

def cp(m):
    return dict(cfg=dict(model=copy.deepcopy(m.cfg)))

class Contract(unittest.TestCase):
    def test_legacy_absence(self):
        self.assertIsNone(residual_topology_contract({}))
        self.assertEqual(resolve_residual_topology({}), 'serial')
        validate_residual_topology_load(NS(cfg={}), {'cfg':{'model':{}}})

    def test_partial_contract(self):
        for field in FIELDS:
            c=config();del c[field]
            with self.assertRaises(ValueError):residual_topology_contract(c)

    def test_same_shape_mismatch(self):
        m=fake(config());other=fake(config(KINDS[1]))
        with self.assertRaises(ValueError):validate_residual_topology_load(m,cp(other))

    def test_mutation_and_deletion(self):
        for kind in KINDS:
            m=fake(config(kind));m.cfg['residual_topology_kind']=KINDS[1-KINDS.index(kind)]
            with self.assertRaises(ValueError):validate_residual_topology_load(m,cp(m))
            m=fake(config(kind))
            for field in FIELDS:del m.cfg[field]
            with self.assertRaises(ValueError):validate_residual_topology_load(m,cp(m))

    def test_actual_block_signature(self):
        m=fake(config(KINDS[1]));m.transformer.h[2].residual_topology_kind='serial'
        with self.assertRaises(ValueError):validate_residual_topology_model(m)

    def test_separate_norms(self):
        m=fake(config(KINDS[1]));m.transformer.h[0].ln2=m.transformer.h[0].ln1
        with self.assertRaises(ValueError):validate_residual_topology_model(m)

    def test_block_count(self):
        m=fake(config());m.transformer.h.pop()
        with self.assertRaises(ValueError):validate_residual_topology_model(m)

def valid(kind,width,table):
    def test(self):
        m=fake(config(kind,width,table));validate_residual_topology_load(m,cp(m))
        self.assertEqual(resolve_residual_topology(m.cfg),kind)
    return test
for kind in KINDS:
 for width in (64,128,256):
  for table in ('legacy_untied','matched_untied','scaled_tied'):
   setattr(Contract,f'test_valid_{kind}_{width}_{table}',valid(kind,width,table))

def invalid(key,value):
    def test(self):
        c=config();c[key]=value
        with self.assertRaises(ValueError):residual_topology_contract(c)
    return test
BAD={'residual_topology_kind':'shared_norm','residual_topology_width_policy':'other',
 'parallel_residual':True,'parallel_attn':True,'shared_block_norm':False,
 'residual_branch_scale':1.,'ffn_input_scale':1.,'weight_tying_kind':'naive',
 'base_width':128,'dim':96,'n_heads':4,'n_layers':6,'attn_type':'kda',
 'residual_scale':1.,'qk_norm':'rms','ffn_hidden_norm':'rms','moe':{},
 'hidden_dim':512,'ffn_bias':True,'dropout':.1}
for key,value in BAD.items():setattr(Contract,'test_invalid_'+key,invalid(key,value))
if __name__ == '__main__':unittest.main(verbosity=2)
