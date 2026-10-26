"""Metadata and Python scalar checks only; standalone execution forbids Torch."""
import builtins,copy,unittest
from types import SimpleNamespace as NS
original_import=builtins.__import__
def no_torch(name,*a,**kw):
    if name=='torch' or name.startswith('torch.'):raise AssertionError('Model work deferred')
    return original_import(name,*a,**kw)
if __name__=='__main__':builtins.__import__=no_torch
from tests.test_residual_gain_contract import config as inherited
from src.models.value_residual import (POLICY,KINDS,FIELDS,value_residual_contract,value_residual_signature,resolve_value_residual,validate_value_residual_model,validate_value_residual_load,mix_values)
def config(kind='local',gain='unit',activation='silu',width=64,scale=.5,topology='serial',table='legacy_untied'):
    c=inherited(gain,activation,width,scale,topology,table)
    c.update(value_residual_kind=kind,value_residual_width_policy=POLICY);return c
def fake(c):
    return NS(cfg=c,value_residual_signature=value_residual_signature(c),transformer=NS(h=[NS(attn=NS(value_residual_kind=resolve_value_residual(c),value_residual_layer=i)) for i in range(4)]))
def cp(m):return dict(cfg=dict(model=copy.deepcopy(m.cfg)))
class Scalar:
    def __init__(self,v):self.v=v;self.shape=(1,2,3,32);self.dtype='float32';self.device='cpu'
    def __rmul__(self,x):return Scalar(x*self.v)
    def __add__(self,x):return Scalar(self.v+x.v)
class Contract(unittest.TestCase):
    def test_legacy(self):
        self.assertIsNone(value_residual_contract({}));validate_value_residual_load(NS(cfg={}),dict(cfg=dict(model={})))
    def test_partial_and_inheritance(self):
        for f in FIELDS:
            c=config();del c[f]
            with self.assertRaises(ValueError):value_residual_contract(c)
        c=config();del c['residual_gain_kind'];del c['residual_gain_width_policy']
        with self.assertRaises(ValueError):value_residual_contract(c)
    def test_all_saved_kind_pairs(self):
        for k in KINDS:
            for o in KINDS:
                if k!=o:
                    with self.assertRaises(ValueError):validate_value_residual_load(fake(config(k)),cp(fake(config(o))))
    def test_current_mutation_and_deletion(self):
        for k in KINDS:
            m=fake(config(k));m.cfg['value_residual_kind']='local' if k!='local' else 'first_blend'
            with self.assertRaises(ValueError):validate_value_residual_load(m,cp(m))
            m=fake(config(k))
            for f in FIELDS:del m.cfg[f]
            with self.assertRaises(ValueError):validate_value_residual_load(m,cp(m))
    def test_layer_and_kind_corruption(self):
        for i in range(4):
            for field,value in [('value_residual_layer',3-i),('value_residual_kind','wrong')]:
                m=fake(config());setattr(m.transformer.h[i].attn,field,value)
                with self.assertRaises(ValueError):validate_value_residual_model(m)
        m=fake(config());m.transformer.h.pop()
        with self.assertRaises(ValueError):validate_value_residual_model(m)
    def test_layer_index_type(self):
        for v in (True,1.,False,0.):
            m=fake(config());m.transformer.h[int(v)].attn.value_residual_layer=v
            with self.assertRaises(ValueError):validate_value_residual_model(m)
    def test_persistent_activation_forbidden(self):
        for n in ('first_layer_value','value_residual_state'):
            for where in ('model','block','attention'):
                m=fake(config());target=m if where=='model' else m.transformer.h[1] if where=='block' else m.transformer.h[1].attn
                setattr(target,n,object())
                with self.assertRaises(ValueError):validate_value_residual_model(m)
    def test_scalar_formula_and_identity(self):
        obj=object()
        for k in KINDS:self.assertIs(mix_values(obj,None,k,0),obj)
        self.assertIs(mix_values(obj,None,'local',3),obj)
        self.assertEqual(mix_values(Scalar(6),None,'local_half',2).v,3)
        self.assertEqual(mix_values(Scalar(6),Scalar(10),'first_blend',3).v,8)
    def test_scalar_invalid_chunks(self):
        for field,value in [('shape',(2,2,3,32)),('dtype','float64'),('device','mps')]:
            first=Scalar(1);setattr(first,field,value)
            with self.assertRaises(ValueError):mix_values(Scalar(2),first,'first_blend',1)
        with self.assertRaises(ValueError):mix_values(Scalar(2),None,'first_blend',1)
        with self.assertRaises(ValueError):mix_values(Scalar(2),Scalar(1),'first_blend',0)
        for i in (-1,4,True,1.5):
            with self.assertRaises(ValueError):mix_values(Scalar(2),None,'local',i)
def valid(k,g,a,w,s,t,table):
    def test(self):
        m=fake(config(k,g,a,w,s,t,table));validate_value_residual_load(m,cp(m))
    return test
for ki,k in enumerate(KINDS):
 for gi,g in enumerate(('unit','static_tenth','learned_tenth')):
  for ai,a in enumerate(('silu','gelu_exact','relu')):
   for w in (64,256):
    for si,s in enumerate((1.,.5,2.)):
     t=('serial','parallel_separate_norms')[(ki+gi+ai+si)%2];table=('legacy_untied','matched_untied','scaled_tied')[(gi+ai+si)%3]
     setattr(Contract,f'test_valid_{k}_{g}_{a}_{w}_{s}',valid(k,g,a,w,s,t,table))
def invalid(k,v):
    def test(self):
        c=config();c[k]=v
        with self.assertRaises(ValueError):value_residual_contract(c)
    return test
bad=[('value_residual_kind',v) for v in (True,None,1,'first','')]
bad+=list(dict(value_residual_width_policy='wrong',value_residual_coefficient=.5,value_residual_detach=False,value_residual_shared_cache=False,value_residual_recursive=False,value_residual_learned=False,value_residual_first_layer=True,n_layers=6,dim=96,attn_type='kda',residual_gain_kind='bad',gated_activation_kind='gelu',attention_logit_scale=3.).items())
for i,(k,v) in enumerate(bad):setattr(Contract,f'test_invalid_{i}_{k}',invalid(k,v))
if __name__=='__main__':unittest.main(verbosity=2)
