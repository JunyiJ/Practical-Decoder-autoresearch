"""Metadata-only prospective residual-gain checks; standalone blocks Torch imports."""
import builtins, copy, unittest
from types import SimpleNamespace as NS
original_import=builtins.__import__
def no_torch(name,*a,**kw):
    if name=='torch' or name.startswith('torch.'):raise AssertionError('Model work deferred until active research exits')
    return original_import(name,*a,**kw)
if __name__=='__main__':builtins.__import__=no_torch
from tests.test_gated_activation_contract import config as inherited
from src.models.residual_gain import (POLICY,KINDS,FIELDS,PARAMETERS,residual_gain_contract,
    residual_gain_signature,resolve_residual_gain,validate_residual_gain_model,validate_residual_gain_load,
    apply_residual_gain)
def config(kind='unit',activation='silu',width=64,scale=.5,topology='serial',table='legacy_untied'):
    c=inherited(activation,width,scale,topology,table)
    c.update(residual_gain_kind=kind,residual_gain_width_policy=POLICY);return c
def fake(c):
    blocks=[]
    for _ in range(4):
        b=NS(residual_gain_kind=resolve_residual_gain(c),_parameters={},_buffers={})
        if b.residual_gain_kind=='learned_tenth':
            for n in PARAMETERS:
                p=NS(shape=(c['dim'],),requires_grad=True);setattr(b,n,p);b._parameters[n]=p
        blocks.append(b)
    return NS(cfg=c,residual_gain_signature=residual_gain_signature(c),transformer=NS(h=blocks))
def cp(m):return dict(cfg=dict(model=copy.deepcopy(m.cfg)))
class Contract(unittest.TestCase):
    def test_legacy(self):
        self.assertIsNone(residual_gain_contract({}));self.assertEqual(resolve_residual_gain({}),'unit')
        validate_residual_gain_load(NS(cfg={}),dict(cfg=dict(model={})))
    def test_partial(self):
        for f in FIELDS:
            c=config();del c[f]
            with self.assertRaises(ValueError):residual_gain_contract(c)
    def test_inherited_activation_required(self):
        c=config();del c['gated_activation_kind'];del c['gated_activation_width_policy']
        with self.assertRaises(ValueError):residual_gain_contract(c)
    def test_kind_mismatch_all_pairs(self):
        for k in KINDS:
            for o in KINDS:
                if k!=o:
                    with self.assertRaises(ValueError):validate_residual_gain_load(fake(config(k)),cp(fake(config(o))))
    def test_mutation_and_deletion(self):
        for k in KINDS:
            m=fake(config(k));m.cfg['residual_gain_kind']='unit' if k!='unit' else 'static_tenth'
            with self.assertRaises(ValueError):validate_residual_gain_load(m,cp(m))
            m=fake(config(k))
            for f in FIELDS:del m.cfg[f]
            with self.assertRaises(ValueError):validate_residual_gain_load(m,cp(m))
    def test_block_kind_and_count(self):
        for k in KINDS:
            m=fake(config(k));m.transformer.h[0].residual_gain_kind='unknown'
            with self.assertRaises(ValueError):validate_residual_gain_model(m)
            m=fake(config(k));m.transformer.h.pop()
            with self.assertRaises(ValueError):validate_residual_gain_model(m)
    def test_unit_identity_static_complete_scalar(self):
        # Python objects/scalars only; not tensor forward verification.
        obj=object();self.assertIs(apply_residual_gain(NS(residual_gain_kind='unit'),obj,'attention'),obj)
        self.assertEqual(apply_residual_gain(NS(residual_gain_kind='static_tenth'),20.+10.,'attention'),3.)
        for branch in ('attention','ffn'):
            self.assertEqual(apply_residual_gain(NS(residual_gain_kind='static_tenth'),-20.,branch),-2.)
        with self.assertRaises(ValueError):apply_residual_gain(NS(residual_gain_kind='unit'),1.,'other')
    def test_learned_registration_shapes_sharing(self):
        for name in PARAMETERS:
            for mutation in ('shape','frozen','unregistered','buffer','missing','shared_branch','shared_layer'):
                m=fake(config('learned_tenth'));b=m.transformer.h[1];p=getattr(b,name)
                if mutation=='shape':p.shape=(1,64)
                elif mutation=='frozen':p.requires_grad=False
                elif mutation=='unregistered':b._parameters.pop(name)
                elif mutation=='buffer':b._buffers[name]=p
                elif mutation=='missing':delattr(b,name)
                elif mutation=='shared_branch':
                    other=PARAMETERS[1-PARAMETERS.index(name)];setattr(b,other,p);b._parameters[other]=p
                else:
                    p=getattr(m.transformer.h[0],name);setattr(b,name,p);b._parameters[name]=p
                with self.assertRaises(ValueError):validate_residual_gain_model(m)
    def test_nonlearned_stray_state(self):
        for k in ('unit','static_tenth'):
            for name in PARAMETERS:
                for where in ('attribute','parameter','buffer'):
                    m=fake(config(k));b=m.transformer.h[0]
                    if where=='attribute':setattr(b,name,NS())
                    elif where=='parameter':b._parameters[name]=None
                    else:b._buffers[name]=None
                    with self.assertRaises(ValueError):validate_residual_gain_model(m)
def valid(k,a,w,s,t,table):
    def test(self):
        m=fake(config(k,a,w,s,t,table));validate_residual_gain_load(m,cp(m));self.assertEqual(resolve_residual_gain(m.cfg),k)
    return test
for k in KINDS:
 for a in ('silu','gelu_exact','relu'):
  for w in (64,256):
   for s in (1.,.5,2.):
    for t in ('serial','parallel_separate_norms'):
     for table in ('legacy_untied','matched_untied','scaled_tied'):
      setattr(Contract,f'test_valid_{k}_{a}_{w}_{s}_{t}_{table}',valid(k,a,w,s,t,table))
def invalid(k,v):
    def test(self):
        c=config();c[k]=v
        with self.assertRaises(ValueError):residual_gain_contract(c)
    return test
bad=[('residual_gain_kind',v) for v in (True,False,None,1,'learned','static','')]
bad+=list(dict(residual_gain_width_policy='other',residual_gain_init=.2,residual_gain_scale=.1,residual_gain_shared=False,residual_gain_activation='sigmoid',layer_scale=.1,layerscale=.1,rezero=True,residual_branch_scale=.1,attention_residual_scale=.1,ffn_residual_scale=.1,dim=96,n_layers=6,base_width=128,ffn='gelu',ffn_bias=True,ffn_width_policy='dense_4x',dropout=.1,attention_logit_scale=3.,attn_type='kda',gated_activation_kind='gelu').items())
for i,(k,v) in enumerate(bad):setattr(Contract,f'test_invalid_{i}_{k}',invalid(k,v))
if __name__=='__main__':unittest.main(verbosity=2)
