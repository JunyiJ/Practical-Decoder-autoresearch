"""Pure metadata/scalar/AST checks; standalone run prohibits tensor imports."""
import ast,builtins,copy,unittest
from pathlib import Path
from types import SimpleNamespace as NS
_original_import=builtins.__import__
def no_torch(name,*a,**kw):
    if name=='torch' or name.startswith('torch.'):raise AssertionError('Actual tensor work deferred')
    return _original_import(name,*a,**kw)
if __name__=='__main__':builtins.__import__=no_torch
from tests.test_value_residual_contract import config as inherited
from src.models.position_family import (POLICY,KINDS,FIELDS,position_family_contract,position_family_signature,resolve_position_family,slopes_for_heads,distance_bias_scalar,validate_position_family_model,validate_position_family_load)
def config(kind='rope',value='local',gain='unit',activation='silu',width=64,scale=.5,topology='serial',table='legacy_untied'):
    c=inherited(value,gain,activation,width,scale,topology,table)
    c.update(position_family_kind=kind,position_family_width_policy=POLICY);return c
def fake(c):
    kind=resolve_position_family(c);rot=kind=='rope'
    blocks=[NS(enable_rope=rot,freq_cos=NS(shape=(128,16)) if rot else None,freq_sin=NS(shape=(128,16)) if rot else None,attn=NS(position_family_kind=kind,position_slopes=slopes_for_heads(c['n_heads']) if kind=='alibi' else (),_parameters={},_buffers={})) for _ in range(4)]
    return NS(cfg=c,position_family_signature=position_family_signature(c),position_encoding=kind,enable_rope=rot,transformer=NS(h=blocks))
def cp(m):return dict(cfg=dict(model=copy.deepcopy(m.cfg)),model_state={})
class Contract(unittest.TestCase):
    def test_legacy(self):
        self.assertIsNone(position_family_contract({}));validate_position_family_load(NS(cfg={}),dict(cfg=dict(model={})))
        tree=ast.parse(Path('src/models/rope.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resolve_position_encoding')
        ns={'__package__':'src.models'};exec(compile(ast.Module(body=[fn],type_ignores=[]),'resolver-only','exec'),ns)
        resolve=ns[fn.name]
        self.assertEqual(resolve(NS(attn_type='mha',rope=0)),'learned')
        self.assertEqual(resolve(NS(attn_type='mha',rope=1)),'rope')
        self.assertEqual(resolve(NS(attn_type='kda',rope=0)),'none')
        for k in KINDS:self.assertEqual(resolve(NS(**config(k))),k)
    def test_partial(self):
        for f in FIELDS:
            c=config();del c[f]
            with self.assertRaises(ValueError):position_family_contract(c)
        c=config();del c['value_residual_kind'];del c['value_residual_width_policy']
        with self.assertRaises(ValueError):position_family_contract(c)
    def test_all_saved_pairs(self):
        for k in KINDS:
            for o in KINDS:
                if k!=o:
                    with self.assertRaises(ValueError):validate_position_family_load(fake(config(k)),cp(fake(config(o))))
    def test_deletion_mutation(self):
        for k in KINDS:
            m=fake(config(k));m.cfg['position_family_kind']='alibi' if k!='alibi' else 'rope'
            with self.assertRaises(ValueError):validate_position_family_load(m,cp(m))
            m=fake(config(k))
            for f in FIELDS:del m.cfg[f]
            with self.assertRaises(ValueError):validate_position_family_load(m,cp(m))
    def test_constructed_modes(self):
        for k in KINDS:
            for field,val in [('position_encoding','learned'),('enable_rope',None)]:
                m=fake(config(k));setattr(m,field,val)
                with self.assertRaises(ValueError):validate_position_family_model(m)
            m=fake(config(k));m.transformer.wpe=object()
            with self.assertRaises(ValueError):validate_position_family_model(m)
            m=fake(config(k));m.transformer.h.pop()
            with self.assertRaises(ValueError):validate_position_family_model(m)
    def test_each_block(self):
        for k in KINDS:
            for i in range(4):
                for field,val in [('position_family_kind','bad'),('position_slopes',(1.,))]:
                    m=fake(config(k));setattr(m.transformer.h[i].attn,field,val)
                    with self.assertRaises(ValueError):validate_position_family_model(m)
                m=fake(config(k));m.transformer.h[i].enable_rope=not(k=='rope')
                with self.assertRaises(ValueError):validate_position_family_model(m)
                for name in ('freq_cos','freq_sin'):
                    m=fake(config(k));setattr(m.transformer.h[i],name,None if k=='rope' else NS(shape=(128,16)))
                    with self.assertRaises(ValueError):validate_position_family_model(m)
    def test_tensor_state_rejections(self):
        for k in KINDS:
            for name in ('position_slopes','alibi_slopes','position_state'):
                for where in ('_parameters','_buffers'):
                    m=fake(config(k));getattr(m.transformer.h[1].attn,where)[name]=object()
                    with self.assertRaises(ValueError):validate_position_family_model(m)
            m=fake(config(k));saved=cp(m);saved['model_state']['transformer.wpe.weight']=object()
            with self.assertRaises(ValueError):validate_position_family_load(m,saved)
    def test_exact_slopes(self):
        expected={2:(1/16,1/256),4:(1/4,1/16,1/64,1/256),8:(1/2,1/4,1/8,1/16,1/32,1/64,1/128,1/256)}
        for h,v in expected.items():self.assertEqual(slopes_for_heads(h),v)
        for h in (True,2.,1,3,16,None):
            with self.assertRaises(ValueError):slopes_for_heads(h)
    def test_absolute_distance(self):
        for h in (2,4,8):
            for head,m in enumerate(slopes_for_heads(h)):
                for i in (0,5,31,127):
                    for j in (0,i):
                        self.assertEqual(distance_bias_scalar(h,head,i,j),-m*(i-j))
                        self.assertEqual(distance_bias_scalar(h,head,i+9,j+9),-m*(i-j))
        for args in ((2,True,1,0),(2,2,1,0),(2,0,-1,0),(2,0,1.,0)):
            with self.assertRaises(ValueError):distance_bias_scalar(*args)
    def test_wiring_order_ast(self):
        s=Path('src/attention/mha.py').read_text();tree=ast.parse(s);mha=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MHA');fn=next(n for n in mha.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        code=ast.unparse(fn)
        self.assertLess(code.index('att = att * self.attention_logit_scale'),code.index('att = add_alibi_bias'))
        self.assertLess(code.index('att = add_alibi_bias'),code.index('att = att.masked_fill'))
        self.assertIn('key_start_pos=0 if cache is not None else start_pos',code)
        self.assertIn('mask_start = 0 if self.position_family_kind is not None and cache is None else start_pos',code)
        self.assertIn('k, v = cache.update(k, v, start_pos)',code)
        for f,symbol in [('src/utils/checkpoint.py','validate_position_family_load(model, checkpoint)'),('src/train/optimizers.py','validate_position_family_model(model)')]:self.assertIn(symbol,Path(f).read_text())
def valid(k,v,g,a,w):
    def test(self):
        scales=(1.,.5,2.);i=KINDS.index(k)+len(v)+len(g)+len(a)+w
        c=config(k,v,g,a,w,scales[i%3],('serial','parallel_separate_norms')[i%2],('legacy_untied','matched_untied','scaled_tied')[i%3])
        m=fake(c);validate_position_family_load(m,cp(m))
    return test
for k in KINDS:
 for v in ('local','local_half','first_blend'):
  for g in ('unit','static_tenth','learned_tenth'):
   for a in ('silu','gelu_exact','relu'):
    for w in (64,128,256):setattr(Contract,f'test_valid_{k}_{v}_{g}_{a}_{w}',valid(k,v,g,a,w))
def invalid(k,v):
    def test(self):
        c=config();c[k]=v
        with self.assertRaises(ValueError):position_family_contract(c)
    return test
bad=[('position_family_kind',v) for v in (True,None,1,'none','learned','')]
bad+=list(dict(position_family_width_policy='bad',rope=0,dim=96,n_heads=3,n_layers=6,attn_type='kda',value_residual_kind='bad',position_encoding='none',alibi_slopes=[.5],alibi_scale=1.,alibi_slope_formula='x',position_offset=0,position_theta=10000,position_learned=False,position_interpolation=False,position_family_slopes=None,enable_rope=False).items())
for i,(k,v) in enumerate(bad):setattr(Contract,f'test_invalid_{i}_{k}',invalid(k,v))
if __name__=='__main__':unittest.main(verbosity=2)
