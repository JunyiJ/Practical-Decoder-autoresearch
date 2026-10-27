"""Standalone metadata/scalar/AST tests; tensor work explicitly prohibited."""
import ast,builtins,copy,math,unittest
from pathlib import Path
from types import SimpleNamespace as NS
_original=builtins.__import__
def blocked(name,*a,**kw):
    if name=='torch' or name.startswith('torch.'):raise AssertionError('Actual model/tensor work deferred')
    return _original(name,*a,**kw)
if __name__=='__main__':builtins.__import__=blocked
from tests.test_logit_z_contract import config as inherited
from src.models.null_attention import (POLICY,KINDS,FIELDS,CONVENTION,FORBIDDEN,STATE_NAMES,
 null_attention_contract,null_attention_signature,resolve_null_attention,
 validate_null_attention_model,validate_null_attention_load,normalize_attention)
def config(kind='standard',objective='none',position='rope',value='local',gain='unit',width=64,activation='silu'):
    c=inherited(objective,position,value,gain,width,activation)
    c.update(null_attention_kind=kind,null_attention_policy=POLICY)
    return c
def fake(c):
    signature=null_attention_signature(c)
    blocks=[NS(attn=NS(null_attention_signature=signature,null_attention_kind=resolve_null_attention(c),causal=True,_parameters={},_buffers={})) for _ in range(4)]
    return NS(cfg=c,null_attention_signature=signature,transformer=NS(h=blocks),_parameters={},_buffers={})
def cp(m):return dict(cfg=dict(model=copy.deepcopy(m.cfg)),model_state={})
class Contracts(unittest.TestCase):
    def test_absence(self):
        self.assertIsNone(null_attention_contract({}));self.assertEqual(resolve_null_attention({}),'standard')
        validate_null_attention_load(NS(cfg={}),dict(cfg=dict(model={})))
    def test_partial(self):
        for key in FIELDS:
            c=config();del c[key]
            with self.assertRaises(ValueError):null_attention_contract(c)
    def test_inherited_objective_required(self):
        c=config()
        for key in ('logit_z_kind','logit_z_coefficient','logit_z_objective_policy'):del c[key]
        with self.assertRaises(ValueError):null_attention_contract(c)
    def test_changed_saved_pair(self):
        for a in KINDS:
            for b in KINDS:
                if a!=b:
                    with self.assertRaises(ValueError):validate_null_attention_load(fake(config(a)),cp(fake(config(b))))
    def test_deletion(self):
        for k in KINDS:
            m=fake(config(k))
            for field in FIELDS:del m.cfg[field]
            with self.assertRaises(ValueError):validate_null_attention_model(m)
            with self.assertRaises(ValueError):validate_null_attention_load(fake(config(k)),dict(cfg=dict(model={})))
            with self.assertRaises(ValueError):validate_null_attention_load(NS(cfg={}),cp(fake(config(k))))
    def test_model_signature(self):
        for value in (None,(),[],['standard',POLICY,CONVENTION],('standard',POLICY,'bad')):
            m=fake(config());m.null_attention_signature=value
            with self.assertRaises(ValueError):validate_null_attention_model(m)
    def test_each_layer(self):
        for i in range(4):
            for field,value in [('null_attention_signature',None),('null_attention_kind','bad'),('causal',False),('causal',1)]:
                m=fake(config());setattr(m.transformer.h[i].attn,field,value)
                with self.assertRaises(ValueError):validate_null_attention_model(m)
        m=fake(config());m.transformer.h.pop()
        with self.assertRaises(ValueError):validate_null_attention_model(m)
    def test_no_null_state(self):
        for key in STATE_NAMES:
            for where in ('model','block','attn'):
                m=fake(config());obj=m if where=='model' else m.transformer.h[0] if where=='block' else m.transformer.h[0].attn
                setattr(obj,key,object())
                with self.assertRaises(ValueError):validate_null_attention_model(m)
            m=fake(config());c=cp(m);c['model_state']['transformer.h.2.attn.'+key]=object()
            with self.assertRaises(ValueError):validate_null_attention_load(m,c)
        for key in FIELDS+('null_attention_signature',):
            for registry in ('_parameters','_buffers'):
                m=fake(config());getattr(m.transformer.h[0].attn,registry)[key]=object()
                with self.assertRaises(ValueError):validate_null_attention_model(m)
    def test_invalid_kind_before_tensor_import(self):
        for k in (None,True,0,'softmax1',''):
            with self.assertRaises(ValueError):normalize_attention(object(),object(),k)
    def test_wiring_ast(self):
        s=Path('src/attention/mha.py').read_text()
        tree=ast.parse(s);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MHA')
        f=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward');text=ast.unparse(f)
        for before in ('add_alibi_bias','att.masked_fill'):
            self.assertLess(text.index(before),text.index('normalize_attention(att, mask, self.null_attention_kind)'))
        self.assertLess(text.index('normalize_attention'),text.index('self.attn_dropout'))
        self.assertLess(text.index('cache.update(k, v, start_pos)'),text.index('normalize_attention'))
        self.assertEqual(text.count('cache.update('),1)
        self.assertIn("if self.null_attention_kind == 'standard':\n        att = torch.softmax(att, dim=-1)",text)
        self.assertIn('validate_null_attention_model(self)',Path('src/models/gpt.py').read_text())
        self.assertIn('validate_null_attention_model(model)',Path('src/train/optimizers.py').read_text())
        self.assertIn('validate_null_attention_load(model, checkpoint)',Path('src/utils/checkpoint.py').read_text())
    def test_primitive_ast(self):
        tree=ast.parse(Path('src/models/null_attention.py').read_text());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='normalize_attention');s=ast.unparse(fn)
        self.assertIn('visible.sum(dim=-1, keepdim=True)',s)
        self.assertIn('mass / (mass + 1)',s)
        self.assertIn('torch.softmax(torch.cat((scores, null), dim=-1), dim=-1)[..., :-1]',s)
        self.assertIn('scores.new_zeros((*scores.shape[:-1], 1))',s)
        self.assertLess(s.index("if kind == 'standard':"),s.index('counts ='))
        for forbidden in ('.detach(','.clamp(','.exp(','.max('):self.assertNotIn(forbidden,s)

def good(k,o,p,v):
    def test(self):
        i=KINDS.index(k)+('none','z1e4','z1e3').index(o)+('rope','nope','alibi').index(p)+('local','local_half','first_blend').index(v)
        c=config(k,o,p,v,('unit','static_tenth','learned_tenth')[i%3],(64,128,256)[(i//2)%3],('silu','gelu_exact','relu')[i%3])
        m=fake(c);validate_null_attention_load(m,cp(m));self.assertEqual(null_attention_signature(c),(k,POLICY,CONVENTION))
        self.assertEqual(c['logit_z_kind'],o);self.assertEqual(c['position_family_kind'],p);self.assertEqual(c['value_residual_kind'],v)
    return test
for k in KINDS:
 for o in ('none','z1e4','z1e3'):
  for p in ('rope','nope','alibi'):
   for v in ('local','local_half','first_blend'):setattr(Contracts,f'test_valid_{k}_{o}_{p}_{v}',good(k,o,p,v))
def bad(k,v):
    def test(self):
        c=config();c[k]=v
        with self.assertRaises(ValueError):null_attention_contract(c)
    return test
invalid=[('null_attention_kind',v) for v in (None,True,1,'softmax1','')]+[('null_attention_policy','bad'),('causal',False),('causal',1),('dim',96),('logit_z_coefficient',True),('position_family_kind','bad')]+[(k,False) for k in FORBIDDEN]
for i,(k,v) in enumerate(invalid):setattr(Contracts,f'test_invalid_{i}',bad(k,v))
# Pure independent scalar reference; does NOT call the tensor primitive.
def probs(scores,kind):
    m=max(scores+[0.] if kind=='null_zero' else scores)
    e=[math.exp(x-m) for x in scores];den=sum(e)+(math.exp(-m) if kind=='null_zero' else 0.)
    factor=len(scores)/(len(scores)+1) if kind=='prefix_control' else 1.
    return [factor*x/den for x in e]
class ScalarReference(unittest.TestCase):
    def test_zero_score_control(self):
        for n in (1,2,5,128):
            for kind in ('prefix_control','null_zero'):
                for x in probs([0.]*n,kind):self.assertAlmostEqual(x,1/(n+1),places=14)
    def test_extremes(self):
        for scores in ([10000.,9999.],[-10000.,-9999.],[10000.,-10000.]):
            for kind in KINDS:
                p=probs(list(scores),kind);self.assertTrue(all(math.isfinite(x) and 0<=x<=1 for x in p));self.assertLessEqual(sum(p),1.+1e-14)
    def test_shift(self):
        x=[-.5,.2,1.3]
        for k in ('standard','prefix_control'):
            for a,b in zip(probs(x,k),probs([v+3 for v in x],k)):self.assertAlmostEqual(a,b,places=14)
        self.assertGreater(sum(probs([v+3 for v in x],'null_zero')),sum(probs(x,'null_zero')))
    def test_independent_vjp(self):
        for n in (1,2,5):
            scores=[.3*j-.7 for j in range(n)];values=[math.sin(j+.2) for j in range(n)]
            for kind in KINDS:
                p=probs(scores,kind);y=sum(a*b for a,b in zip(p,values));factor=n/(n+1) if kind=='prefix_control' else 1.
                for j in range(n):
                    analytic=p[j]*(values[j]-y/factor)
                    a=scores.copy();b=scores.copy();a[j]+=1e-5;b[j]-=1e-5
                    numeric=(sum(x*v for x,v in zip(probs(a,kind),values))-sum(x*v for x,v in zip(probs(b,kind),values)))/2e-5
                    self.assertAlmostEqual(analytic,numeric,places=9)
    def test_prefix_counts(self):
        for prefix,chunk in ((0,1),(0,11),(7,4),(120,8)):
            for row in range(chunk):
                count=sum(k<=prefix+row for k in range(prefix+chunk))
                self.assertEqual(count,prefix+row+1)
                self.assertAlmostEqual(sum(probs([0.]*count,'prefix_control')),count/(count+1),places=13)
if __name__=='__main__':unittest.main(verbosity=2)
