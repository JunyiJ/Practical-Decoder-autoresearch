"""Metadata/scalar/AST only. Standalone execution forbids tensor imports."""
import ast,builtins,copy,math,unittest
from pathlib import Path
from types import SimpleNamespace as NS
_original=builtins.__import__
def blocked(name,*a,**kw):
    if name=='torch' or name.startswith('torch.'):raise AssertionError('Tensor/model work deferred')
    return _original(name,*a,**kw)
if __name__=='__main__':builtins.__import__=blocked
from tests.test_position_family_contract import config as inherited
from src.models.logit_z import (POLICY,COEFFICIENTS,FIELDS,CONVENTION,logit_z_contract,
    logit_z_signature,validate_logit_z_model,validate_logit_z_load,objective_requested,training_logit_z)
def config(kind='none',position='rope',value='local',gain='unit',width=64,activation='silu'):
    c=inherited(position,value,gain,activation,width)
    c.update(logit_z_kind=kind,logit_z_coefficient=COEFFICIENTS[kind],logit_z_objective_policy=POLICY)
    return c
def fake(c):return NS(cfg=c,logit_z_signature=logit_z_signature(c),_parameters={},_buffers={})
def checkpoint(c):return dict(cfg=dict(model=copy.deepcopy(c)),model_state={})
class Contracts(unittest.TestCase):
    def test_historical_absence(self):
        self.assertIsNone(logit_z_contract({}));validate_logit_z_load(NS(cfg={}),checkpoint({}))
        self.assertIsNone(objective_requested(NS(cfg={}),False,None))
        with self.assertRaises(ValueError):objective_requested(NS(cfg={}),True,object())
    def test_partial_fields(self):
        for f in FIELDS:
            c=config();del c[f]
            with self.assertRaises(ValueError):logit_z_contract(c)
    def test_missing_inherited_position(self):
        c=config();del c['position_family_kind'];del c['position_family_width_policy']
        with self.assertRaises(ValueError):logit_z_contract(c)
    def test_kind_coefficient_pairs(self):
        for a in COEFFICIENTS:
            for b in COEFFICIENTS:
                if a!=b:
                    c=config(a);c['logit_z_coefficient']=COEFFICIENTS[b]
                    with self.assertRaises(ValueError):logit_z_contract(c)
                    with self.assertRaises(ValueError):validate_logit_z_load(fake(config(a)),checkpoint(config(b)))
    def test_current_deletion(self):
        for kind in COEFFICIENTS:
            m=fake(config(kind))
            for f in FIELDS:del m.cfg[f]
            with self.assertRaises(ValueError):validate_logit_z_model(m)
            with self.assertRaises(ValueError):validate_logit_z_load(fake(config(kind)),checkpoint({}))
            with self.assertRaises(ValueError):validate_logit_z_load(NS(cfg={}),checkpoint(config(kind)))
    def test_constructed_mutations(self):
        for kind in COEFFICIENTS:
            for sig in (None,(),('none',0.,'bad',CONVENTION),('none',0.,POLICY,'bad')):
                m=fake(config(kind));m.logit_z_signature=sig
                with self.assertRaises(ValueError):validate_logit_z_model(m)
            m=fake(config(kind));del m.logit_z_signature
            with self.assertRaises(ValueError):validate_logit_z_model(m)
    def test_signature_type_identity(self):
        m=fake(config());m.logit_z_signature=('none',False,POLICY,CONVENTION)
        with self.assertRaises(ValueError):validate_logit_z_model(m)
        with self.assertRaises(ValueError):training_logit_z(object(),object(),object(),m.logit_z_signature)
        m=fake(config());m.logit_z_signature=list(m.logit_z_signature)
        with self.assertRaises(ValueError):validate_logit_z_model(m)

    def test_no_tensor_state(self):
        for field in FIELDS+('logit_z_signature',):
            for registry in ('_parameters','_buffers'):
                m=fake(config());getattr(m,registry)[field]=object()
                with self.assertRaises(ValueError):validate_logit_z_model(m)
    def test_explicit_flag_not_train_mode(self):
        for kind in COEFFICIENTS:
            m=fake(config(kind))
            for train_mode in (True,False):
                m.training=train_mode
                self.assertIsNone(objective_requested(m,False,object()))
                self.assertEqual(objective_requested(m,True,object()),m.logit_z_signature)
            with self.assertRaises(ValueError):objective_requested(m,True,None)
            for flag in (0,1,None,'true'):
                with self.assertRaises(ValueError):objective_requested(m,flag,object())
    def test_zero_path_identity_no_tensor_access(self):
        ce=object();loss,meta=training_logit_z(object(),object(),ce,logit_z_signature(config()))
        self.assertIs(loss,ce)
        self.assertEqual(meta,dict(logit_z_coefficient=0.,logit_z_mean=None,logit_z_squared_mean=None,logit_z_penalty=None))
    def test_invalid_primitive_signatures(self):
        for sig in (None,(),('bad',0.,POLICY,CONVENTION),('none',.1,POLICY,CONVENTION),('none',0.,POLICY,'bad')):
            with self.assertRaises(ValueError):training_logit_z(object(),object(),object(),sig)
    def test_gpt_default_and_loss_order_ast(self):
        tree=ast.parse(Path('src/models/gpt.py').read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef))
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        defaults=dict(zip([a.arg for a in fn.args.args][-len(fn.args.defaults):],fn.args.defaults))
        self.assertIs(defaults['include_training_objective'].value,False)
        s=ast.unparse(fn)
        self.assertLess(s.index('logits = logits + self.output_bias'),s.index('training_logit_z(logits, targets, ce_loss, objective)'))
        self.assertLess(s.index('loss, objective_breakdown = training_logit_z'),s.index('loss = loss + self.aux_loss_weight'))
        self.assertIn('objective_requested(self, include_training_objective, targets)',s)
        self.assertNotIn('self.training',s)
        self.assertIn('"aux_loss": aux_loss',Path('src/models/gpt.py').read_text())
    def test_trainer_and_evaluation_optin_ast(self):
        tree=ast.parse(Path('src/train/train.py').read_text())
        evalfn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_evaluate')
        for call in ast.walk(evalfn):
            if isinstance(call,ast.Call):
                self.assertFalse(any(k.arg=='include_training_objective' and not(isinstance(k.value,ast.Constant) and k.value.value is False) for k in call.keywords))
        s=Path('src/train/train.py').read_text()
        self.assertEqual(s.count('"include_training_objective": True'),1)
        self.assertIn('model(xb, yb, return_loss_breakdown=True, **objective_kwargs)',s)
        for k in ('logit_z_squared_mean','logit_z_penalty','logit_z_mean'):
            self.assertIn(k,s)
        for path in ('src/research/averaging_eval.py','src/research/checkpoint_eval.py'):
            p=Path(path)
            if p.exists():self.assertNotIn('include_training_objective=True',p.read_text())
    def test_optimizer_checkpoint_guards_ast(self):
        self.assertIn('validate_logit_z_model(model)',Path('src/train/optimizers.py').read_text())
        self.assertIn('validate_logit_z_load(model, checkpoint)',Path('src/utils/checkpoint.py').read_text())
    def test_penalty_math_ast(self):
        s=ast.unparse(ast.parse(Path('src/models/logit_z.py').read_text()))
        self.assertIn('torch.logsumexp(flat[valid], dim=-1)',s)
        self.assertIn('targets.reshape(-1).ne(-100)',s)
        self.assertIn('squared = log_z.square().mean()',s)
        self.assertIn('penalty = coefficient * squared',s)
        self.assertNotIn('.detach(',s)
        self.assertLess(s.index('if coefficient == 0.0:'),s.index('import torch'))

def valid(k,p,v,g,w):
    def test(self):
        a=('silu','gelu_exact','relu')[(len(k)+len(p)+w)%3]
        c=config(k,p,v,g,w,a);m=fake(c);validate_logit_z_load(m,checkpoint(c))
        self.assertEqual(logit_z_contract(c)['logit_z_coefficient'],COEFFICIENTS[k])
    return test
for k in COEFFICIENTS:
 for p in ('rope','nope','alibi'):
  for v in ('local','local_half','first_blend'):
   for g in ('unit','static_tenth','learned_tenth'):
    for w in (64,128,256):setattr(Contracts,f'test_valid_{k}_{p}_{v}_{g}_{w}',valid(k,p,v,g,w))
def invalid(k,v):
    def test(self):
        c=config();c[k]=v
        with self.assertRaises(ValueError):logit_z_contract(c)
    return test
bad=[('logit_z_coefficient',v) for v in (True,None,'0',float('nan'),float('inf'),-.1,.00001)]
bad += [('logit_z_kind',v) for v in (True,None,'zloss','')]
bad += [('logit_z_objective_policy','bad'),('position_family_kind','bad'),('dim',96)]
bad += [(k,False) for k in ('logit_z_center','logit_z_temperature','logit_z_clip','logit_z_detach','logit_z_width_scale','label_smoothing','logit_z_ignore_index','logit_z_log_vocab_subtract')]
for i,(k,v) in enumerate(bad):setattr(Contracts,f'test_invalid_{i}',invalid(k,v))
if __name__=='__main__':unittest.main(verbosity=2)
