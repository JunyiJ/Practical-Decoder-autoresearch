"""Standalone pure metadata tests prohibit torch imports."""
import builtins
import copy
import math
from types import SimpleNamespace as NS
import unittest

original_import=builtins.__import__
def no_models(name,*a,**kw):
    if name=='torch' or name.startswith('torch.'):
        raise AssertionError('Actual model work deferred while research is live')
    return original_import(name,*a,**kw)
if __name__=='__main__': builtins.__import__=no_models
from tests.test_output_bias_contract import config as bias_config
from src.models.weight_tying import (POLICY,KINDS,FIELDS,weight_tying_contract,
    weight_tying_signature,readout_scale,validate_weight_tying_model,validate_weight_tying_load)

def config(kind='legacy_untied',width=64,norm='layer_affine',bias='none'):
    c=bias_config(bias,width,norm)
    c.update(weight_tying_kind=kind,weight_tying_width_policy=POLICY)
    return c

def fake(c):
    e,q=object(),object();kind=c['weight_tying_kind']
    pairs=[('transformer.wte.weight',e)]
    if kind!='scaled_tied':pairs.append(('lm_head.weight',q))
    m=NS(cfg=c,weight_tying_signature=weight_tying_signature(c),table_readout_scale=readout_scale(c),
        transformer=NS(wte=NS(weight=e)),lm_head=None if kind=='scaled_tied' else NS(weight=q),pairs=pairs)
    m.named_parameters=lambda remove_duplicate=False: iter(m.pairs)
    return m

def checkpoint(m):return dict(cfg=dict(model=copy.deepcopy(m.cfg)),model_state=dict(m.pairs))

class Contract(unittest.TestCase):
    def test_legacy_absence(self):
        self.assertIsNone(weight_tying_contract({}));self.assertEqual(readout_scale({}),1.)
        validate_weight_tying_load(NS(cfg={}),dict(cfg=dict(model={})))
    def test_partial_policy(self):
        for field in FIELDS:
            c=config();del c[field]
            with self.assertRaises(ValueError):weight_tying_contract(c)
    def test_fixed_scale(self):
        self.assertEqual(readout_scale(config()),1.)
        for k in KINDS[1:]:self.assertEqual(readout_scale(config(k)),1/math.sqrt(192))
    def test_same_shape_legacy_matched_semantics(self):
        m=fake(config());c=config('matched_untied')
        with self.assertRaises(ValueError):validate_weight_tying_load(m,dict(cfg=dict(model=c)))
    def test_config_mutation(self):
        m=fake(config());m.cfg['weight_tying_kind']='matched_untied'
        with self.assertRaises(ValueError):validate_weight_tying_load(m,checkpoint(m))
    def test_config_deletion(self):
        for k in KINDS:
            m=fake(config(k))
            for f in FIELDS:del m.cfg[f]
            with self.assertRaises(ValueError):validate_weight_tying_load(m,checkpoint(m))
    def test_scale_mutation(self):
        m=fake(config('scaled_tied'));m.table_readout_scale=1.
        with self.assertRaises(ValueError):validate_weight_tying_model(m)
    def test_duplicate_table_alias(self):
        m=fake(config('scaled_tied'));m.pairs.append(('other.weight',m.transformer.wte.weight))
        with self.assertRaises(ValueError):validate_weight_tying_model(m)
    def test_illegal_untied_alias(self):
        m=fake(config('matched_untied'));m.lm_head.weight=m.transformer.wte.weight
        with self.assertRaises(ValueError):validate_weight_tying_model(m)
    def test_illegal_tied_head(self):
        m=fake(config('scaled_tied'));m.lm_head=NS(weight=object())
        with self.assertRaises(ValueError):validate_weight_tying_model(m)
    def test_checkpoint_missing_or_extra_head(self):
        for k in KINDS:
            m=fake(config(k));cp=checkpoint(m)
            if k=='scaled_tied':cp['model_state']['lm_head.weight']=object()
            else:del cp['model_state']['lm_head.weight']
            with self.assertRaises(ValueError):validate_weight_tying_load(m,cp)
    def test_checkpoint_missing_table(self):
        m=fake(config('scaled_tied'));cp=checkpoint(m);del cp['model_state']['transformer.wte.weight']
        with self.assertRaises(ValueError):validate_weight_tying_load(m,cp)

def valid(kind,width,norm,bias):
    def test(self):
        m=fake(config(kind,width,norm,bias));validate_weight_tying_load(m,checkpoint(m))
        self.assertEqual(weight_tying_contract(m.cfg)['weight_tying_kind'],kind)
    return test
for kind in KINDS:
 for width in (64,128,256):
  for norm in ('layer_affine','layer_no_bias','rms'):
   for bias in ('none','learned_zero','learned_train_unigram'):
    setattr(Contract,f'test_valid_{kind}_{width}_{norm}_{bias}',valid(kind,width,norm,bias))

def invalid(key,value):
    def test(self):
        c=config('scaled_tied');c[key]=value
        with self.assertRaises(ValueError):weight_tying_contract(c)
    return test
BAD={'weight_tying_kind':'naive','weight_tying_width_policy':'other','base_width':128,
     'dim':96,'n_heads':4,'n_layers':6,'attn_type':'kda','readout_norm_kind':'other',
     'input_embedding_scale':.75,'output_bias_kind':'other','output_bias_prior':None,
     'tie_weights':True,'tie_embeddings':True,'readout_scale':1.,'weight_tying_scale':.1,
     'shared_table_scale':.1,'output_init_std':.1,'readout_init_std':.1,
     'residual_scale':1.,'qk_norm':'rms','ffn_hidden_norm':'rms','moe':{},
     'hidden_dim':512,'ffn_bias':True,'dropout':.1}
for key,value in BAD.items():setattr(Contract,'test_invalid_'+key,invalid(key,value))
if __name__=='__main__':unittest.main(verbosity=2)
