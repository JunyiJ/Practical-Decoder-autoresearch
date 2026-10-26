"""Prospective metadata-only contract checks; actual model gates remain deferred."""
import builtins,copy,sys,unittest
from types import SimpleNamespace
original_import=builtins.__import__
def forbid_models(name,*a,**kw):
    if name=='torch' or name.startswith('torch.'):
        raise AssertionError('Actual model imports are deferred while research is live')
    return original_import(name,*a,**kw)
if __name__=='__main__': builtins.__import__=forbid_models
from src.models.readout_norm import readout_norm_contract,readout_norm_signature,validate_readout_norm_load
from src.models.embedding_scale import input_scale_contract

def cfg(kind='layer_affine',width=64,scale=1.):
    return dict(n_layers=4,dim=width,n_heads=width//32,mlp_type='dense',ffn='swiglu',ffn_bias=False,ffn_width_policy='swiglu_21_8',depth_width_policy='depth4_unscaled',attn_type='mha',attention_width_policy='mha_head32',parameterization='mup',base_width=64,norm_type='rmsnorm',rope=1,dropout=0.,attn_dropout=0.,block_size=128,input_width_policy='mha4_input_scale',input_embedding_scale=scale,readout_norm_kind=kind,readout_norm_width_policy='mha4_readout_norm',readout_norm_epsilon=1e-5)
class Contract(unittest.TestCase):
    def test_legacy_without_policy_untouched(self):
        self.assertIsNone(readout_norm_contract({}));self.assertEqual(readout_norm_signature({}),('layer_affine',1e-5))
        validate_readout_norm_load(SimpleNamespace(cfg={}),{'cfg':{'model':{}}})
    def test_partial_policy_rejected(self):
        for key in ('readout_norm_kind','readout_norm_width_policy','readout_norm_epsilon'):
            c=cfg();del c[key]
            with self.assertRaises(ValueError):readout_norm_contract(c)
    def test_same_shape_different_semantics_rejected(self):
        c=cfg('rms');model=SimpleNamespace(cfg=c,readout_norm_signature=('rms',1e-5))
        with self.assertRaises(ValueError):validate_readout_norm_load(model,{'cfg':{'model':cfg('layer_no_bias')}})
    def test_mutated_config_cannot_disguise_constructed_kind(self):
        c=cfg('rms');model=SimpleNamespace(cfg=c,readout_norm_signature=('rms',1e-5));c['readout_norm_kind']='layer_no_bias'
        with self.assertRaises(ValueError):validate_readout_norm_load(model,{'cfg':{'model':copy.deepcopy(c)}})
    def test_deleting_policy_cannot_disguise_constructed_kind(self):
        c=cfg('rms');model=SimpleNamespace(cfg=c,readout_norm_signature=('rms',1e-5))
        for key in ('readout_norm_kind','readout_norm_width_policy','readout_norm_epsilon'):del c[key]
        with self.assertRaises(ValueError):validate_readout_norm_load(model,{'cfg':{'model':copy.deepcopy(c)}})
    def test_explicit_policy_does_not_silently_load_legacy(self):
        c=cfg();model=SimpleNamespace(cfg=c,readout_norm_signature=('layer_affine',1e-5));saved=copy.deepcopy(c)
        for key in ('readout_norm_kind','readout_norm_width_policy','readout_norm_epsilon'):del saved[key]
        with self.assertRaises(ValueError):validate_readout_norm_load(model,{'cfg':{'model':saved}})
    def test_input_scale_contract_validates_combined_readout_policy(self):
        c=cfg();c['readout_norm_epsilon']=1e-6
        with self.assertRaises(ValueError):input_scale_contract(c)
    def test_namespace_config_supported(self):
        c=SimpleNamespace(**cfg('rms'));self.assertEqual(readout_norm_signature(c),('rms',1e-5))

def valid(kind,width,scale):
    def test(self):
        c=cfg(kind,width,scale);v=readout_norm_contract(c)
        self.assertEqual(v,dict(readout_norm_kind=kind,readout_norm_width_policy='mha4_readout_norm',readout_norm_epsilon=1e-5))
        self.assertEqual(input_scale_contract(c),dict(input_width_policy='mha4_input_scale',input_embedding_scale=scale))
        validate_readout_norm_load(SimpleNamespace(cfg=c,readout_norm_signature=(kind,1e-5)),{'cfg':{'model':copy.deepcopy(c)}})
    return test
for kind in ('layer_affine','layer_no_bias','rms'):
 for width in (64,128,256):
  for scale in (1.,.5,.25):setattr(Contract,f'test_valid_{kind}_{width}_{str(scale).replace(".","p")}',valid(kind,width,scale))

def invalid(key,value):
    def test(self):
        c=cfg();c[key]=value
        with self.assertRaises(ValueError):readout_norm_contract(c)
    return test
bad=[('readout_norm_kind','unknown'),('readout_norm_width_policy','unknown'),('readout_norm_epsilon',None),('readout_norm_epsilon',float('nan')),('readout_norm_epsilon',True),('readout_norm_epsilon',1e-6),('n_layers',6),('n_layers',4.),('dim',63),('n_heads',4),('n_heads',2.),('base_width',32),('input_embedding_scale',.75),('input_embedding_scale',True),('input_width_policy',None),('attn_type','gqa'),('norm_type','layernorm'),('ffn','gelu'),('ffn_bias',True),('block_size',512),('dropout',.1),('tie_weights',True),('ffn_hidden_norm','rms'),('qk_norm','rms'),('readout_norm_epsilon',float('inf'))]
for i,(key,value) in enumerate(bad):setattr(Contract,f'test_invalid_{i}_{key}',invalid(key,value))
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    assert 'torch' not in sys.modules
    print('Metadata-only: no torch import, actual model construction, training, evaluation or MPS.')
    raise SystemExit(not result.wasSuccessful())
