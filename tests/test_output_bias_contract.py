"""Pure metadata/scalar tests. Standalone invocation rejects torch imports."""
import builtins
import copy
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

original_import = builtins.__import__
def no_models(name, *a, **kw):
    if name == 'torch' or name.startswith('torch.'):
        raise AssertionError('Model execution deferred until active study exits')
    return original_import(name, *a, **kw)
if __name__ == '__main__': builtins.__import__ = no_models
from src.models.output_bias import (build_training_prior, centered_log_prior,
    validate_prior, output_bias_contract, output_bias_signature,
    validate_output_bias_load, validate_output_bias_training_data, POLICY, FIELDS, digest)

DATA = b'aaaaabbbcczz'
def config(kind='none', width=64, norm='layer_affine', scale=1.):
    return dict(n_layers=4,dim=width,n_heads=width//32,mlp_type='dense',ffn='swiglu',
        ffn_bias=False,ffn_width_policy='swiglu_21_8',depth_width_policy='depth4_unscaled',
        attn_type='mha',attention_width_policy='mha_head32',parameterization='mup',
        base_width=64,norm_type='rmsnorm',rope=1,dropout=0.,attn_dropout=0.,block_size=128,
        input_width_policy='mha4_input_scale',input_embedding_scale=scale,
        readout_norm_kind=norm,readout_norm_width_policy='mha4_readout_norm',readout_norm_epsilon=1e-5,
        vocab_size=4,output_bias_kind=kind,output_bias_width_policy=POLICY,
        output_bias_prior=build_training_prior(DATA,.75,1/12))

class Contract(unittest.TestCase):
    def test_legacy_absence(self):
        self.assertIsNone(output_bias_contract({}))
        validate_output_bias_load(SimpleNamespace(cfg={}), {'cfg':{'model':{}}})
    def test_counts_training_only(self):
        p=build_training_prior(DATA,.75,1/12)
        self.assertEqual(p['counts'],[5,3,1,0])
        self.assertEqual((p['train_end'],p['val_end']), (9,10))
        self.assertEqual(p['alphabet'],['a','b','c','z'])
        validate_prior(p,4)
    def test_tail_frequencies_do_not_change_initialization(self):
        a=build_training_prior(b'aaaabbcczzzz',.5,.25)
        b=build_training_prior(b'aaaabbzzcczz',.5,.25)
        self.assertEqual(a['counts'],b['counts'])
        self.assertEqual(a['initial_values'],b['initial_values'])
        self.assertNotEqual(a['corpus_sha256'],b['corpus_sha256'])
    def test_global_alphabet_dependency_disclosed(self):
        a=build_training_prior(b'aaaabbcczzzz',.5,.25)
        b=build_training_prior(b'aaaabbccyyyy',.5,.25)
        self.assertNotEqual(a['alphabet'],b['alphabet'])
    def test_universal_newline_matches_loader(self):
        p=build_training_prior(b'a\r\na\rb\nc\r\n',.5,.25)
        self.assertEqual(p['token_count'],8)
        self.assertEqual(p['counts'],[2,2,0,0])
    def test_formula_independent_log_ratio_and_zero_mean(self):
        counts=[10,2,0,4];v=centered_log_prior(counts)
        self.assertAlmostEqual(sum(v),0.,places=14)
        for i in range(4):
            for j in range(4):self.assertAlmostEqual(v[i]-v[j],math.log((counts[i]+1)/(counts[j]+1)),places=14)
        e=[math.exp(x) for x in v]
        for i in range(4):self.assertAlmostEqual(e[i]/sum(e),(counts[i]+1)/20,places=14)
    def test_partial_policy(self):
        for k in FIELDS:
            c=config();del c[k]
            with self.assertRaises(ValueError):output_bias_contract(c)
    def test_same_shape_init_and_lineage_change(self):
        c=config('learned_zero');m=SimpleNamespace(cfg=c,output_bias_signature=output_bias_signature(c))
        for changed in (config('learned_train_unigram'), config('none')):
            with self.assertRaises(ValueError):validate_output_bias_load(m,{'cfg':{'model':changed}})
    def test_mutable_config_and_deleted_fields(self):
        c=config('learned_zero');m=SimpleNamespace(cfg=c,output_bias_signature=output_bias_signature(c))
        c['output_bias_kind']='learned_train_unigram'
        with self.assertRaises(ValueError):validate_output_bias_load(m,{'cfg':{'model':copy.deepcopy(c)}})
        for k in FIELDS:del c[k]
        with self.assertRaises(ValueError):validate_output_bias_load(m,{'cfg':{'model':copy.deepcopy(c)}})
    def test_training_loader_binding_without_tensor_or_sampling(self):
        c=config();tokens=[0]*5+[1]*3+[2]
        loader=SimpleNamespace(train_fraction=.75,val_fraction=1/12,vocab_size=4,
            itos=dict(enumerate('abcz')),train_data=SimpleNamespace(tolist=lambda:tokens))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'toy.txt';p.write_bytes(DATA)
            validate_output_bias_training_data(c,loader,p)
            tokens[0]=3
            with self.assertRaises(ValueError):validate_output_bias_training_data(c,loader,p)
            tokens[0]=0;p.write_bytes(b'aaaaabbbcczy')
            with self.assertRaises(ValueError):validate_output_bias_training_data(c,loader,p)

def valid(kind,width,norm):
    def test(self):
        c=config(kind,width,norm);contract=output_bias_contract(c)
        self.assertEqual(contract['output_bias_kind'],kind)
        m=SimpleNamespace(cfg=c,output_bias_signature=output_bias_signature(c))
        validate_output_bias_load(m,{'cfg':{'model':copy.deepcopy(c)}})
    return test
for kind in ('none','learned_zero','learned_train_unigram'):
 for width in (64,128,256):
  for norm in ('layer_affine','layer_no_bias','rms'):
   setattr(Contract,f'test_valid_{kind}_{width}_{norm}',valid(kind,width,norm))

def bad_prior(key,value):
    def test(self):
        p=config()['output_bias_prior'];p[key]=value
        # Rehashing the enclosing payload must not defeat semantic validation.
        p['prior_sha256']=digest({k:v for k,v in p.items() if k!='prior_sha256'})
        with self.assertRaises(ValueError):validate_prior(p,4)
    return test
bad=[('counts',[5,3,0,1]),('counts',[True,3,1,4]),('counts',[-1,5,1,4]),
    ('train_end',10),('train_end',9.),('val_end',11),('token_count',11),
    ('alphabet',['z','b','c','a']),('alphabet',['a','b','c','c']),
    ('initial_values',[0.,0.,0.,0.]),('pseudocount',2),('pseudocount',True),
    ('tokenizer','tiktoken'),('corpus_sha256','z'*64),('counts_sha256','0'*64),
    ('initial_values_sha256','0'*64),('train_fraction',0.),('val_fraction',1.),('version',True)]
for i,(k,v) in enumerate(bad):setattr(Contract,f'test_bad_prior_{i}_{k}',bad_prior(k,v))

def bad_config(key,value):
    def test(self):
        c=config();c[key]=value
        with self.assertRaises(ValueError):output_bias_contract(c)
    return test
for i,(k,v) in enumerate([('output_bias_kind','fixed'),('output_bias_width_policy','other'),
    ('vocab_size',3),('vocab_size',True),('n_layers',6),('attn_type','kda'),
    ('dim',512),('tie_weights',True),('readout_norm_epsilon',1e-6),('input_embedding_scale',.75)]):
    setattr(Contract,f'test_bad_config_{i}_{k}',bad_config(k,v))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contract))
    assert 'torch' not in sys.modules
    print('Pure synthetic text/metadata/scalar tests; no real corpus, torch, model, scoring or MPS.')
    raise SystemExit(not result.wasSuccessful())
