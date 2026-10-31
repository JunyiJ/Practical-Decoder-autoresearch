"""Preparation-only contracts: no model, optimizer, MPS entry or launch."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from research.block_norm_v1 import campaign_contract as c, campaign_rules as r
from research.block_norm_v1.test_campaign_rules import SEEDS
from research.block_norm_v1.runtime.train.lr_schedule import schedule_contract, multiplier

OUT=c.ROOT/'research_runs/tokenizer-block-norm-v2-contract-fixture/evidence'


def config(stage='pilots',slot=0,spec=None):
    spec=spec or (r.pilot_specs(SEEDS) if stage=='pilots' else r.screen_specs(SEEDS))[slot]
    path=OUT/stage/f'{slot:02d}'
    return c.trial_config(path,spec,SEEDS),path,spec


def test_every_slot_configuration_and_schedule(record_property):
    before=copy.deepcopy(SEEDS);checked=0
    rows=[('pilots',i,s) for i,s in enumerate(r.pilot_specs(SEEDS))]+[('quality',i,s) for i,s in enumerate(r.screen_specs(SEEDS))]
    for a in r.RATES:
        for b in r.RATES:
            rows += [('quality',i+6,s) for i,s in enumerate(r.confirmation_specs(SEEDS,dict(zip(r.ARMS,[a,b]))))]
    for stage,i,spec in rows:
        cfg,path,_=config(stage,i,spec);t=cfg['training'];pilot=stage=='pilots'
        c.validate_trial_config(cfg,path,spec,SEEDS)
        assert t['max_iters']==(20 if pilot else 1200) and t['eval_seed']==SEEDS['evaluation']
        assert t['batch_size']==8 and t['batch_layout']=='paired_blocks512'
        assert t['weight_averaging']==dict(kind='spaced_tail_ema',half_lives=[200],start_step=13 if pilot else 800,interval=1 if pilot else 10,total_steps=spec['steps'])
        schedule=schedule_contract(SimpleNamespace(**t))
        assert multiplier(schedule,0)==1. and multiplier(schedule,schedule['start_step'])==1. and multiplier(schedule,spec['steps']-1)==.1
        assert cfg['model']['tokenizer_architecture_policy']==c.GROUPED_POLICY
        assert cfg['model']['embedding_backward']==c.EMBEDDING_SPEC
        assert cfg['model']['norm_type']==('rmsnorm' if spec['arm']==r.ARMS[0] else 'layernorm')
        assert cfg['data']==c.baseline()['data'] and not path.exists()
        checked+=1
    assert SEEDS==before;record_property('slot_selection_configurations_checked',checked)


MUTATIONS=[('model.dim',128),('model.norm_type','layernorm'),('model.block_normalization.epsilon',1e-5),
 ('model.block_normalization.affine',1),('model.embedding_backward.max_gradient_tokens',2048),
 ('model.tokenizer_architecture_policy','tokenizer_mla_block_norm_v1'),('model.output_bias_prior.train_ids_sha256','bad'),
 ('training.seed',373305),('training.eval_seed',2026),('training.learning_rate',.01),('training.max_iters',21),
 ('training.batch_size',1),('training.batch_layout','contiguous'),('training.optimizer','adamw'),
 ('training.weight_decay',0.),('training.decay_mode','coupled'),('training.resume_from','old.pt'),
 ('training.strict_device',False),('training.device','cpu'),('training.eval_iters',11),
 ('training.weight_averaging.start_step',1),('training.lr_schedule.start_step',19),
 ('training.metrics_file','/tmp/metrics.jsonl'),('data.val_fraction',.2),('data.tokenizer','character'),('extra',True)]


@pytest.mark.parametrize('key,value',MUTATIONS)
def test_configuration_mutations_reject(key,value):
    cfg,path,spec=config();part=cfg;keys=key.split('.')
    for k in keys[:-1]:part=part[k]
    part[keys[-1]]=value
    with pytest.raises(ValueError):c.validate_trial_config(cfg,path,spec,SEEDS)


@pytest.mark.parametrize('kind',['relative','outside','closed','traversal','stage','format','slot','wrong_spec','type','extra_spec'])
def test_slot_binding_rejections(kind):
    _,path,spec=config()
    if kind=='relative':path=Path('research_runs/tokenizer-block-norm-v2-fixture/evidence/pilots/00')
    elif kind=='outside':path=Path('/private/tmp/tokenizer-block-norm-v2-fixture/evidence/pilots/00')
    elif kind=='closed':path=c.ROOT/'research_runs/tokenizer-activation-001/evidence/pilots/00'
    elif kind=='traversal':path=path.parent/'..'/'pilots'/'00'
    elif kind=='stage':path=OUT/'test'/'00'
    elif kind=='format':path=OUT/'pilots'/'0'
    elif kind=='slot':path=OUT/'pilots'/'02'
    elif kind=='wrong_spec':spec=r.pilot_specs(SEEDS)[1]
    elif kind=='type':spec['width']=256.0
    else:spec['override']=True
    with pytest.raises(ValueError):c.trial_config(path,spec,SEEDS)


def toy(root):
    for d in c.SOURCE_ROOTS:
        p=root/d;p.mkdir(parents=True);(p/'one.py').write_text('x=1\n')
    (root/'requirements.txt').write_text('torch\n')
    p=root/c.BASELINE;p.parent.mkdir(parents=True);p.write_text('{}\n')
    return root


@pytest.mark.parametrize('kind',['add','delete','edit','rename','symlink_file','symlink_dir','hardlink'])
def test_source_inventory_rejects_drift_and_aliases(tmp_path,kind):
    root=toy(tmp_path/'repo');expected=c.source_inventory(root);p=root/'src/one.py'
    c.verify_sources(expected,root)
    if kind=='add':(p.parent/'new.py').write_text('new=1\n')
    elif kind=='delete':p.unlink()
    elif kind=='edit':p.write_text('x=2\n')
    elif kind=='rename':p.rename(p.with_name('two.py'))
    elif kind=='symlink_file':(p.parent/'two.py').symlink_to(p)
    elif kind=='symlink_dir':(p.parent/'alias').symlink_to(root/'research/block_norm_v1',target_is_directory=True)
    else:os.link(p,p.with_name('two.py'))
    with pytest.raises(ValueError):c.verify_sources(expected,root)


def test_output_symlink_and_bounded_file_checks(tmp_path):
    root=tmp_path/'repo';root.mkdir();real=root/'real';real.mkdir()
    (root/'research_runs').symlink_to(real,target_is_directory=True)
    with pytest.raises(ValueError):c._output(root/'research_runs/tokenizer-block-norm-v2-fixture/evidence/pilots/00',root)
    f=real/'bounded';f.write_bytes(b'abc')
    with pytest.raises(ValueError):c.file_sha(f,root=root,max_bytes=2)
    with pytest.raises(ValueError):c.file_sha(real,root=root)
    with pytest.raises(ValueError):c.file_sha(f,root=real/'outside')


def test_frozen_sources_and_bytecode_exclusion(tmp_path,record_property):
    root=toy(tmp_path/'repo');expected=c.source_inventory(root)
    cache=root/'src/__pycache__';cache.mkdir();(cache/'one.pyc').write_bytes(b'not source')
    c.verify_sources(expected,root)
    actual=c.source_inventory();frozen=c.frozen_predecessors(actual)
    assert len(frozen['recovery_modules'])==13
    record_property('source_inventory_files',len(actual))
    record_property('source_inventory_sha256',c.fingerprint(actual))


def test_actual_data_coverage_without_test_tokenization(monkeypatch,record_property):
    real=c.tiktoken.get_encoding('gpt2');calls=[]
    class Spy:
        def __getattr__(self,name):return getattr(real,name)
        def encode(self,text,**kw):calls.append(text);return real.encode(text,**kw)
    monkeypatch.setattr(c.tiktoken,'get_encoding',lambda name:Spy())
    data=c.bind_data();text=Path(data['path']).read_text()
    assert calls==[text[:int(.8*len(text))],text[int(.8*len(text)):int(.9*len(text))]]
    assert data['train_tokens']==267688 and data['validation_tokens']==34279
    assert data['validation_targets']==34278 and data['validation_target_bytes']==111536
    assert data['validation_window_target_counts']==[128]*267+[102]
    assert data['statistical_block_target_counts']==[2048]*16+[1510]
    assert sum(data['validation_window_target_counts'])==sum(data['statistical_block_target_counts'])==34278
    assert data['holdout_tokenized'] is False and data['holdout_scored'] is False
    record_property('validation_binding',data)


@pytest.mark.parametrize('name,value',[('PYTORCH_ENABLE_MPS_FALLBACK','1'),('PYTORCH_MPS_FAST_MATH','1'),('OMP_NUM_THREADS','8'),('OPENBLAS_NUM_THREADS','8'),('MKL_NUM_THREADS','8')])
def test_environment_change_rejected(monkeypatch,name,value):
    monkeypatch.setenv(name,value)
    with pytest.raises(ValueError):c.environment()


def test_actual_environment_and_manifest_roundtrip(record_property):
    manifest=c.preparation_manifest(SEEDS)
    c.verify_preparation_manifest(json.loads(c.canonical(manifest)))
    assert manifest['launch_permitted'] is False and manifest['seeds']==SEEDS
    record_property('preparation_manifest',manifest)


@pytest.mark.parametrize('key,value',[('launch_permitted',True),('baseline_sha256','wrong'),('limits',{}),('data',{}),('sources',{}),('environment',{}),('remaining_gates',[]),('extra',True)])
def test_manifest_edit_rejected(key,value):
    manifest=c.preparation_manifest(SEEDS);manifest[key]=value
    with pytest.raises(ValueError):c.verify_preparation_manifest(manifest)


@pytest.mark.parametrize('value',[float('nan'),float('inf'),(1,2),{1:'one'},Path('x')])
def test_nonportable_values_rejected(value):
    with pytest.raises(ValueError):c.canonical(value)
