"""Adopt complete-buffer hashing after exact training/cache/artifact checks."""
import copy
import hashlib
import json
import os
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent
BASE=OUT.parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):return json.loads(p.read_text())


def rec(p):return dict(path=str(p),size=p.stat().st_size,sha256=sha(p))


def verify(r):
    p=Path(r['path']);p=p if p.is_absolute() else ROOT/p
    assert p.stat().st_size==r['size'] and sha(p)==r['sha256']


def publish(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


assert not (OUT/'recipe.json').exists()
manifest=read(OUT/'implementation.json')
for r in manifest['sources']:verify(r)
verify(manifest['source_map']);mapping=read(Path(manifest['source_map']['path']))
for pair in mapping['pairs']:
    verify(pair['original']);verify(pair['candidate'])
    s=(ROOT/pair['candidate']['path']).read_text()
    for a,c in reversed(pair['exact_replacements']):s=s.replace(c,a)
    assert s==(ROOT/pair['original']['path']).read_text()
train_dir=BASE/'bulk-hash-training001';train=read(train_dir/'result.json')
assert train['completed'] and train['exact_history_and_endpoint'] and train['exact_cached_logits_and_reset']
assert train['cpu']['exact_resume_step1_to_step2'] and train['cpu']['exact_boundaries']==[1,2]
assert train['total_numerical_updates']==45 and train['quality_evaluations']==0
for name,digest in train['evidence'].items():assert sha(train_dir/name)==digest
rows=[read(train_dir/f'mps-step{i:02}.json') for i in range(1,21)]
ratios=[r['seconds']['batched']/r['seconds']['reference'] for r in rows[4:]]
assert statistics.median(ratios)==train['median_candidate_over_reference']<=.95
assert sum(r<1 for r in ratios)==train['paired_faster']>=12
for name,lo,hi in [('before_averaging',5,12),('with_averaging',13,20)]:
    assert statistics.median(r['candidate_over_reference'] for r in rows if lo<=r['step']<=hi)==train['phase_median_ratios'][name]<=1
assert train['useful_training_improvement'] and all(train['gates'].values())
for backend,step in [('cpu',1),('cpu',2),('mps',20)]:
    assert read(train_dir/f'{backend}-reference-state{step:02}.json')==read(train_dir/f'{backend}-batched-state{step:02}.json')
assert read(train_dir/'cpu-resume-state02.json')==read(train_dir/'cpu-reference-state02.json')
for backend in ('cpu','mps'):
    cache=read(train_dir/f'{backend}-cache.json');assert cache['exact'] and cache['reset']
    assert cache['fingerprints']['reference']==cache['fingerprints']['batched']
art_dir=BASE/'bulk-hash-artifact001';artifact=read(art_dir/'result.json')
assert all(artifact[k] for k in ('completed','raw_restore_and_capture_exact','h200_export_exact','h200_load_exact','original_model_class_preserved'))
assert artifact['training_updates']==artifact['quality_evaluations']==artifact['new_weight_files']==0
micro_dir=BASE/'bulk-hash-runtime001';micro=read(micro_dir/'result.json')
assert micro['accepted'] and micro['useful_mps_improvement']
assert {c['backend'] for c in micro['cases']}=={'cpu','mps'}
for c in micro['cases']:
    assert len(c['buffers'])==14 and len(c['edge_checks'])==4
    for buffer in c['buffers']:
        faults=buffer['mutation_rejections'];assert len(faults)==2
        assert faults['reference']['mutated_sha256']==faults['bulk_bytes']['mutated_sha256']
    assert all(e['mutated_digests']['reference']==e['mutated_digests']['bulk_bytes'] for e in c['edge_checks'])
for directory,result in [(micro_dir,micro),(train_dir,train),(art_dir,artifact)]:
    assert result['guard_sha256']==sha(directory/'guard.json')
    guard=read(directory/'guard.json');assert guard['status']=='finished' and guard['reason'] is None
    assert result['plan_sha256']==sha(directory/'plan.json') and result['runner_sha256']==sha(directory/'run.py')
    for r in read(directory/'plan.json')['bindings']:verify(r)
previous=BASE/'optimizer-state-adoption001/recipe.json';recipe=copy.deepcopy(read(previous))
assert recipe['status']=='adopted_equivalent_batched_optimizer_state_implementation'
recipe.update(status='adopted_equivalent_bulk_hash_implementation',previous_implementation_recipe=rec(previous),
    implementation=rec(OUT/'implementation.json'),training_implementation=rec(OUT/'implementation.json'),
    implementation_qualification='Runtime training/cache proof plus saved RAW/H200 and public-loader proof complete. The immutable implementation manifest records its pre-artifact-check preparation state; this recipe supplies the completed artifact receipt.',
    consumer=rec(ROOT/'research/byte_input_checkpoint_consumer_v4.py'),
    training_entrypoints=dict(model='research.byte_input_model_v6',optimizer='research.byte_input_model_v6.optimizer',
        numerical_state='research.byte_input_state_v6',training='research.byte_input_training_v6',
        inference_export='research.byte_input_inference_v5',cache='research.byte_input_cache_v4'),
    training_implementation_cost_evidence=rec(train_dir/'result.json'),warmed_training_comparison=rec(train_dir/'result.json'),
    artifact_interoperation=rec(art_dir/'result.json'),runtime_loading=rec(art_dir/'result.json'),
    training_implementation_limits=manifest['scope'])
recipe['loader'].update(module='research.byte_input_checkpoint_consumer_v4',implementation_record=rec(OUT/'implementation.json'))
publish('recipe.json',recipe)
publish('closure.json',dict(status='closed_adopted_equivalent_bulk_hash_implementation',recipe=rec(OUT/'recipe.json'),
    implementation=rec(OUT/'implementation.json'),artifact_interoperation=rec(art_dir/'result.json'),
    median_mps_update_latency_reduction=1-train['median_candidate_over_reference'],
    exact_training_cache_resume_and_saved_artifacts=True,new_quality_evaluations=0,new_weight_files=0,
    saved_evidence_audit_only=True))
publish('STOP',dict(status='completed_closed_adopted_equivalent_bulk_hash_implementation',no_replay=True))
print(json.dumps(dict(adopted=True,recipe=str(OUT/'recipe.json'),median_latency_reduction=1-train['median_candidate_over_reference'])))
