"""Publish the equivalent training implementation after auditing saved evidence."""
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


def record(p):return dict(path=str(p),size=p.stat().st_size,sha256=sha(p))


def verify(r):
    p=Path(r['path']);p=p if p.is_absolute() else ROOT/p
    assert p.stat().st_size==r['size'] and sha(p)==r['sha256']


def publish(name,j):
    with (OUT/name).open('x') as f:
        json.dump(j,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


assert not (OUT/'recipe.json').exists()
previous=BASE/'finite-batch-adoption001/recipe.json';recipe=read(previous)
assert recipe['status']=='adopted_equivalent_batched_finite_implementation'
source_map=BASE/'optimizer-state-integration-source001/source-map.json';mapping=read(source_map)
for pair in mapping['pairs']:
    verify(pair['original']);verify(pair['candidate'])
    s=(ROOT/pair['candidate']['path']).read_text()
    for a,c in reversed(pair['exact_replacements']):s=s.replace(c,a)
    assert s==(ROOT/pair['original']['path']).read_text()
export_map=BASE/'optimizer-state-export001/source-map.json';em=read(export_map)
verify(em['original']);verify(em['candidate'])
s=(ROOT/em['candidate']['path']).read_text()
for a,c in reversed(em['exact_replacements']):s=s.replace(c,a)
assert s==(ROOT/em['original']['path']).read_text()
micro_dir=BASE/'optimizer-state-runtime003';micro=read(micro_dir/'result.json')
assert micro['correctness_passed'] and micro['useful_mps_improvement']
assert len(micro['cases'])==2 and {c['backend'] for c in micro['cases']}=={'cpu','mps'}
for c in micro['cases']:
    assert len(c['rejections'])==20 and all(r['failure_before_mutation'] for r in c['rejections'])
for suffix in ('001','002','003'):
    assert sha(BASE/('optimizer-state-runtime'+suffix)/'cpu.json')==sha(micro_dir/'cpu.json')
train_dir=BASE/'optimizer-state-training001';train=read(train_dir/'result.json')
assert train['completed'] and train['exact_history_and_endpoint'] and train['useful_training_improvement']
assert train['cpu']['exact_boundaries']==[1,2] and train['cpu']['exact_resume_step1_to_step2']
assert train['total_numerical_updates']==45 and train['quality_evaluations']==0
for name,h in train['evidence'].items():assert sha(train_dir/name)==h
rows=[read(train_dir/f'mps-step{i:02}.json') for i in range(1,21)]
assert all(r['exact_history'] for r in rows)
ratios=[r['seconds']['batched']/r['seconds']['reference'] for r in rows[4:]]
assert ratios==[r['candidate_over_reference'] for r in rows[4:]]
assert statistics.median(ratios)==train['median_candidate_over_reference']<=.95
assert sum(r<1 for r in ratios)==train['paired_faster']>=12
for name,lo,hi in [('before_averaging',5,12),('with_averaging',13,20)]:
    median=statistics.median(r['candidate_over_reference'] for r in rows if lo<=r['step']<=hi)
    assert median==train['phase_median_ratios'][name]<=1
assert all(train['gates'].values())
for backend,step in [('cpu',1),('cpu',2),('mps',20)]:
    assert read(train_dir/f'{backend}-reference-state{step:02}.json')==read(train_dir/f'{backend}-batched-state{step:02}.json')
assert read(train_dir/'cpu-resume-state02.json')==read(train_dir/'cpu-reference-state02.json')
export_dir=BASE/'optimizer-state-export001';export=read(export_dir/'result.json')
assert all(export[k] for k in ('completed','raw_restore_and_capture_exact','h200_export_exact','h200_load_exact','original_model_class_preserved'))
assert export['training_updates']==export['quality_evaluations']==export['new_weight_files']==0
for directory,result in [(micro_dir,micro),(train_dir,train),(export_dir,export)]:
    assert result['guard_sha256']==sha(directory/'guard.json')
    guard=read(directory/'guard.json');assert guard['status']=='finished' and guard['reason'] is None
    assert result['plan_sha256']==sha(directory/'plan.json') and result['runner_sha256']==sha(directory/'run.py')
    for r in read(directory/'plan.json')['bindings']:verify(r)
for suffix in ('001','002'):
    assert read(BASE/('optimizer-state-runtime'+suffix)/'guard.json')['status']=='fixture_exception'
implementation=dict(kind='qualified_byte_input_batched_optimizer_state_training_v1',
    previous_recipe=record(previous),sources=[record(ROOT/p['candidate']['path']) for p in mapping['pairs']]+[record(ROOT/em['candidate']['path'])],
    source_maps=[record(source_map),record(export_map)],validator_evidence=record(micro_dir/'result.json'),
    training_evidence=record(train_dir/'result.json'),artifact_interoperation=record(export_dir/'result.json'),
    guards=[record(d/'guard.json') for d in (micro_dir,train_dir,export_dir)],
    training_plan=record(train_dir/'plan.json'),model_and_cache_unchanged=True,
    public_selected_checkpoint_loader_unchanged='research.byte_input_checkpoint_consumer_v3.load',
    scope='Equivalent optimizer-state validation and typed training/state/export consumers. Same selected weights, model math, optimizer updates, formats and quality evidence. Local short warmed throughput only.',
    failure_lineage=[record(BASE/('optimizer-state-runtime'+suffix)/name) for suffix in ('001','002') for name in ('failure.json','guard.json','STOP')])
publish('implementation.json',implementation)
recipe=copy.deepcopy(recipe)
recipe.update(status='adopted_equivalent_batched_optimizer_state_implementation',
    previous_implementation_recipe=record(previous),training_implementation=record(OUT/'implementation.json'),
    training_entrypoints=dict(model='research.byte_input_model_v5',optimizer='research.byte_input_optimizer_v6',
        numerical_state='research.byte_input_state_v5',training='research.byte_input_training_v5',
        inference_export='research.byte_input_inference_v4',cache='research.byte_input_cache_v3'),
    training_implementation_cost_evidence=record(train_dir/'result.json'),
    training_implementation_limits=implementation['scope'])
publish('recipe.json',recipe)
publish('closure.json',dict(status='closed_adopted_equivalent_optimizer_state_training_implementation',
    recipe=record(OUT/'recipe.json'),implementation=record(OUT/'implementation.json'),
    median_mps_update_latency_reduction=1-train['median_candidate_over_reference'],
    exact_training_and_resume=True,exact_saved_raw_h200_interoperation=True,quality_panel_repeated=False,
    no_new_weights=True,adoption_used_saved_evidence_only=True))
publish('STOP',dict(status='completed_closed_adopted_equivalent_optimizer_state_implementation',no_replay=True))
print(json.dumps(dict(adopted=True,recipe=str(OUT/'recipe.json'),median_latency_reduction=1-train['median_candidate_over_reference'])))
