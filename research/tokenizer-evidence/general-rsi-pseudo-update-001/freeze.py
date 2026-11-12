import ast,fcntl,json,time,shutil,hashlib
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import inputs,put,sha,OUT
from research.general_rsi_data_v1 import GeneralData
import torch
start=time.monotonic()
lock=open('research_state/tokenizer-worker-lock/.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.load(open('research/tokenizer-evidence/general-rsi-pseudo-repetition-001/plan.json'))
for p,h in old['sources'].items(): assert sha(p)==h,p
parents=[]
for i,slot in enumerate((0,2,4)):
    base=62111208060100+i*100
    roles=dict(zip(('native_initializer','composer_initializer','train_sampler','validation_sampler'),range(base+1,base+5)))
    config=dict(parent=old['slots'][slot]['parent'],roles=roles,pseudo_seed=base+5,ce_seed=base+6,evaluation_seed=base+7)
    r=config['parent'];assert Path(r['path']).stat().st_size==r['size'] and sha(r['path'])==r['sha256']
    config['inputs']=inputs(config,GeneralData(roles['train_sampler'],roles['validation_sampler'],'cpu'))[0]
    parents.append(config)
source='research/general_rsi_pseudo_update_diagnostic_v1.py'
sources=dict(old['sources']);sources[source]=sha(source)
sources['src/research/cpu_prerequisite_guard.py']=sha('src/research/cpu_prerequisite_guard.py')
sources['research/GENERAL_RSI_PSEUDO_MECHANISM_BOUNDARY.md']=sha('research/GENERAL_RSI_PSEUDO_MECHANISM_BOUNDARY.md')
sources['research/tokenizer-evidence/general-rsi-pseudo-mechanism-001/matched-prefix.json']=sha('research/tokenizer-evidence/general-rsi-pseudo-mechanism-001/matched-prefix.json')
plan=dict(kind='common_state_pseudo_update_diagnostic_v1',created_unix=time.time(),torch_version=torch.__version__,sources=sources,
 parents=parents,branches=['identity','zero','pseudo','ce'],ema_half_life_ns=old['ema_half_life_ns'],
 scope='Mechanism diagnostic only. Three selected parents each after one fresh CE step. Eight sampled pseudo/CE cases share that one parent state. All96 branches retained. No adoption or quality confidence interval.',
 predictions=['pseudo minus zero positive natural loss supports a local direct-gradient interference component',
 'zero minus identity positive natural loss supports a momentum/independent-decay advance component',
 'mixed or negligible effects oppose a simple one-step account and motivate accumulated-state or span-distribution analysis'],
 controls='Explicit zero gradients for every parameter. Identity does not step. CE and pseudo each take one original optimizer step. All4 start from byte-identical complete capture; restore digest checked every branch.',
 optimizer_policy='base LR0.00025, Muon scale0.5, child weight_decay0; wrapper independent multiplier1-0.00006 each step. Original clipping1.0. No sweep, batch-reuse arm, schedule change, final model or generation.',
 evaluation='Same64 training-document-local128-target windows per parent, uniformly sampled eligible document then start. All windows disjoint from each other and all seed/CE/pseudo source intervals. Exclusions across all8 cases. Parents previously trained on corpus; not a holdout. FP32 inference RMSNorm and MPS CE with FP64 CPU aggregation, as qualified scorer.',
 contrasts=['pseudo-zero','zero-identity','pseudo-identity','ce-identity'],
 reporting='All paired cases and per-parent means/ranges/sign counts. Repeated states/windows correlated; no independent-sample inference. Displacements/gradient-update cosine per Muon/AdamW group; zero and identity cosine undefined. No null/negative suppression.',
 retention='Immutable plan, sources, launch, all96 case records,3parent records, guard, result or failure and stdout. No tensor/checkpoint serialization even on failure. All original evidence untouched.',
 seed_role_audit='No existing6211120806 matches before freeze. New per-parent seeds62111208060101..107,62111208060201..207,62111208060301..307.',
 timeout_seconds=600,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,reserve_bytes=10*1024**3,
 retained_output_cap_bytes=16*1024**2,atomic_overlap_bytes=16*1024**2,test_opened=False,validation_scored=False)
OUT.mkdir(exist_ok=False)
put(OUT/'plan.json',plan)
free=shutil.disk_usage('.').free
components=dict(updates_75=75*.371,evaluation_99_passes=99*8192*old['forecast']['measured_full_validation_seconds']/100416,
 model_loads=30,restore_96=96,complete_state_digests=90,displacement_and_gradient_copies=30,input_and_source_binding=15)
forecast=sum(components.values());required=plan['reserve_bytes']+plan['retained_output_cap_bytes']+plan['atomic_overlap_bytes']
assert free>=required+2*1024**3 and forecast<600
put(OUT/'admission.json',dict(plan_sha256=sha(OUT/'plan.json'),admitted=True,created_unix=time.time(),
 actual_free_bytes=free,required_free_bytes=required,extra_headroom_bytes=2*1024**3,
 existing_retained_outputs='Already deducted from free bytes. No cleanup required.',
 checkpoint_serialization_bytes=0,components_seconds=components,forecast_seconds=forecast,
 failure_and_variability_allowance_seconds=600-forecast,ceiling_seconds=600,
 cost_evidence='Original pseudo integration mean update0.370253s; full document scorer22.063626s/100416targets. Other components are explicit conservative allowances, not measured diagnostic timings. Same-source/binding read34MB. Full state~445MB, two CPU clones plus gradient114MB and streamed FP64 largest matrix~310MB alongside MPS state; prior peakRSS2.386GB/MPS2.502GB. RSS4GiB/MPS6GiB guarded. No new prerequisite is rerun.',
 checks=dict(source_hashes=len(sources),parents=3,inputs_frozen=True,nonblocking_lock_acquired=True,no_training_worker_on_process_inspection=True,
 source_ast_valid=isinstance(ast.parse(Path(source).read_text()),ast.Module)),
 preparation_seconds=time.monotonic()-start))
print(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),free_bytes=free,required_bytes=required,forecast_seconds=forecast,preparation_seconds=time.monotonic()-start)))
