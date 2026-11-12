import ast,fcntl,json,shutil,time
from pathlib import Path
from research.general_rsi_pseudo_accumulated_v1 import inputs,OUT
from research.general_rsi_pseudo_update_diagnostic_v1 import put,sha,batch_hash
from research.general_rsi_data_v1 import GeneralData
import torch
start=time.monotonic()
lock=open('research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.load(open('research/tokenizer-evidence/general-rsi-pseudo-update-001/plan.json'))
for p,h in old['sources'].items():assert sha(p)==h,p
orders=[['ce','ce_zero','ce_pseudo'],['ce_zero','ce_pseudo','ce'],['ce_pseudo','ce','ce_zero']]
parents=[]
for i,prior in enumerate(old['parents']):
    base=62111208250100+i*100
    roles=dict(zip(('native_initializer','composer_initializer','train_sampler','validation_sampler'),range(base+1,base+5)))
    config=dict(parent=prior['parent'],roles=roles,pseudo_seed=base+5,evaluation_seed=base+6,arm_order=orders[i])
    r=config['parent'];assert Path(r['path']).stat().st_size==r['size'] and sha(r['path'])==r['sha256']
    data=GeneralData(roles['train_sampler'],roles['validation_sampler'],'cpu')
    config['inputs']=inputs(config,data)[0]
    # Check only the newly frozen input inventory against the existing sampler.
    # No model, gradient, restart fixture or quality prerequisite is run here.
    for row in config['inputs']['natural']:assert batch_hash(*data.batch())==row['sha256']
    parents.append(config)
sources=dict(old['sources'])
for path in ['research/general_rsi_pseudo_accumulated_v1.py',__file__,
             'research/GENERAL_RSI_PSEUDO_ACCUMULATED_NEXT.md',
             'research/tokenizer-evidence/general-rsi-pseudo-update-001/analysis.json',
             'research/tokenizer-evidence/general-rsi-pseudo-update-001/completion-review.json',
             'research/tokenizer-evidence/general-rsi-pseudo-mechanism-001/literature-review.json']:
    path=str(Path(path).resolve().relative_to(Path.cwd()));sources[path]=sha(path)
plan=dict(kind='pseudo_accumulated_state_diagnostic_v1',created_unix=time.time(),torch_version=torch.__version__,sources=sources,
 parents=parents,probe_natural_updates=[1,10,25,50,100],natural_updates_per_arm=100,extra_updates_per_interleaved_arm=99,
 ema_half_life_ns=old['ema_half_life_ns'],lr=.00025,lr_policy='constant; no EMA or cooldown in diagnostic',
 scope='Fresh100-natural-update trajectories to distinguish accumulated extra optimizer advances from pseudo-gradient effects. No quality confirmation, tuning sweep or model adoption.',
 controls='Same complete pristine parent state and exact natural prefixes in3arms per parent. Explicit all-parameter zero gradients versus original pseudo gradients, both99extra steps between100CE steps. No extra step after final CE.',
 contrasts=['ce_zero-ce','ce_pseudo-ce_zero','ce_pseudo-ce'],
 interpretation='All curves and per-parent endpoints retained. Positive zero-minus-CE supports accumulated advance/decay effect; positive pseudo-minus-zero supports additional pseudo-gradient effects including optimizer interactions. Mixed/negligible effects weaken simple accounts. No confidence bound or adoption gate.',
 evaluation='64training-doc-local windows128targets each per parent, disjoint from all100CE and99pseudo source intervals and each other. Same qualified FP32RMSNorm/MPSCE scorer; training-only, not held-out. Before first CE once and after CE1,10,25,50,100 before extra advance. Exact CE1 equality acrossarms.',
 seed_role_audit='No prior6211120825 matches before source/protocol freeze. New62111208250101..106,62111208250201..206,62111208250301..306.',
 retention='All1494step records,45probes,3initialparent evaluations,9trajectory results, summaries, exactinputs, launch/guard/failure logs retained. No tensor checkpoint serialization even on failure. No generation.',
 timeout_seconds=1200,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,reserve_bytes=10*1024**3,
 retained_output_cap_bytes=16*1024**2,atomic_overlap_bytes=16*1024**2,test_opened=False,validation_scored=False)
put(OUT/'plan.json',plan)
free=shutil.disk_usage('.').free
components=dict(optimizer_advances_1494=1494*.4,evaluation_passes_48=48*8192*22.063625833950937/100416,
 loads=30,restores=9,state_summary_and_displacement=54,binding_and_inputs=15,logging_and_boundary_checks=1494*.02)
forecast=sum(components.values());required=plan['reserve_bytes']+plan['retained_output_cap_bytes']+plan['atomic_overlap_bytes']
assert free>=required+2*1024**3 and forecast<1200
put(OUT/'admission.json',dict(plan_sha256=sha(OUT/'plan.json'),admitted=True,created_unix=time.time(),
 actual_free_bytes=free,required_free_bytes=required,extra_headroom_bytes=2*1024**3,
 existing_outputs='All retained outputs already deducted from free bytes. No cleanup needed.',
 checkpoint_serialization_bytes=0,retained_cap_bytes=plan['retained_output_cap_bytes'],overlap_bytes=plan['atomic_overlap_bytes'],
 cost_components_seconds=components,forecast_seconds=forecast,ceiling_seconds=1200,failure_variability_allowance_seconds=1200-forecast,
 measured_basis='Qualified pseudo update0.370253s and scorer22.063626s/100416targets. Charge every zero/CE/pseudo step0.4s. Prior common-state diagnostic247.653s, peakRSS2046558208bytes, sampledMPS2497495040bytes. New model/optimizer unchanged; pristine capture smaller than prior populated capture. Other components are conservative allowances.',
 checks=dict(nonblocking_lock_acquired=True,no_worker_on_process_inspection=True,all_parent_hashes_verified=True,
 sampler_inventory_matches_original_300_batches=True,source_hashes=len(sources),new_source_ast_valid=isinstance(ast.parse(Path('research/general_rsi_pseudo_accumulated_v1.py').read_text()),ast.Module)),
 preparation_seconds=time.monotonic()-start))
print(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),forecast_seconds=forecast,free_bytes=free,preparation_seconds=time.monotonic()-start)))
