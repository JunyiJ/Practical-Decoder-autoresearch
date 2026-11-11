"""Freeze fresh matched-update FP32 quality roles and measured complete admission."""
import ast
import datetime
import fcntl
import json
from pathlib import Path
import shutil
import statistics
from research.general_rsi_training_v2 import sha
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/tokenizer-evidence/general-rsi-composer-fp32-quality-001'
Q=P/'qualification'

def put(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n')

def main():
    q=json.loads((Q/'result.json').read_text());assert q['integration_qualified']
    guard=json.loads((Q/'guard.json').read_text());assert guard['status']=='finished' and guard['reason'] is None
    qp=json.loads((Q/'plan.json').read_text())
    for path,h in qp['sources'].items():assert sha(ROOT/path)==h,path
    old=ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001'
    assert json.loads((old/'completion-review.json').read_text())['all_144_first_outputs_rated_blind']
    native=json.loads((old/'plan.json').read_text())
    sources=dict(qp['sources'])
    for name in ('research/general_rsi_fp32_quality_prepare_v1.py','research/general_rsi_fp32_quality_worker_v1.py','research/general_rsi_fp32_quality_review_v1.py','research/general_rsi_generation_v2.py'):
        sources[name]=sha(ROOT/name)
    slots=[]
    for b in range(3):
        roles=dict(zip(('native_initializer','composer_initializer','train_sampler','validation_sampler'),range(62111118001+b*100,62111118005+b*100)))
        order=['existing_byte','byte_composer_fp32'] if b!=1 else ['byte_composer_fp32','existing_byte']
        for arm in order:slots.append(dict(slot=len(slots),phase='confirmation',bundle=b+1,arm=arm,lr=.0025,horizon=10240,roles=roles,timeout_seconds=14400))
    assert len({v for s in slots[::2] for v in s['roles'].values()})==12
    byte_prior=json.loads((ROOT/'research/tokenizer-evidence/general-rsi-exposure-001/slot-05/result.json').read_text())
    measured_final=max(e['weights']['size'] for e in byte_prior['endpoints'])
    assert measured_final<64*1024**2
    receipts=[a['checkpoint'] for a in q['arms']]
    for r in receipts:assert Path(r['path']).stat().st_size==r['size'] and sha(r['path'])==r['sha256']
    final_cap=12*64*1024**2;restart_cap=2*512*1024**2;metadata=256*1024**2;failed=5*1024**3;reserve=10*1024**3
    incremental=final_cap+restart_cap+metadata+failed
    # Qualification outputs and all earlier retained payloads already consume
    # actual filesystem space. They are not subtracted twice from free bytes.
    timing=max(a['mean_update_seconds'] for a in q['arms'])
    projected=61440*timing
    forecast=projected*1.6+7200
    assert forecast<86400
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        free=shutil.disk_usage(ROOT).free;assert free>=incremental+reserve
        plan=dict(kind='general_rsi_composer_fp32_quality_panel_v1',sources=sources,torch_version=qp['torch_version'],codebook_assets=qp['codebook_assets'],slots=slots,
            seed_role_audit='Before assignment rg exact13 roles over research/research_state JSON/JSONL returned no matches. Three paired initializers/samplers, unique across bundles. Bundle2 order reversed prospectively.',
            qualification=dict(plan_sha256=sha(Q/'plan.json'),result_sha256=sha(Q/'result.json'),guard_sha256=sha(Q/'guard.json')),
            schedule='linear cooldown floor(2H/3)..H to.1; H10240',averaging='H200; snapshots6830..10240 every10 updates;342 snapshots; RAW secondary',
            checkpoint_interval_steps=1024,probe_steps=[3413,6826],probe_documents=native['probe_documents'],generation_protocol=native['generation_protocol'],
            acceptance=dict(primary='h200',delta='composer_fp32 minus existing_byte',minimum_mean_gain=.02,all_three_same_sign=True,bootstrap_seed=62111118999,replicates=10000,upper_quantile=.99,method='PCG64 resample108documentIDs; average paired seed loss sums before resampling; sampled target-count denominator; linear99th percentile upper<0 in winner direction',source_regression_max=.05,symmetric=True,neither_direction='inconclusive; no equivalence claim',generation='separate diagnostics, no adoption gate',arithmetic='FP64 and cross-precision discrepancies diagnostic-only; no threshold veto'),
            max_attempts_per_slot=2,tranche_seconds=86400,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,reserve_bytes=reserve,incremental_peak_bytes=incremental,
            forecast=dict(measured_final_weight_bytes=measured_final,measured_restart_bytes=max(r['size'] for r in receipts),final_weight_cap_bytes=final_cap,restart_with_serialization_overlap_cap_bytes=restart_cap,metadata_cap_bytes=metadata,failed_write_allowance_bytes=failed,qualification_retained_bytes=sum(r['size'] for r in receipts),qualification_already_in_actual_disk_usage=True,prior_fresh_native_weights_retained=12,prior_retention_receipt_sha256=sha(old/'completion-review.json'),all_old_artifacts_already_in_actual_disk_usage=True,free_bytes_at_freeze=free,required_free_bytes=incremental+reserve,projected_training_seconds=projected,timing_multiplier=1.6,io_evaluation_generation_failure_margin_seconds=7200,total_forecast_seconds=forecast,fixture_seconds_separate=guard['elapsed_seconds']),
            recovery='Implementation/resource recovery only, preserve failed attempts. No seed/LR changes for divergence; rerolled generations prohibited. Every attempt charged to cumulative slot/tranche guard. Uncheckpointed time retained in lost_training_ns; exact matched-update endpoint restored.',test_opened=False)
        put(P/'plan.json',plan)
        put(P/'admission.json',dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),plan_sha256=sha(P/'plan.json'),qualification_passed=True,free_bytes=free,complete_required_free_bytes=incremental+reserve,headroom_bytes=free-incremental-reserve,shared_lock_available=True,one_local_worker_only=True,cleanup_required=False,test_opened=False))
    # Unchanged operational safeguards are inherited, not numerically rerun.
    def fn(path,name):
        return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name))
    oldworker=ROOT/'research/general_rsi_native_horizon_worker_v1.py';newworker=ROOT/'research/general_rsi_fp32_quality_worker_v1.py'
    for name in ('put','replace','stop','stopped','boot','retire','remaining_guard_seconds'):assert fn(oldworker,name)==fn(newworker,name)
    put(P/'preflight.json',dict(plan_sha256=sha(P/'plan.json'),unchanged_operational_function_ASTs=['put','replace','stop','stopped','boot','retire','remaining_guard_seconds'],source_bindings_verified=len(sources),exact_new_runtime_restart_qualified=True,old_scorer_generator_reused=True,fresh_roles_count=13,complete_resource_forecast=True,test_opened=False))
    print(sha(P/'plan.json'),free,incremental+reserve,forecast)
if __name__=='__main__':main()
