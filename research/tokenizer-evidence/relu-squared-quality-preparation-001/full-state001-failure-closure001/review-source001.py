"""One-time metadata/source/first-recorded-streamhash closure; no tensor loads."""
import argparse,fcntl,hashlib,json,os,stat
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path(__file__).resolve().parents[4]
R=Path(__file__).resolve().parent
B=R.parent
A=ROOT/'research_runs/tokenizer-relu-squared-full-state-v1-001'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for v in iter(lambda:f.read(1024**2),b''):h.update(v)
 return h.hexdigest()
def put(p,v):
 raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 assert sum(x.stat().st_size for x in R.rglob('*') if x.is_file())+len(raw)<=16*1024**2
 with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def read(p):return json.loads(Path(p).read_text())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);args=ap.parse_args()
 assert sha(R/'plan.json')==args.plan_sha and not (R/'attempt.json').exists()
 plan=read(R/'plan.json')
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  for pid in plan['exited_pids']:
   try:os.kill(pid,0)
   except ProcessLookupError:pass
   else:raise AssertionError('Original process remains active')
  put(R/'attempt.json',dict(pid=os.getpid(),plan_sha256=args.plan_sha))
  try:
   with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    f=B/'full-state001';assert read(f/'STOP')['status']==read(A/'STOP')['status']=='closed_failed'
    assert read(f/'native_swiglu-process.json')['returncode']==1
    g=read(f/'native_swiglu-mps-guard.json');assert g['status']=='fixture_exception' and g['elapsed_seconds']<600
    assert '85 / 281' in g['reason'] and '0.0002193450927734375' in g['reason']
    log=(f/'native_swiglu-process.log').read_text();assert '"step": 1200, "ema_snapshots": 41' in log and 'full-state001.py", line 171' in log
    assert not (f/'native_swiglu-result.json').exists() and not (f/'completed.json').exists()
    for arm in ['ungated_silu','ungated_relu_squared']:assert not (f/(arm+'-attempt.json')).exists() and not (A/arm).exists()
    cpu=[read(p) for p in sorted(f.glob('native_swiglu-cpu-*.json'))];assert len(cpu)==4 and all(x['status']=='finished' and x['reason'] is None for x in cpu)
    assert read(B/'role-scan001/result.json')['status']=='passed_new_literal_bounded_role_stream_oracle'
    c=read(B/'full-proof-census001/result.json');assert c['status']=='new_full_horizon_fixture_roles_unused_before_construction' and not c['prior_matches']
    t=read(B/'full-config-statistics001/result.json');assert t['status']=='passed_new_full_horizon_config_selection_statistics_fixture' and t['valid_configs']==18 and len(t['negative_cases'])==102
    weights=[]
    for item in plan['weights']:
     p=ROOT/item['path'];s=p.lstat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink() and s.st_nlink==1
     assert dict(size=s.st_size,inode=s.st_ino,device=s.st_dev)=={k:item[k] for k in ['size','inode','device']}
     h=sha(p);q=p.lstat();assert (s.st_dev,s.st_ino,s.st_size,s.st_nlink)==(q.st_dev,q.st_ino,q.st_size,q.st_nlink)
     weights.append(dict(item,sha256=h,singlelink=True,hash_scope='FIRST independent recorded streamhash; original publish refs were local-only,not a comparison with a persisted original weight hash'))
    mirrored=[]
    for item in plan['metadata']:
     p=ROOT/item['path'];assert not p.is_symlink() and sha(p)==item['sha256']
     dest=R/'originals'/item['path'];dest.parent.mkdir(parents=True,exist_ok=True);raw=p.read_bytes()
     with dest.open('xb') as out:out.write(raw);out.flush();os.fsync(out.fileno())
     assert sha(dest)==item['sha256'];mirrored.append(item)
    assert sum(x.stat().st_size for x in R.rglob('*') if x.is_file())<16*1024**2
    for item in plan['metadata']:assert sha(ROOT/item['path'])==item['sha256']
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    maximum=0.0002193450927734375;tol=3e-5;n=281;failed=85
    put(R/'review.json',dict(status='closed_failed_original_generated1200_fixture_metadata_preserved',sources_unchanged=len(plan['sources']),weights=weights,protected_weight_bytes=sum(x['size'] for x in weights),original_metadata=mirrored,mirrored_files=len(mirrored),mirrored_bytes=sum(x['size'] for x in mirrored),
     observed_failure=dict(arm='native_swiglu',completed_logged_updates=1200,completed_logged_EMA=41,per_target_count=281,per_target_mismatches=85,max_abs_ce_difference=maximum,fixed_atol=tol,fixed_rtol=0.,max_index=6),
     inference_limits=dict(no_accepted_arm_result=True,other_two_arms_unattempted=True,no_full_chain_qualification=True,local_arrays_unpersisted_no_reconstruction=True,pre_failure_checks_reached_but_results_local_only=True,no_genuine_CE_gather_backward_after_failure=True,no_final_source_RNG_assertions_reached=True,no_corpus_quality_ranking_or_old_projection_cause=True),
     existing_evidence_only_bound=dict(max_abs_over_atol=maximum/tol,mean_absolute_error_strict_lower_bound=(maximum+(failed-1)*tol)/n,mean_absolute_error_upper_bound=(failed*maximum+(n-failed)*tol)/n,signed_mean_abs_upper_bound=(failed*maximum+(n-failed)*tol)/n,signed_mean_actual_unavailable=True,not_an_aggregate_acceptance_measurement=True),
     no_tensor_model_optimizer_corpus_scoring_bootstrap_census_RNG_replay=True,no_weight_copy_deletion_or_retirement=True,no_retry_or_source_tolerance_seed_backend_horizon_rescue=True))
   put(R/'STOP',dict(status='complete_original_failed_fixture_metadata_closure',no_replay=True,no_retry=True))
  except BaseException as exc:
   if not (R/'STOP').exists():put(R/'STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
   raise
if __name__=='__main__':main()
