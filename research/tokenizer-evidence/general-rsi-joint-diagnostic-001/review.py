"""Verify complete frozen diagnostic receipts before interpreting contrasts."""
import datetime,fcntl,json,math,os,shutil
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import sha,put
from research.general_rsi_joint_diagnostic_v1 import ROOT,OUT,ARMS
os.chdir(ROOT);A=OUT/'attempt-01';read=lambda p:json.loads(Path(p).read_text())
lock=open('research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=read(OUT/'plan.json');r=read(A/'result.json');g=read(A/'guard.json');analysis=read(OUT/'analysis.json')
assert g['status']=='finished' and g['reason'] is None
assert r['plan_sha256']==analysis['plan_sha256']==sha(OUT/'plan.json')
assert r['updates']==900 and r['natural_components']==900 and r['pseudo_components']==300
assert r['total_seconds']<p['timeout_seconds'] and g['observation']['peak_rss_bytes']<p['rss_limit_bytes']
assert r['peak_mps_boundary_bytes']<p['mps_limit_bytes']
for path,h in p['sources'].items():assert sha(path)==h,path
probes=0;updates=0
for c in p['parents']:
    assert sha(c['parent']['path'])==c['parent']['sha256'];slot=c['parent']['slot']
    for arm in ARMS:
        rows=[json.loads(l) for l in (A/f'p{slot:02d}-{arm}-steps.jsonl').read_text().splitlines()];assert len(rows)==100
        for n,row in enumerate(rows,1):
            assert row['step']==n and row['lr_ratio']==1 and row['batch_sha256']==c['inputs']['natural'][n-1]['sha256']
            assert math.isfinite(row['loss']) and math.isfinite(row['clip']['pre_norm'])
            if arm=='ce_pseudo_mean':assert row['pseudo_sha256']==c['inputs']['synthetic'][n-1]['sha256'] and row['pseudo_rows']==c['inputs']['synthetic'][n-1]['rows']
            else:assert row['pseudo_sha256'] is None and row['pseudo_targets']==0
            updates+=1
        endpoint=read(A/f'p{slot:02d}-{arm}-result.json');assert endpoint['pristine_initial_state_equal']
        assert endpoint['component_counts']==dict(ce=100,pseudo=100 if arm=='ce_pseudo_mean' else 0)
        for n in p['probes']:
            probe=read(A/f'p{slot:02d}-{arm}-n{n:03d}.json');assert probe['natural_updates']==probe['optimizer_steps']==n
            natural=probe['scores']['natural'];rep=probe['scores']['repetition'];assert natural['targets']==8192 and len(natural['window_loss_sums'])==64
            assert math.fsum(natural['window_loss_sums'])/8192==natural['ce']
            assert len(rep['batches'])==8 and sum(x['targets'] for x in rep['batches'])==rep['targets']
            ps=math.fsum(x['probability_sum'] for x in rep['batches']);prev=math.fsum(x['previous_probability_sum'] for x in rep['batches'])
            assert ps/rep['targets']==rep['mean_probability'] and prev/rep['targets']==rep['mean_previous_probability'] and ps/prev==rep['probability_ratio']
            assert math.fsum(x['penalty_sum'] for x in rep['batches'])/rep['targets']==rep['penalty']
            if n==100:assert probe['scores']==endpoint['final_scores']
            probes+=1
assert updates==900 and probes==45 and not list(A.glob('*failure*'))
metadata=sum(x.stat().st_size for x in OUT.rglob('*') if x.is_file());assert metadata<p['metadata_cap_bytes']
put(OUT/'completion-review.json',dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_execution=True,
    plan_sha256=sha(OUT/'plan.json'),result_sha256=sha(A/'result.json'),analysis_sha256=sha(OUT/'analysis.json'),source_hashes_unchanged=len(p['sources']),
    selected_parent_hashes_verified=3,all_updates_reviewed=updates,all_arm_probes_reviewed=probes,pristine_initial_state_comparisons=9,
    worker_pid=read(A/'launch.json')['pid'],worker_exit_code=0,worker_ended=True,nonblocking_lock_reacquired=True,
    metadata_bytes_before_review=metadata,retained_cap_bytes=p['metadata_cap_bytes'],actual_free_bytes=shutil.disk_usage(ROOT).free,
    total_seconds=r['total_seconds'],synchronized_update_seconds=r['synchronized_update_seconds'],peak_rss_bytes=g['observation']['peak_rss_bytes'],
    peak_mps_boundary_bytes=r['peak_mps_boundary_bytes'],failures=0,retries=0,new_model_payloads=0,validation_scored=False,test_opened=False,
    quality_decision_changed=False,directional_mechanism_support=analysis['directional_mechanism_support']))
print(json.dumps(dict(accepted=True,updates=updates,probes=probes)))
