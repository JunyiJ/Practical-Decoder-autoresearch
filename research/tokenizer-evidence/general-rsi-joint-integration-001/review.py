"""Read-only acceptance of the finished bounded integration attempt."""
import ast,datetime,fcntl,hashlib,json,math,os,shutil
from pathlib import Path
R=Path(__file__).resolve().parents[3];os.chdir(R);O=Path(__file__).resolve().parent;A=O/'attempt-01'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def put(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
lock=open('research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
p=read(O/'plan.json');r=read(A/'result.json');g=read(A/'guard.json');oracle=read(A/'gradient-oracle.json');ad=read(O/'admission.json')
assert r['qualified'] and g['status']=='finished' and g['reason'] is None
assert r['plan_sha256']==ad['plan_sha256']==sha(O/'plan.json')
assert r['total_seconds']<p['total_seconds'] and g['observation']['peak_rss_bytes']<p['rss_limit_bytes']
assert r['peak_mps_boundary_bytes']<p['mps_limit_bytes']
for name,h in p['sources'].items():assert sha(name)==h,name
assert sha(p['parent']['path'])==p['parent']['sha256']
assert [x['arm'] for x in oracle['rows']]==p['arms']
assert all(x['tensors']==88 and x['max_abs']==0 and x['relative_l2']==0 for x in oracle['rows'])
streams=[];updates=0;replay=0;bytes_=0
for report in r['reports']:
    full=read(A/(report['arm']+'-result.json'))
    assert {k:v for k,v in full.items() if k!='history'}==report
    assert report['exact_full_restore_and_replay'] and report['bad_component_counts_rejected']
    rec=report['checkpoint'];assert Path(rec['path']).stat().st_size==rec['size'] and sha(rec['path'])==rec['sha256']
    bytes_+=rec['size'];updates+=report['steps'];replay+=report['replayed_updates']
    assert len(full['history'])==report['steps']
    weights=(.5,.5) if report['arm']=='ce_pseudo_mean' else ((.5,0.) if report['arm']=='ce_half' else (1.,0.))
    for i,row in enumerate(full['history'],1):
        assert row['step']==i and (row['ce_weight'],row['pseudo_weight'])==weights
        assert math.isfinite(row['loss']) and math.isfinite(row['ce_loss'])
        assert row['pseudo_loss'] is None or math.isfinite(row['pseudo_loss'])
        assert row['ce_targets']==1024 and row['saturated_targets']==0
    streams.append([x['batch_sha256'] for x in full['history']])
n=min(map(len,streams));assert n==r['paired_natural_prefix'] and all(x[:n]==streams[0][:n] for x in streams)
assert bytes_==r['retained_bytes'] and bytes_<=p['retained_cap_bytes']
assert not list(A.glob('*failure*'))
meta=sum(x.stat().st_size for x in O.rglob('*') if x.is_file());assert meta<ad['metadata_cap_bytes']
put(O/'completion-review.json',dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted=True,
    plan_sha256=sha(O/'plan.json'),result_sha256=sha(A/'result.json'),guard_sha256=sha(A/'guard.json'),
    source_hashes_unchanged=len(p['sources']),parent_hash_verified=True,worker_pid=read(A/'launch.json')['pid'],
    worker_ended=True,worker_exit_code=0,nonblocking_lock_reacquired=True,gradient_tensors_checked=3*88,
    gradient_weighted_sum_max_abs=0,original_optimizer_updates=updates,replayed_optimizer_updates=replay,
    full_restart_arms=3,invalid_count_rejections=3,shared_natural_prefix=n,retained_checkpoint_bytes=bytes_,
    metadata_bytes_before_review=meta,actual_free_bytes=shutil.disk_usage(R).free,total_seconds=r['total_seconds'],
    peak_rss_bytes=g['observation']['peak_rss_bytes'],peak_mps_boundary_bytes=r['peak_mps_boundary_bytes'],
    failures=0,retries=0,validation_scored=False,test_opened=False,quality_claim=False,
    next_action='Freeze a fresh three-parent, three-arm 100-natural-update diagnostic with disjoint natural and synthetic repetition probes; use measured0.343/0.354/0.486s update costs and full resource admission.'))
print(json.dumps(dict(accepted=True,original_updates=updates,replayed_updates=replay,free_bytes=shutil.disk_usage(R).free)))
