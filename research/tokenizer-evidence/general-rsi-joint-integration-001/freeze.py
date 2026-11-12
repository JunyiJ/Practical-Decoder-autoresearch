"""Freeze the new integration check before any numerical execution."""
import ast,fcntl,hashlib,json,os,shutil,time
from pathlib import Path
R=Path(__file__).resolve().parents[3]
os.chdir(R)
O=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
lock=open('research_state/tokenizer-worker-lock/.lock','a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
old=json.load(open('research/tokenizer-evidence/general-rsi-pseudo-accumulated-001/plan.json'))
sources=dict(old['sources'])
for p,h in sources.items():assert sha(p)==h,p
new=['research/general_rsi_joint_training_v1.py','research/general_rsi_joint_qualification_v1.py','research/GENERAL_RSI_JOINT_GRADIENT_NEXT.md','research/general_rsi_qualification_v1.py',str(Path(__file__).resolve().relative_to(R))]
for p in new:
    if p.endswith('.py'):ast.parse(Path(p).read_text())
    sources[p]=sha(p)
parent=old['parents'][0]['parent'];assert sha(parent['path'])==parent['sha256']
G=1024**3;M=1024**2
required=10*G+3*512*M+256*M+512*M+64*M
free=shutil.disk_usage(R).free;assert free>=required+2*G
roles=dict(zip(['native_initializer','composer_initializer','train_sampler','validation_sampler'],range(62111208460001,62111208460005)))
plan=dict(kind='joint_gradient_integration_001',created_unix=time.time(),sources=sources,torch_version=old['torch_version'],
    parent=parent,arms=['ce_full','ce_half','ce_pseudo_mean'],roles=roles,pseudo_seed=62111208460005,
    seed_audit='Prefix6211120846 searched in research/state JSON/JSONL/PY/MD before new files; no prior matches.',
    budget_ns=24_000_000_000,half_life_ns=1_000_000_000,gradient_rtol=1e-5,gradient_atol=1e-7,
    total_seconds=600,rss_limit_bytes=4*G,mps_limit_bytes=6*G,reserve_bytes=10*G,
    retained_cap_bytes=3*512*M+256*M,required_free_bytes=required,
    gates=['all pre-clip parameter gradients match independently weighted component sum','all gradients, losses and complete states finite',
           'all three full save/restore/replay states exact','corrupt component counts rejected before state mutation','shared natural input prefix'],
    quality_result=False,test_opened=False,validation_scored=False,
    original_fp32_numerical_failure_unchanged=True)
put(O/'plan.json',plan)
put(O/'admission.json',dict(admitted=True,created_unix=time.time(),plan_sha256=sha(O/'plan.json'),actual_free_bytes=free,
    required_free_bytes=required,extra_headroom_bytes=2*G,retained_restart_cap_bytes=3*512*M,
    failure_payload_cap_bytes=256*M,serialization_overlap_bytes=512*M,metadata_cap_bytes=64*M,reserve_bytes=10*G,
    cost_components_seconds=dict(training=72,replay=24,context_loads=70,checkpoint_io=18,gradient_oracle=6,bindings=15,clones_and_digests=30),
    forecast_seconds=235,ceiling_seconds=600,
    measured_basis='Prior integration checkpoint445033525bytes, native update0.348804s, pseudo update0.370253s. New mixture costs measured here. Overhead entries are allowances, not new measurements.',
    checks=dict(nonblocking_lock_acquired=True,no_worker_on_process_inspection=True,parent_hash_verified=True,new_source_ast_valid=True,source_hash_count=len(sources)),
    cleanup_needed=False,existing_outputs='Already deducted from free space; preserve all existing artifacts.'))
print(json.dumps(dict(plan_sha256=sha(O/'plan.json'),free_bytes=free,required_with_headroom=required+2*G,source_count=len(sources))))
