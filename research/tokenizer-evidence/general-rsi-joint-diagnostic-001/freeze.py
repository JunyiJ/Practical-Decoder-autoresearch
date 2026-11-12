"""Freeze all fresh roles, probes and full diagnostic cost before model scoring."""
import ast,fcntl,json,os,shutil,time
from pathlib import Path
from research.general_rsi_joint_diagnostic_v1 import ROOT,OUT,ARMS,inputs,scorer_oracle,sha,put,check_stop
from research.general_rsi_data_v1 import GeneralData
os.chdir(ROOT);tick=time.monotonic();check_stop()
lock=open('research_state/tokenizer-worker-lock/.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
prior=json.loads(Path('research/tokenizer-evidence/general-rsi-pseudo-accumulated-001/plan.json').read_text())
qualified=json.loads(Path('research/tokenizer-evidence/general-rsi-joint-integration-001/plan.json').read_text())
review=json.loads(Path('research/tokenizer-evidence/general-rsi-joint-integration-001/completion-review.json').read_text());assert review['accepted']
sources=dict(qualified['sources'])
for p,h in sources.items():assert sha(p)==h,p
for p in ['research/general_rsi_joint_diagnostic_v1.py','research/general_rsi_joint_diagnostic_analysis_v1.py','research/GENERAL_RSI_JOINT_DIAGNOSTIC_NEXT.md',str(Path(__file__).resolve().relative_to(ROOT)),'research/tokenizer-evidence/general-rsi-joint-integration-001/attempt-01/result.json','research/tokenizer-evidence/general-rsi-joint-integration-001/completion-review.json']:
    if p.endswith('.py'):ast.parse(Path(p).read_text())
    sources[p]=sha(p)
G=1024**3;M=1024**2;required=10*G+128*M
free=shutil.disk_usage(ROOT).free;assert free>=required+2*G
parents=[]
for i,old in enumerate(prior['parents'],1):
    roles=dict(zip(['native_initializer','composer_initializer','train_sampler','validation_sampler'],range(62111209020000+i*100+1,62111209020000+i*100+5)))
    c=dict(parent=old['parent'],roles=roles,pseudo_seed=62111209020000+i*100+5,evaluation_seed=62111209020000+i*100+6,repetition_seed=62111209020000+i*100+7,arm_order=ARMS[i-1:]+ARMS[:i-1])
    assert sha(c['parent']['path'])==c['parent']['sha256']
    data=GeneralData(roles['train_sampler'],roles['validation_sampler'],'cpu');inventory,_,_=inputs(c,data);c['inputs']=inventory;parents.append(c)
plan=dict(kind='joint_gradient_diagnostic_001',created_unix=time.time(),sources=sources,torch_version=qualified['torch_version'],parents=parents,
    probes=[1,10,25,50,100],natural_updates=100,arms=ARMS,clock_budget_ns=1200_000_000_000,half_life_ns=1_000_000_000,
    timeout_seconds=1200,rss_limit_bytes=4*G,mps_limit_bytes=6*G,reserve_bytes=10*G,metadata_cap_bytes=64*M,required_free_bytes=required,
    source_protocol='research/GENERAL_RSI_JOINT_DIAGNOSTIC_NEXT.md',seed_audit='Prefix6211120902 absent from research/state JSON/JSONL/PY/MD before new files.',
    primary_horizon=100,contrasts=['joint_minus_full','joint_minus_half','half_minus_full'],
    directional_support='At100, joint minus both controls negative in every parent for natural CE, pseudo penalty, mean repeated probability and aggregate repeated/previous probability ratio. Descriptive only; no adoption.',
    test_opened=False,validation_scored=False,new_saved_models=0,new_generations=0)
put(OUT/'plan.json',plan)
qr=json.loads(Path('research/tokenizer-evidence/general-rsi-joint-integration-001/attempt-01/result.json').read_text())
cost=dict(training=300*sum(r['mean_update_seconds'] for r in qr['reports']),natural_scoring=48*22.063626*8192/100416,
    synthetic_scoring=384*.18,loads=90,state_checks_and_digests=30,bindings_and_inputs=20,logging_and_overhead=30)
forecast=sum(cost.values());assert forecast<1200
put(OUT/'admission.json',dict(admitted=True,created_unix=time.time(),plan_sha256=sha(OUT/'plan.json'),actual_free_bytes=free,
    required_free_bytes=required,extra_headroom_bytes=2*G,metadata_cap_bytes=64*M,serialization_bytes=0,overlap_bytes=64*M,
    cost_components_seconds=cost,forecast_seconds=forecast,ceiling_seconds=1200,remaining_margin_seconds=1200-forecast,
    measured_basis='Qualified mean update costs and prior22.063626seconds/100416 natural targets. Synthetic scoring0.18s/batch and other overhead are conservative allowances, not new measurements.',
    checks=dict(shared_nonblocking_lock=True,no_worker_on_process_inspection=True,parent_hashes=3,source_hashes=len(sources),
        disjoint_input_inventories=3,new_aggregation_oracle=scorer_oracle(),new_source_ast_valid=True),
    preparation_seconds=time.monotonic()-tick,cleanup_needed=False))
print(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),forecast_seconds=forecast,free_bytes=free,parents=[dict(slot=c['parent']['slot'],rep_proposals=c['inputs']['repetition_proposals']) for c in parents])))
