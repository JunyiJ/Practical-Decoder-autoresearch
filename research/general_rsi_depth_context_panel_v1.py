"""Freeze two-depth scientific protocol only after complete measured admission."""
import datetime,json,math,platform,re,shutil,subprocess
from pathlib import Path
import torch
from research.general_rsi_training_v2 import sha
from research.general_rsi_context_length_worker_v1 import put
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-quality-001'
INTEGRATION=ROOT/'research/tokenizer-evidence/general-rsi-depth-context-integration-003'
GIB=2**30

def read(p):return json.loads(Path(p).read_text())
def available_pages(vm):
    size=int(re.search(r'page size of (\d+) bytes',vm).group(1))
    counts=[int(re.search(r'^'+re.escape(name)+r':\s+(\d+)\.',vm,re.M).group(1)) for name in ('Pages free','Pages inactive','Pages speculative')]
    return size*sum(counts)
def main():
    from research.general_rsi_depth_context_analysis_v1 import fixture
    from research.general_rsi_depth_context_flops_v1 import estimate
    from research.general_rsi_depth_context_integration_v3 import host_memory
    qualification=read(INTEGRATION/'completion-review.json');assert qualification['accepted']
    prior=read(INTEGRATION/'plan.json');result=read(INTEGRATION/'attempt-01/result.json')
    assert result['accepted'] and result['plan_sha256']==sha(INTEGRATION/'plan.json') and qualification['result_sha256']==sha(INTEGRATION/'attempt-01/result.json')
    assert all(sha(ROOT/n)==h for n,h in prior['sources'].items())
    sources=dict(prior['sources'])
    files=['research/GENERAL_RSI_DEPTH_CONTEXT_QUALITY_DESIGN.md','research/general_rsi_depth_context_worker_v1.py','research/general_rsi_depth_context_review_v1.py','research/general_rsi_depth_context_analysis_v1.py','research/general_rsi_depth_context_flops_v1.py','research/general_rsi_depth_context_engineering_review_v1.py','research/general_rsi_context_length_score_v1.py','research/general_rsi_native_horizon_review_v1.py','research/general_rsi_time_journal_v1.py','research/tokenizer-evidence/general-rsi-context-transfer-two-depth-planning-001/forecast.json']
    for f in [Path(__file__),INTEGRATION/'completion-review.json',INTEGRATION/'attempt-01/result.json',*[ROOT/x for x in files]]:sources[str(f.relative_to(ROOT))]=sha(f)
    for f in (INTEGRATION/'attempt-01').glob('*/guard.json'):sources[str(f.relative_to(ROOT))]=sha(f)
    roles=[]
    for b in (1,2,3):
        base=62111423050000+b*100
        roles.append(dict(roles=dict(zip(['native_initializer','composer_initializer','train_sampler','validation_sampler'],range(base+1,base+5))),extra_group_seeds=list(range(base+5,base+8))))
    order=[(1,4,[128,256]),(1,16,[256,128]),(2,16,[128,256]),(2,4,[256,128]),(3,4,[128,256]),(3,16,[256,128])]
    slots=[]
    for b,d,contexts in order:
        for c in contexts:slots.append(dict(slot=len(slots),phase='confirmation',arm=f'context{c}',depth=d,context=c,bundle=b,lr=.0025,budget_ns=14400*10**9,timeout_seconds=21600,**roles[b-1]))
    measurements={(x['shape']['depth'],x['shape']['context']):x for x in result['summaries']}
    assert set(measurements)=={(4,128),(4,256),(16,128),(16,256)}
    index=[json.loads(x) for x in (ROOT/'data/raw/rsi-general-10m-v1/val.index.jsonl').read_text().splitlines()]
    probes=[0,1,2,3,85,86,87,88];probe_targets=sum(index[i]['tokens']-1 for i in probes)
    components=dict(training=172800,endpoint_scoring=0.,probes=0.,generation=0.,checkpoint_readback=0.,outside_clock_logs=0.,initialization=12*15.)
    for s in slots:
        m=measurements[s['depth'],s['context']]
        # Training-document evaluator PLUS independent FP64 oracle measured for
        # every native/common context. Charging this whole value for EACH full
        # pass deliberately overestimates evaluator cost.
        per_target=m['evaluation_and_oracle_seconds']/(prior['training_document']['tokens']-1)
        components['endpoint_scoring']+=(4 if s['context']==256 else 2)*100416*per_target
        components['probes']+=2*probe_targets*per_target
        components['generation']+=1.5*m['generation_seconds']
        components['checkpoint_readback']+=122*max(3.,1.5*m['restart_io_seconds'])
        components['outside_clock_logs']+=.01*14400/m['steady_median_seconds']
    engineering_guards=[ROOT/'research/tokenizer-evidence/general-rsi-depth-context-fixture-002/attempt-01/guard.json',ROOT/'research/tokenizer-evidence/general-rsi-depth-context-integration-002/attempt-01/d4-c128/guard.json',*(INTEGRATION/'attempt-01').glob('*/guard.json')]
    engineering_seconds=sum(read(f)['elapsed_seconds'] for f in engineering_guards)
    assert engineering_seconds<3600
    full_tranche=228240;tranche=full_tranche-3600
    forecast=sum(components.values());assert forecast<tranche,(forecast,components)
    # Parameter and serialized metadata growth are bounded prospectively.
    endpoint_bytes=sum(3*sum(v['size'] for v in m['endpoint_receipts'].values()) for m in measurements.values())
    final_endpoint_cap=math.ceil(endpoint_bytes*1.02)
    restart_cap=640*2**20
    max_restart=max(v['size'] for m in measurements.values() for v in m['restart_receipts'])
    assert max_restart+64*2**20<restart_cap,(max_restart,restart_cap)
    retained=result['retained_bytes'];assert retained<GIB
    storage=dict(all24_endpoints=final_endpoint_cap,active_restart_atomic_overlap=2*restart_cap,metadata=GIB,engineering=GIB,failed_attempt_allowance=2*GIB,reserve=10*GIB,extra_headroom=2*GIB)
    reserve=12*GIB;incremental=sum(storage.values())-reserve-retained
    free=shutil.disk_usage(ROOT).free;assert free>=reserve+incremental,(free,reserve+incremental)
    # Assess actual pressure/headroom through observed phases, not parameter size.
    samples=[json.loads(line) for f in (INTEGRATION/'attempt-01').glob('*/memory.jsonl') for line in f.read_text().splitlines()]
    min_reclaimable=min(available_pages(x['host']['vm_stat']) for x in samples)
    max_pressure=max(int(x['host']['pressure'].strip().split(':')[-1]) for x in samples)
    # 256MiB explicitly covers additional CPU evaluation data and long-run
    # history beyond the short fixture; keep1GiB of observed free/inactive/spec.
    headroom=GIB;unmeasured_host_allowance=256*2**20
    assert min_reclaimable>=headroom+unmeasured_host_allowance and max_pressure<=2,(min_reclaimable,max_pressure)
    maxrss=max(x['peak_rss_bytes'] for x in qualification['shape_checks']);maxmps=max(x['peak_mps_bytes'] for x in qualification['shape_checks'])
    assert maxrss+unmeasured_host_allowance<4*GIB and maxmps<6*GIB
    current=host_memory();assert int(current['pressure'].strip().split(':')[-1])==1
    assert available_pages(current['vm_stat'])>=headroom+unmeasured_host_allowance
    memory=dict(physical_bytes=17179869184,max_observed_rss_bytes=maxrss,max_observed_mps_driver_bytes=maxmps,host_extra_allowance_bytes=unmeasured_host_allowance,minimum_observed_free_inactive_speculative_bytes=min_reclaimable,required_headroom_bytes=headroom,maximum_pressure_level=max_pressure,current=current,interpretation='Free+inactive+speculative is a reclaimable-page proxy, not guaranteed free RAM; combined-path pressure/peaks were measured. RSS and driver counts overlap and are not added. Current pressure must be normal. Continuous RSS/disk and per-update MPS guards remain enforced.')
    generation=dict(prior['generation_protocol'],scope='two_depth_context_transfer_confirmation',endpoint='final physical-time EMA of each of12runs',expected_completions=288)
    generation['annotation']=dict(generation['annotation'],blinding='Remove context,depth,bundle and checkpoint names; shuffle all288 rows with Python random.Random(62111423050902) before rating; save mapping separately.')
    plan=dict(kind='two_depth_context_quality_v1',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,torch_version=torch.__version__,python_version=platform.python_version(),scientific_training=True,slots=slots,generation_protocol=generation,ema_half_life_ns=127184813977,checkpoint_interval_ns=120000000000,probe_documents=probes,validation_targets=100416,bootstrap_seed=62111423050901,blind_order_seed=62111423050902,bootstrap_resamples=100000,primary_contrasts=['delta4','delta16','delta16_minus_delta4'],primary_interval_quantiles=[.01/6,1-.01/6],practical_ce_threshold=.02,source_regression_guard=.05,criteria_document='research/GENERAL_RSI_DEPTH_CONTEXT_QUALITY_DESIGN.md',fresh_role_audit='rg prefix62111423050 across research/research_state JSON/JSONL/Python returned0matches before assignment; original scientific seeds not reused.',max_attempts_per_slot=3,full_comparison_tranche_seconds=full_tranche,engineering_envelope_seconds=3600,completed_engineering_seconds=engineering_seconds,tranche_seconds=tranche,rss_limit_bytes=4*GIB,mps_limit_bytes=6*GIB,reserve_bytes=reserve,incremental_peak_bytes=incremental,restart_cap_bytes=restart_cap,forecast_seconds=components,forecast_with_margin_seconds=forecast,failure_variability_allowance_seconds=tranche-forecast,storage_components_bytes=storage,already_retained_engineering_bytes=retained,free_bytes_at_freeze=free,memory_admission=memory,test_opened=False)
    fixture_result=fixture();flops=estimate()
    OUT.mkdir();put(OUT/'plan.json',plan);put(OUT/'decision-fixture.json',fixture_result);put(OUT/'flops.json',flops)
    put(OUT/'admission.json',dict(accepted=True,plan_sha256=sha(OUT/'plan.json'),free_bytes=free,required_free_bytes=reserve+incremental,margin_bytes=free-reserve-incremental,forecast_seconds=forecast,science_tranche_seconds=tranche,total_with_engineering_envelope_seconds=forecast+3600,full_comparison_ceiling_seconds=full_tranche,engineering_result_sha256=sha(INTEGRATION/'attempt-01/result.json'),memory=memory,measured_max_restart_bytes=max_restart,retained_failed_harness_payloads_charged_through_actual_free_space=True,scientific_scores_used=0))
    print(json.dumps(dict(binding=sha(OUT/'plan.json'),forecast_hours=forecast/3600,required_free_bytes=reserve+incremental,margin_bytes=free-reserve-incremental,slots=len(slots))))
if __name__=='__main__':main()
