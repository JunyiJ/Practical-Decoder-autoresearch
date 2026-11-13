"""Prospective scientific freeze only after integration and complete admission."""
import datetime,json,math,platform,shutil
from pathlib import Path
import torch
from research.general_rsi_training_v2 import sha
from research.general_rsi_context_length_worker_v1 import put
from research.general_rsi_cross_size_flops_v1 import estimate
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-quality-001'
def read(p):return json.loads(Path(p).read_text())
def main():
    integration=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-integration-002'
    result=read(integration/'attempt-01/result.json');assert result['accepted_integration']
    guard=read(integration/'attempt-01/guard.json');assert guard['status']=='finished'
    prior=read(integration/'plan.json');assert result['binding']==sha(integration/'plan.json')
    assert all(sha(ROOT/n)==h for n,h in prior['sources'].items())
    sources=dict(prior['sources'])
    sources['research/general_rsi_cross_size_panel_v1.py']=sha(__file__)
    for p in [ROOT/'research/GENERAL_RSI_CROSS_SIZE_QUALITY_DESIGN.md',ROOT/'research/general_rsi_cross_size_flops_v1.py',ROOT/'research/general_rsi_cross_size_review_v1.py',ROOT/'research/general_rsi_cross_size_analysis_v1.py',integration/'attempt-01/result.json',integration/'attempt-01/guard.json',ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001/completion-review.json',ROOT/'research/tokenizer-evidence/general-rsi-cross-size-eval-002/dataset.json',ROOT/'research/tokenizer-evidence/general-rsi-cross-size-eval-002/protocol.json']:
        sources[str(p.relative_to(ROOT))]=sha(p)
    roles=[]
    for b in range(1,4):
        base=62111312034000+b*100
        roles.append(dict(roles=dict(zip(['native_initializer','composer_initializer','train_sampler','validation_sampler'],range(base+1,base+5))),extra_group_seeds=list(range(base+5,base+8))))
    order=[(1,4,['native','raw_bytes']),(1,8,['raw_bytes','native']),(1,16,['native','raw_bytes']),
           (2,8,['native','raw_bytes']),(2,16,['raw_bytes','native']),(2,4,['raw_bytes','native']),
           (3,16,['native','raw_bytes']),(3,4,['native','raw_bytes']),(3,8,['raw_bytes','native'])]
    slots=[]
    engineering=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001/completion-review.json')
    timing={(r['depth'],r['arm']):r['p95_seconds'] for r in engineering['timing']}
    # Terminal scoring plus all fixed probes. Worst measured engineering scorer
    # includes full loader-corrected logits and no quality-dependent selection.
    pilot=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001/attempt-01/result.json')
    scorer=max(s['oracle']['score_seconds'] for s in pilot['summaries'])/258
    index=[json.loads(x) for x in (ROOT/'data/raw/rsi-general-10m-v1/val.index.jsonl').read_text().splitlines()]
    probe_targets=sum(index[i]['tokens']-1 for i in [0,1])
    generation=read(integration/'attempt-01/slot-00/generation/result.json')['seconds']
    components=dict(training=sum(timing.values())*10240*3,
        terminal_scoring=18*2*(100416+32768)*scorer,
        probes=18*9*probe_targets*scorer,
        generation=18*max(96,generation),checkpoint_and_failure_replay=3600)
    engineering_guards=list((ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001').glob('attempt-*/*/guard.json'))+list((ROOT/'research/tokenizer-evidence/general-rsi-cross-size-integration-001').glob('attempt-*/guard.json'))+list(integration.glob('attempt-*/guard.json'))
    engineering_seconds=sum(read(p)['elapsed_seconds'] for p in engineering_guards)
    components['completed_engineering']=engineering_seconds
    forecast=sum(components.values())*1.3
    assert forecast<54*3600,(forecast,components)
    for bundle,depth,arms in order:
        for arm in arms:
            # Per-slot ceilings include cumulative attempts; all slots also share
            # the hard54h tranche, so retry ceilings cannot reset total cost.
            timeout=math.ceil(1.3*(timing[depth,arm]*10240+2*(100416+32768)*scorer+9*probe_targets*scorer+max(96,generation)+200))
            slots.append(dict(slot=len(slots),bundle=bundle,arm=arm,depth=depth,horizon=10240,lr=.0025,timeout_seconds=timeout,**roles[bundle-1]))
    storage=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-boundary-001/allocation.json')['storage_planning_bytes']
    retained=engineering['retained_endpoint_bytes']+result['retained_bytes']
    assert retained<storage['engineering_artifacts']
    reserve=storage['reserve']+storage['extra_headroom']
    incremental=sum(storage.values())-reserve-retained
    free=shutil.disk_usage(ROOT).free;assert free>=reserve+incremental,(free,reserve+incremental)
    broader=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-eval-002/dataset.json')
    for p,h in broader['payload_sha256'].items():assert sha(ROOT/p)==h
    generation_protocol=dict(prior['generation_protocol'],scope='cross_size_equal_token_confirmation',endpoint='finalH200 for each of18confirmations',expected_completions=432)
    generation_protocol['annotation']=dict(generation_protocol['annotation'],blinding='Remove arm,depth,bundle and checkpoint names; shuffle all432 rows with Python random.Random(62111312034902) before rating; save mapping separately.')
    plan=dict(kind='cross_size_equal_token_quality_v1',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources=sources,
        torch_version=torch.__version__,python_version=platform.python_version(),scientific_training=True,slots=slots,
        codebook_assets=prior['codebook_assets'],broader_payloads=broader['payload_sha256'],
        generation_protocol=generation_protocol,checkpoint_interval=1024,probe_steps=list(range(1024,10240,1024)),probe_documents=[0,1],
        validation_targets=100416,broader_targets=32768,bootstrap_seed=62111312034901,blind_order_seed=62111312034902,bootstrap_resamples=100000,
        primary_contrasts=['delta4','delta8','delta16','delta16_minus_delta4'],primary_interval_quantiles=[.00125,.99875],practical_ce_threshold=.02,source_regression_guard=.05,broader_bootstrap_seeds={'pg19_books':62111312034903,'python_library_prose':62111312034904},broader_interval_quantiles=[.025,.975],
        criteria_document='research/GENERAL_RSI_CROSS_SIZE_QUALITY_DESIGN.md',
        fresh_role_audit='rg62111312034[0-9]+ over59541 JSON/JSONL files returned0matches before assignment',
        max_attempts_per_slot=3,full_comparison_tranche_seconds=54*3600,completed_engineering_seconds=engineering_seconds,tranche_seconds=54*3600-engineering_seconds,rss_limit_bytes=4*2**30,mps_limit_bytes=6*2**30,reserve_bytes=reserve,incremental_peak_bytes=incremental,
        forecast_seconds=components,forecast_with_margin_seconds=forecast,storage_components_bytes=storage,already_retained_engineering_bytes=retained,
        free_bytes_at_freeze=free,test_opened=False)
    from research.general_rsi_cross_size_analysis_v1 import fixture
    fixture_result=fixture()
    OUT.mkdir();put(OUT/'plan.json',plan);put(OUT/'flops.json',estimate());put(OUT/'decision-fixture.json',fixture_result)
    put(OUT/'admission.json',dict(plan_sha256=sha(OUT/'plan.json'),accepted=True,free_bytes=free,required_free_bytes=reserve+incremental,
        margin_bytes=free-reserve-incremental,forecast_seconds=forecast,tranche_seconds=plan['tranche_seconds'],integration_result_sha256=sha(integration/'attempt-01/result.json'),scientific_scores_used=0))
    print(json.dumps(dict(binding=sha(OUT/'plan.json'),forecast_hours=forecast/3600,storage_margin_bytes=free-reserve-incremental,slots=len(slots))))
if __name__=='__main__':main()
