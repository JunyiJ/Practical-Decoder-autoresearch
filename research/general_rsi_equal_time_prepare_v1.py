"""Freeze the qualified equal-time panel before first scientific work."""
import fcntl,json,math,shutil,datetime
from pathlib import Path
from research.general_rsi_training_v2 import sha
from research.general_rsi_equal_time_worker_v2 import put
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-equal-time-001'


def main():
    assert not OUT.exists()
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not json.loads((ROOT/'research_state/general-rsi-state.json').read_text()).get('stop_requested',False)
        previous=json.loads((ROOT/'research/tokenizer-evidence/general-rsi-worker-qualification-001/plan.json').read_text())
        sources=dict(previous['sources'])
        for path,digest in sources.items():assert sha(ROOT/path)==digest,path
        evidence={}
        for directory in ('general-rsi-time-qualification-001','general-rsi-io-qualification-001','general-rsi-worker-qualification-001'):
            folder=ROOT/'research/tokenizer-evidence'/directory
            result=json.loads((folder/'result.json').read_text());guard=json.loads((folder/'guard.json').read_text())
            assert result['qualified'] and guard['status']=='finished' and guard['reason'] is None
            for name in ('plan.json','result.json','guard.json'):
                evidence[str((folder/name).relative_to(ROOT))]=sha(folder/name)
        check=ROOT/'research/tokenizer-evidence/general-rsi-worker-qualification-001/cumulative-guard-check.json'
        assert json.loads(check.read_text())['worker_sha256']==sha(ROOT/'research/general_rsi_equal_time_worker_v2.py')
        evidence[str(check.relative_to(ROOT))]=sha(check)
        for name in ('research/general_rsi_equal_time_worker_v2.py','research/general_rsi_equal_time_prepare_v1.py',
                     'research/general_rsi_equal_time_review_v1.py','research/GENERAL_RSI_EQUAL_TIME_DESIGN.md'):
            sources[name]=sha(ROOT/name)
        slots=[]
        role_names=('native_initializer','composer_initializer','train_sampler','validation_sampler')
        screen=dict(zip(role_names,range(62110700001,62110700005)))
        for lr in (.00125,.0025):
            for arm in ('native','raw_bytes'):
                slots.append(dict(slot=len(slots),phase='screen',arm=arm,lr=lr,budget_ns=1800*10**9,roles=screen,timeout_seconds=3600))
        for bundle in (1,2,3):
            roles=dict(zip(role_names,range(62110700001+100*bundle,62110700005+100*bundle)))
            for arm in ('native','raw_bytes'):
                slots.append(dict(slot=len(slots),phase='confirmation',arm=arm,lr='selected_after_four_screens',budget_ns=7200*10**9,roles=roles,bundle=bundle,timeout_seconds=10800))
        timing=ROOT/'research/tokenizer-evidence/general-rsi-next-001/timing.json'
        means=json.loads(timing.read_text())['means'];half_life_ns=round(200*means['raw_bytes']['training_update_seconds']/10240*10**9)
        evidence[str(timing.relative_to(ROOT))]=sha(timing)
        projected_outputs=1794396880
        metadata=256*1024**2;failed_writes=10*512*1024**2
        incremental=projected_outputs+2*512*1024**2+metadata+failed_writes
        free=shutil.disk_usage(ROOT).free;reserve=10*1024**3
        assert free>=reserve+incremental
        scores={};generations={}
        for arm in ('native','raw_bytes'):
            scores[arm]=json.loads((ROOT/f'research/tokenizer-evidence/general-rsi-qualification-001/{arm}-result.json').read_text())['full_validation_seconds']
            generations[arm]=json.loads((ROOT/f'research/tokenizer-evidence/general-rsi-io-qualification-001/{arm}-result.json').read_text())['generation_seconds']
        updates=sum(s['budget_ns']/1e9/(means[s['arm']]['training_update_seconds']/10240) for s in slots)
        costs=dict(training_seconds=50400,
            endpoint_and_conservative_probe_scoring_seconds=20*sum(scores.values()),
            generation_seconds=3*sum(generations.values()),
            checkpoint_io_seconds=440*2.0,initialization_seconds=10*30,
            end_records_and_step_logging_seconds=updates*.01)
        estimated=sum(costs.values());assert estimated<72000
        forecast=dict(final_pairs=10,projected_outputs_bytes=projected_outputs,active_restart_overlap_bytes=2*512*1024**2,
            metadata_and_header_margin_bytes=metadata,failed_write_allowance_bytes=failed_writes,incremental_peak_bytes=incremental,
            reserve_bytes=reserve,required_free_bytes=reserve+incremental,actual_free_bytes=free,
            measured_full_validation_seconds=scores,measured_generation_seconds=generations,
            predicted_total_updates=updates,cost_components_seconds=costs,estimated_total_seconds=estimated,
            tranche_seconds=72000,remaining_failure_and_variability_allowance_seconds=72000-estimated,
            assumptions='Checkpoint allowance2s each versus measured native assembled average0.802s; probes pessimistically cost full validation. Initialization30s/run and outside-clock logging0.01s/update are conservative engineering estimates, not measured guarantees.')
        plan=dict(kind='general_rsi_equal_time_scientific_panel_v1',sources=sources,qualification_evidence=evidence,
            torch_version=previous['torch_version'],codebook_assets=previous['codebook_assets'],slots=slots,
            seed_role_audit='Before assignment rg found no62110700 JSON/JSONL matches. Bootstrap62110700999; annotation shuffle62110500998 retained.',
            ema_half_life_ns=half_life_ns,ema_rule='Start at first update starting >=2/3 budget, initialize from that update weights, then every update decay by elapsed time between update starts; includes averaging cost.',
            checkpoint_interval_ns=120*10**9,probe_documents=[0,1,2,3,85,86,87,88],
            generation_protocol=json.loads((ROOT/'research/general-rsi-generation-eval-v1.json').read_text()),
            acceptance=dict(endpoint='time_ema',all_three_deltas_negative=True,mean_delta_at_most=-.02,source_regression_at_most=.05,
                bootstrap=dict(seed=62110700999,replicates=10000,method='PCG64 resample108documentIDs; average paired seed loss sums before resampling; sampled target-count denominator; linear99th percentile upper<0'),symmetric=True),
            max_attempts_per_slot=2,tranche_seconds=72000,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,
            reserve_bytes=reserve,incremental_peak_bytes=incremental,forecast=forecast,test_opened=False)
        OUT.mkdir();put(OUT/'plan.json',plan)
        put(OUT/'admission.json',dict(plan_sha256=sha(OUT/'plan.json'),admitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            all_new_prerequisites_passed=True,shared_lock_held=True,forecast=forecast,scientific_scores_seen=False))
        print(sha(OUT/'plan.json'),flush=True)
if __name__=='__main__':main()
