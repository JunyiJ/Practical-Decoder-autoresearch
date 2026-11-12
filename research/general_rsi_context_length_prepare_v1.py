"""Freeze fresh context quality roles, criterion and complete measured admission."""
import datetime,fcntl,json,math,shutil,subprocess
from pathlib import Path
from research.general_rsi_pseudo_update_diagnostic_v1 import ROOT,sha,put,check_stop
OUT=ROOT/'research/tokenizer-evidence/general-rsi-context-length-quality-001'
def read(p):return json.loads(Path(p).read_text())
def main():
    with open(ROOT/'research_state/tokenizer-worker-lock/.lock','a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);check_stop();assert not OUT.exists()
        check=subprocess.run(['rg','-l','621112110100','research','research_state','research_runs','--glob','*.json','--glob','*.jsonl'],cwd=ROOT,capture_output=True,text=True)
        assert check.returncode==1 and not check.stdout,check.stdout
        integration=ROOT/'research/tokenizer-evidence/general-rsi-context-length-integration-001'
        review=read(integration/'completion-review.json');r=read(integration/'attempt-01/result.json');assert review['accepted']
        assert review['worker_source_sha256']==sha(ROOT/'research/general_rsi_context_length_worker_v1.py') and review['aggregation_source_sha256']==sha(ROOT/'research/general_rsi_context_length_score_v1.py')
        prior=read(ROOT/'research/tokenizer-evidence/general-rsi-native-horizon-001/plan.json')
        sources=dict(read(integration/'plan.json')['sources'])
        for name in ['research/general_rsi_context_length_worker_v1.py','research/general_rsi_context_length_score_v1.py','research/general_rsi_context_length_integration_review_v1.py','research/general_rsi_context_length_prepare_v1.py','research/GENERAL_RSI_CONTEXT_LENGTH_QUALITY_DESIGN.md']:
            sources[name]=sha(ROOT/name)
        assert all(sha(ROOT/f)==h for f,h in sources.items())
        slots=[]
        for bundle,order in enumerate([(128,256),(256,128),(128,256)],1):
            roles=dict(zip(['native_initializer','composer_initializer','train_sampler','validation_sampler'],range(62111211010000+bundle*10+1,62111211010000+bundle*10+5)))
            for context in order:slots.append(dict(slot=len(slots),phase='confirmation',arm=f'context{context}',context=context,bundle=bundle,lr=.0025,budget_ns=14400*10**9,roles=roles,timeout_seconds=22800))
        pilot=read(ROOT/'research/tokenizer-evidence/general-rsi-context-length-pilot-001/attempt-01/result.json')
        updates=86400/min(x['mean_seconds'] for x in pilot['summaries'])
        components=dict(training_seconds=86400,endpoint_probe_upper_seconds=30*r['training_document_evaluation_and_oracle_seconds']/r['train_document_targets']*100416,generation_seconds=6*r['generation_seconds']*1.5,checkpoint_seconds=732*max(3.,2*r['restart_io_seconds']),initialization_seconds=180,outside_clock_logging_seconds=updates*.01)
        output=12*128*2**20;restart=2*512*2**20;metadata=512*2**20;failed=5*2**30;incremental=output+restart+metadata+failed;reserve=10*2**30;extra=2*2**30;required=incremental+reserve+extra
        assert required==21474836480
        forecast=dict(measured_weight_bytes=max(x['size'] for x in r['endpoint_receipts'].values()),measured_restart_bytes=max(x['size'] for x in r['restart_receipts']),final_weight_count=12,final_output_cap_bytes=output,active_restart_and_serialization_overlap_bytes=restart,metadata_bytes=metadata,failed_output_allowance_bytes=failed,incremental_peak_bytes=incremental,reserve_bytes=reserve,extra_headroom_bytes=extra,required_free_bytes=required,actual_free_bytes=shutil.disk_usage(ROOT).free,predicted_updates=updates,cost_components_seconds=components,estimated_seconds=sum(components.values()),tranche_seconds=36*3600,failure_and_variability_allowance_seconds=36*3600-sum(components.values()),assumptions='All existing files already charged in actual free space. Per-arm timing from24-update pilot; score upper bound charges30full108-document passes at per-target cost of two-context training-document scoring plus FP64 oracle. Six generation sets at1.5x measured uncached cost. Checkpoint allowance covers732writes.512MiB metadata covers paired token scores, step/interval logs, all first outputs and audits.5GiB failed-output allowance; no partial panel.')
        assert forecast['actual_free_bytes']>=required and forecast['failure_and_variability_allowance_seconds']>0
        qualification={str(p.relative_to(ROOT)):sha(p) for p in [integration/'completion-review.json',ROOT/'research/tokenizer-evidence/general-rsi-context-length-fixture-002/completion-review.json',ROOT/'research/tokenizer-evidence/general-rsi-context-length-pilot-001/completion-review.json',ROOT/'research/tokenizer-evidence/general-rsi-capacity-009/completed.json']}
        plan=dict(kind='general_rsi_context_length_quality_v1',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),torch_version='2.14.1',sources=sources,qualification_evidence=qualification,slots=slots,seed_role_audit='No prefix621112110100 in prior research/research_state/research_runs JSON/JSONL; fresh paired bundles and separate bootstrap role.',ema_half_life_ns=prior['ema_half_life_ns'],ema_rule=prior['ema_rule'],checkpoint_interval_ns=120*10**9,probe_documents=prior['probe_documents'],generation_protocol=prior['generation_protocol'],acceptance=dict(endpoint='time_ema',candidate='context256',baseline='context128',all_three_deltas_negative=True,mean_delta_at_most=-.02,source_regression_at_most=.05,bootstrap=dict(seed=62111211010099,replicates=10000,method='PCG64 resample108documentIDs; average paired seed loss sums before resampling; sampled target-count denominator; linear99th percentile upper<0'),otherwise='Retain context128; no best seed or secondary/generation rescue; no pooling old scores.'),evaluation=dict(targets=100416,documents=108,primary='Native-length within-document nonoverlapping chunks, every target after first token exactly once includingEOT.',secondary='All256 RAW/EMA endpoints also scored at128. Compare same target IDs/documents. No test.'),max_attempts_per_slot=2,tranche_seconds=36*3600,rss_limit_bytes=4*2**30,mps_limit_bytes=6*2**30,reserve_bytes=reserve,incremental_peak_bytes=incremental,forecast=forecast,test_opened=False)
        OUT.mkdir();put(OUT/'plan.json',plan)
        put(OUT/'admission.json',dict(plan_sha256=sha(OUT/'plan.json'),admitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),shared_lock_held=True,scientific_scores_seen=False,changed_context_paths_qualified=True,prior_closed_prerequisites_reused=True,forecast=forecast))
        print(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),forecast=forecast),indent=2))
if __name__=='__main__':main()
