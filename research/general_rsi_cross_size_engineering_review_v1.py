"""Review new-depth execution and derive prospective whole-panel resource costs."""
import datetime,fcntl,hashlib,json,shutil
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-pilot-001'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(OUT/'plan.json');r=read(OUT/'attempt-01/result.json');assert r['accepted'] and r['plan_sha256']==sha(OUT/'plan.json')
        assert all(sha(ROOT/n)==h for n,h in plan['sources'].items())
        cpu=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-fixture-001';cr=read(cpu/'attempt-01/result.json');cp=read(cpu/'plan.json')
        assert cr['accepted'] and cr['plan_sha256']==sha(cpu/'plan.json') and all(sha(ROOT/n)==h for n,h in cp['sources'].items())
        bindings={str(p.relative_to(ROOT)):sha(p) for p in [OUT/'plan.json',OUT/'attempt-01/result.json',cpu/'attempt-01/result.json',cpu/'attempt-01/guard.json']}
        timing=[];peak_rss=0;retained=0;updates=0;expected_retired={}
        for s in r['summaries']:
            depth=s['slot']['depth'];arm=s['slot']['arm'];d=OUT/'attempt-01'/f'd{depth}-{arm}'
            assert read(d/'review.json')==s and s['accepted']
            g=read(d/'guard.json');assert g['status']=='finished' and g['elapsed_seconds']<plan['max_seconds_per_slot']
            assert g['observation']['peak_rss_bytes']<plan['rss_limit_bytes'];peak_rss=max(peak_rss,g['observation']['peak_rss_bytes'])
            for f in [d/'review.json',d/'guard.json',d/'updates.jsonl']:
                bindings[str(f.relative_to(ROOT))]=sha(f)
            rows=[json.loads(x) for x in (d/'updates.jsonl').read_text().splitlines()]
            assert len(rows)==s['steps'] and all(np.isfinite(x['loss']) and x['mps_bytes']<plan['mps_limit_bytes'] for x in rows)
            updates+=len(rows)
            if depth>4:
                replay=[json.loads(x) for x in (d/'replay.jsonl').read_text().splitlines()];assert len(replay)==10
                for a,b in zip(rows[30:],replay):assert {k:v for k,v in a.items() if k not in ('seconds','mps_bytes')}=={k:v for k,v in b.items() if k not in ('seconds','mps_bytes')}
                assert s['full_restart_replay_exact'] and s['active_ema_restart'] and s['wrong_depth_rejected_before_mutation'];updates+=10
                expected_retired[s['transient_prefix']['path']]=s['transient_prefix']
                bindings[str((d/'replay.jsonl').relative_to(ROOT))]=sha(d/'replay.jsonl')
            w=s['endpoint'];path=Path(w['path']);assert path.stat().st_size==w['size'] and sha(path)==w['sha256'];retained+=w['size']
            assert s['oracle']['score_count']==258 and s['oracle']['score_targets_exact']
            assert abs(s['oracle']['score_cpu_ce']-s['oracle']['score_mps_ce'])<=plan['scoring_ce_atol']
            p95=float(np.quantile(s['synchronized_update_seconds'][5:20],.95))
            timing.append(dict(depth=depth,arm=arm,median_seconds=s['steady_median_seconds'],p95_seconds=p95,three_run_training_forecast_seconds=p95*10240*3))
        assert updates==240 and retained==r['retained_bytes']<=plan['retained_inference_cap_bytes']
        j=OUT/'attempt-01/deletion-journal.jsonl';entries=[json.loads(x) for x in j.read_text().splitlines()];assert len(entries)==8
        for i in range(0,len(entries),2):
            a,b=entries[i:i+2];assert a['event']=='intent' and b['event']=='deleted'
            assert {k:v for k,v in a.items() if k!='event'}=={k:v for k,v in b.items() if k!='event'}
            expected=expected_retired[a['path']];assert a['size']==expected['size'] and a['sha256']==expected['sha256'] and not Path(a['path']).exists()
        bindings[str(j.relative_to(ROOT))]=sha(j)
        ev=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-eval-002';er=read(ev/'dataset.json');ep=read(ev/'protocol.json')
        assert er['protocol_sha256']==sha(ev/'protocol.json') and er['targets']==32768 and er['coverage_exact'] and er['model_scores_used']==0
        assert all(sha(ROOT/n)==h for n,h in ep['sources'].items()) and all(sha(ROOT/n)==h for n,h in er['payload_sha256'].items())
        for p in [ev/'dataset.json',ev/'protocol.json']:bindings[str(p.relative_to(ROOT))]=sha(p)
        training=sum(x['three_run_training_forecast_seconds'] for x in timing)
        scoring=18*2*(100416+32768)/258*max(s['oracle']['score_seconds'] for s in r['summaries'])
        base=dict(training=training,scoring=scoring,generation=18*24*4,checkpoint_and_replay=3600)
        forecast=sum(base.values())*1.3
        allocation=read(ROOT/'research/tokenizer-evidence/general-rsi-cross-size-boundary-001/allocation.json')
        remaining=allocation['storage_total_bytes']-retained
        result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_engineering=True,source_sha256=sha(__file__),sources_verified=len(plan['sources']),cpu_depth4_parity_passed=True,new_depths=[8,16],all_four_new_depth_arm_restart_replays_exact=True,all_six_training_only_scorer_checks_pass=True,optimizer_updates=updates,peak_rss_bytes=peak_rss,peak_mps_bytes=max(s['peak_mps_bytes'] for s in r['summaries']),retained_endpoint_bytes=retained,retained_endpoints_hash_verified=6,exact_transient_prefixes_retired=4,timing=timing,forecast_seconds_before_margin=base,forecast_with_30_percent_margin_seconds=forecast,prospective_tranche_seconds=54*3600,tranche_revision_reason='Measured full18-run forecast exceeds initial36h.54h covers49.615h forecast before scientific outcomes; no run/seed shortened or dropped.',storage_remaining_planning_bytes=remaining,free_bytes=shutil.disk_usage(ROOT).free,storage_remaining_planning_fits=shutil.disk_usage(ROOT).free>=remaining,shared_lock_available=True,worker_exit_verified='exec session23415 exit0; process inspection found no research worker',evidence_sha256=bindings,scientific_training_admitted=False,next_required='Freeze fresh scientific roles, interaction/quality criteria, FLOP accounting and broader-evaluation bindings. Integrate/qualify only new endpoint loader and complete serial worker paths; refresh complete resource admission.',test_opened=False)
        assert forecast<result['prospective_tranche_seconds'] and result['storage_remaining_planning_fits']
        with (OUT/'completion-review.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
        print(json.dumps({k:v for k,v in result.items() if k not in ('timing','evidence_sha256')},indent=2))
if __name__=='__main__':main()
