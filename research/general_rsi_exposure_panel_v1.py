"""Freeze the evidence-grounded short/long exposure comparison before training."""
import fcntl
import json
from pathlib import Path
import shutil
from research.general_rsi_training_v1 import sha

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-exposure-001'
PRIOR=ROOT/'research/tokenizer-evidence/general-rsi-panel-001'

def prepare():
    prior=json.loads((PRIOR/'plan.json').read_text())
    decision=json.loads((PRIOR/'decision.json').read_text())
    assert decision['selected_architecture']=='raw_bytes' and all(decision['byte_gates'].values())
    assert decision['plan_sha256']==sha(PRIOR/'plan.json')
    sources=dict(prior['sources'])
    for name in ('general_rsi_training_v2.py','general_rsi_exposure_worker_v1.py','general_rsi_exposure_panel_v1.py','general_rsi_exposure_preflight_v1.py','GENERAL_RSI_EXPOSURE_DESIGN.md'):
        path='research/'+name;sources[path]=sha(ROOT/path)
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    slots=[]
    for bundle in (1,2,3):
        roles=dict(zip(('native_initializer','composer_initializer','train_sampler','validation_sampler'),range(62110400000+bundle*100+1,62110400000+bundle*100+5)))
        for horizon in (10240,20480):
            slots.append(dict(phase='exposure',arm='raw_bytes',lr=.0025,horizon=horizon,roles=roles,generation=bundle==1))
    retained=0; forecasts=[]
    for i,slot in enumerate(slots):
        endpoint=132975578;restart=529357538
        forecasts.append(dict(slot=i,retained_final_before=retained,final_endpoint_pair_bytes=endpoint,two_restart_generations_bytes=restart,high_water_bytes=retained+endpoint+restart+64*1024**2))
        retained+=endpoint
    peak=max(f['high_water_bytes'] for f in forecasts)
    assert peak==1394319870
    forecast=max(json.loads((PRIOR/f'slot-{i:02d}/guard-01.json').read_text())['elapsed_seconds'] for i in (5,7,9))*9*1.3+3600
    assert forecast<30*3600
    plan=dict(kind='general_rsi_exposure_panel_v1',sources=sources,torch_version=prior['torch_version'],codebook_assets=prior['codebook_assets'],slots=slots,
        preceding_decision=dict(path=str(PRIOR/'decision.json'),sha256=sha(PRIOR/'decision.json')),
        roles_audit='Before assignment rg62110400[1-3]0[1-4]|62110400999 over research/research_state JSON/JSONL returned exit1/no matches.',
        data_manifest_sha256=prior['data_manifest_sha256'],qualification=prior['qualification'],
        implementation_delta='training_v2 differs from qualified v1 only by accepting horizon20480; worker uses new paths/fixed LR/exposure phase, keeps full2048 scoring, and requires previous accepted reviews. Numerical update, restore, scoring and generation unchanged.',
        reserve_bytes=10*1024**3,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,incremental_peak_bytes=peak,storage_forecast=forecasts,
        forecast_seconds=forecast,tranche_seconds=30*3600,slot_timeout_seconds=5*3600,
        storage_assumptions='Retain all6 RAW/H200 pairs; two restart generations each with16MiB history/RNG allowance;64MiB metadata;10GiB reserve includes2GiB headroom. Existing fresh weights preserved.',
        bootstrap_seed=62110400999,bootstrap_resamples=10000,bootstrap_upper_quantile=.99,bootstrap_method='NumPy PCG64 default_rng; paired seed document loss sums averaged then108IDs resampled; ratio of sums; linear quantile',
        paired_delta_mean_threshold=-.05,all_three_negative=True,source_regression_limit=.05,decision='Adopt H20480 only if all three long-minus-short H200 deltas<0, mean<=-.05, bootstrap upper<0 and neither source mean regression>.05; otherwise retain H10240.',
        expected_validation_targets=100416,probe_documents=prior['probe_documents'],prompts=prior['prompts'],
        stop_policy=prior['stop_policy'],test_opened=False)
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan['free_bytes_at_freeze']=shutil.disk_usage(ROOT).free
        assert plan['free_bytes_at_freeze']>=plan['reserve_bytes']+peak
        assert not OUT.exists();OUT.mkdir()
        with (OUT/'plan.json').open('x') as f: json.dump(plan,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),forecast_hours=forecast/3600,required_bytes=plan['reserve_bytes']+peak,free_bytes=plan['free_bytes_at_freeze'])))

if __name__=='__main__':prepare()
