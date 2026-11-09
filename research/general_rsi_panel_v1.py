"""Freeze the specified panel and select LR only after all four screens finish."""
import argparse
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-panel-001'


def put(path,value):
    with path.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def prepare():
    from research.general_rsi_training_v1 import sha
    prior=json.loads((ROOT/'research/tokenizer-evidence/general-rsi-pilot-001/plan.json').read_text())
    qualifier=ROOT/'research/tokenizer-evidence/general-rsi-qualification-001'
    q=json.loads((qualifier/'result.json').read_text())
    assert q['complete_restart_qualified'] and q['document_scoring_qualified']
    assert json.loads((qualifier/'guard.json').read_text())['status']=='finished'
    by={r['arm']:r for r in q['arms']}
    pilot=json.loads((ROOT/'research/tokenizer-evidence/general-rsi-pilot-001/result.json').read_text())
    costs={r['arm']:r for r in pilot['arms']}
    sources=dict(prior['sources'])
    for name in ('general_rsi_training_v1.py','general_rsi_qualification_v1.py','general_rsi_worker_v1.py','general_rsi_panel_v1.py'):
        path='research/'+name;sources[path]=sha(ROOT/path)
    sources['research/general-rsi-prompts-v1.json']=sha(ROOT/'research/general-rsi-prompts-v1.json')
    for path,digest in sources.items():assert sha(ROOT/path)==digest,path
    def roles(bundle):
        return dict(zip(('native_initializer','composer_initializer','train_sampler','validation_sampler'),
                        range(62110300000+bundle*100+1,62110300000+bundle*100+5)))
    slots=[]
    for lr in (.00125,.0025):
        for arm in ('native','raw_bytes'):
            slots.append(dict(phase='screen',arm=arm,lr=lr,horizon=2048,roles=roles(1),generation=False))
    for bundle in (2,3,4):
        for arm in ('native','raw_bytes'):
            slots.append(dict(phase='confirmation',arm=arm,lr='selected_after_four_screens',horizon=10240,
                              roles=roles(bundle),generation=bundle==2))
    retained=0;forecasts=[]
    for i,s in enumerate(slots):
        arm=s['arm']
        endpoint=2*(costs[arm]['checkpoint']['size']+1024**2)
        restart=by[arm]['checkpoint']['size']+16*1024**2
        forecasts.append(dict(slot=i,retained_final_before=retained,final_endpoint_pair_bytes=endpoint,
                              two_restart_generations_bytes=2*restart,
                              high_water_bytes=retained+endpoint+2*restart+64*1024**2))
        retained+=endpoint
    peak=max(r['high_water_bytes'] for r in forecasts)
    training_seconds=sum(costs[s['arm']]['median_update_seconds']*s['horizon'] for s in slots)
    forecast_seconds=training_seconds*1.3+3600
    assert forecast_seconds<36*3600
    assert shutil.disk_usage(ROOT).free>=10*1024**3+peak
    plan=dict(kind='general_rsi_frozen_panel_v1',sources=sources,torch_version=prior['torch_version'],
        codebook_assets=prior['codebook_assets'],slots=slots,
        roles_audit='rg before assignment:62110300[1-4]0[1-4]|62110300999, exit1 and no prior JSON/JSONL matches.',
        data_manifest_sha256='b1849a5fed1fb648f507e407e5794946bba64328f519e6b0d6c360fa4e2e4fa8',
        qualification=dict(path=str(qualifier/'result.json'),sha256=sha(qualifier/'result.json')),
        reserve_bytes=10*1024**3,rss_limit_bytes=4*1024**3,mps_limit_bytes=6*1024**3,
        incremental_peak_bytes=peak,storage_forecast=forecasts,free_bytes_at_freeze=shutil.disk_usage(ROOT).free,
        forecast_seconds=forecast_seconds,tranche_seconds=36*3600,slot_timeout_seconds=5*3600,
        storage_assumptions='All10 RAW/H200 final inference pairs retained; two concurrent restart generations;16MiB future history/RNG allowance per restart;64MiB all other outputs. Numerical prerequisite payloads retired with exact audit, evidence retained. Entire remaining panel must fit before each slot.',
        lr_selection='Minimum final H200 full-validation CE within each arm across its two rates; ties lower LR; only after four successful screens.',
        bootstrap_seed=62110300999,bootstrap_resamples=10000,bootstrap_upper_quantile=.99,
        paired_delta_mean_threshold=-.02,all_three_negative=True,source_regression_limit=.05,
        expected_validation_targets=100416,probe_documents=[0,1,2,3,85,86,87,88],
        prompts=json.loads((ROOT/'research/general-rsi-prompts-v1.json').read_text())['prompts'],
        stop_policy='Human stop => signal checkpoint at update boundary; source/failure evidence immutable; execution recovery preserves scientific roles/gates.',
        test_opened=False)
    assert not OUT.exists();OUT.mkdir();put(OUT/'plan.json',plan)
    print(json.dumps(dict(plan_sha256=sha(OUT/'plan.json'),forecast_hours=forecast_seconds/3600,
                         incremental_peak_bytes=peak,free_bytes=plan['free_bytes_at_freeze'])),flush=True)


def select():
    from research.general_rsi_training_v1 import sha
    plan=json.loads((OUT/'plan.json').read_text());results=[];receipts=[]
    for slot in range(4):
        path=OUT/f'slot-{slot:02d}/result.json';r=json.loads(path.read_text())
        assert r['status']=='completed' and r['plan_sha256']==sha(OUT/'plan.json')
        assert r['slot']==plan['slots'][slot] and r['steps']==2048
        guards=list(path.parent.glob('guard-*.json'))
        assert guards and json.loads(sorted(guards)[-1].read_text())['status']=='finished'
        for endpoint in r['endpoints']:assert sha(endpoint['weights']['path'])==endpoint['weights']['sha256']
        results.append(r);receipts.append(dict(path=str(path),sha256=sha(path)))
    selected={}
    for arm in ('native','raw_bytes'):
        candidates=[r for r in results if r['slot']['arm']==arm]
        best=min(candidates,key=lambda r:(next(e['ce'] for e in r['endpoints'] if e['endpoint']=='h200'),r['slot']['lr']))
        selected[arm]=best['slot']['lr']
    put(OUT/'selection.json',dict(plan_sha256=sha(OUT/'plan.json'),screen_results=receipts,
                                selected_lrs=selected,architecture_adopted=False))
    print(json.dumps(selected),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','select']);args=parser.parse_args()
    prepare() if args.action=='prepare' else select()
