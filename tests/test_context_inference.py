import copy
import json
from types import SimpleNamespace

import pytest
import torch

from src.models.gpt import GPT
from src.research import context_inference as study, inference_bench as bench
from src.research.protocol import file_digest
from tests.test_gqa_study import gqa_config
from tests.test_inference_bench import job, result


def test_context512_actual_cache_and_full_cached_logits():
    torch.set_num_threads(1);torch.manual_seed(17)
    cfg=SimpleNamespace(model=gqa_config(64));cfg.model.block_size=512
    model=GPT(cfg.model).eval();caches=bench.make_caches(model,cfg,1)
    tokens=(torch.arange(512)%cfg.model.vocab_size).reshape(1,512)
    with torch.no_grad():full=model(tokens)[0]
    _,cached=bench.cached_pass(model,tokens,caches,256,collect=True)
    torch.testing.assert_close(cached,full,atol=2e-6,rtol=1e-5)
    assert bench.cache_bytes(caches)==2*512*64*4


def test_512_result_capacity_contract():
    j=job();j.update(capacity=512,prompt=128,decode=128)
    j['models']['candidate'].update(family='d2_gqa_c512',attention='gqa')
    r=result(j)
    for v in r['correctness'].values():
        v['cache_allocated_bytes']*=4;v['cache_logical_used_bytes']*=4
    study.validate_result(j,r)
    r['correctness']['candidate']['cache_logical_used_bytes']*=2
    with pytest.raises(ValueError):study.validate_result(j,r)


def records():
    rows=[]
    for l,c in study.PANEL:
        for seed in study.SEEDS:
            for b in [1,8]:
                for p in [c//4,c//2]:
                    rows.append(dict(id=f'{l}-{c}-{seed}-{b}-{p}',models={'candidate':{'family':f'd{l}_gqa_c{c}'}},
                        summary={phase:dict(paired_ratio_median=.99,paired_ratio_iqr=[.98,1.01]) for phase in ['prefill','decode']},
                        correctness={'reference':{'cache_allocated_bytes':100},'candidate':{'cache_allocated_bytes':50}}))
    return rows


def test_cache_savings_do_not_hide_one_seed_workload_latency_regression():
    rows=records();assert all(r['operational_efficiency_eligible'] for r in study.summarize_cells(rows))
    rows[-1]['summary']['decode']['paired_ratio_median']=1.051
    summary=study.summarize_cells(rows)
    assert not summary[-1]['operational_efficiency_eligible']
    assert summary[-1]['cache_benefit_all_workloads']
    assert all(r['operational_efficiency_eligible'] for r in summary[:-1])
    rows=records();rows[0]['summary']['prefill']['paired_ratio_iqr'][1]=1.101
    assert not study.summarize_cells(rows)[0]['operational_efficiency_eligible']


def test_speed_route_requires_repetition_stability():
    rows=records()
    for r in rows:
        r['correctness']['candidate']['cache_allocated_bytes']=80
        r['summary']['decode']=dict(paired_ratio_median=.89,paired_ratio_iqr=[.88,.99])
    assert all(r['operational_efficiency_eligible'] for r in study.summarize_cells(rows))
    rows[0]['summary']['decode']['paired_ratio_iqr'][1]=1.001
    assert not study.summarize_cells(rows)[0]['operational_efficiency_eligible']
    with pytest.raises(ValueError):study.summarize_cells(rows[:-1])


def fixture_panel(tmp_path):
    directory=tmp_path/'research_runs'/study.STUDY;directory.mkdir(parents=True)
    rows=[];comparisons=[];choices={}
    for l,c in study.PANEL:
        candidate=f'd{l}_gqa_c{c}';reference=f'd{l}_mha_c{c}'
        comparisons.append(dict(contrast='gqa_minus_mha',candidate=candidate,reference=reference,width=256,quality_eligible=True))
        for f in [candidate,reference]:
            choices[f]={'lr':.005}
            for seed in study.SEEDS:
                tid=f'{f}-{seed}';d=directory/'trials'/tid;d.mkdir(parents=True)
                (d/'request.json').write_text(json.dumps(dict(config={'model':{'vocab_size':1}})))
                rows.append(dict(family=f,width=256,seed=seed,stage='confirmation',lr=.005,status='completed',
                                 checkpoint=tid,checkpoint_sha256='a'*64,trial_id=tid,request_sha256=file_digest(d/'request.json')))
    report=dict(records=rows,comparisons=comparisons,selection={'choices':choices})
    (directory/'report.json').write_text(json.dumps(report));(directory/'plan.json').write_text(json.dumps({'data_sha256':'b'*64}))
    return directory,report


def test_panel_all_fresh_seeds_and_matched_workloads(tmp_path):
    directory,report=fixture_panel(tmp_path);jobs=study.panel_jobs(tmp_path)
    assert len(jobs)==len({j['id'] for j in jobs})==36
    assert len({s['checkpoint'] for j in jobs for s in j['models'].values()})==18
    assert {j['seed'] for j in jobs}=={2031,2032,2033}
    assert all(j['prompt']+j['decode']<=j['capacity'] for j in jobs)
    assert {(j['capacity'],j['prompt']) for j in jobs}=={(128,32),(128,64),(512,128),(512,256)}
    report['comparisons'][0]['quality_eligible']=False
    (directory/'report.json').write_text(json.dumps(report))
    with pytest.raises(ValueError):study.panel_jobs(tmp_path)


def test_report_rejects_duplicate_or_wrong_panel():
    with pytest.raises(ValueError):study.build_report({'jobs':[]},[])


def test_shared_controller_dispatches_backend_without_historical_panel(tmp_path,monkeypatch):
    calls=[];backend=SimpleNamespace(verify_plan=lambda p,*a:calls.append('verify'),build_report=lambda p,r:dict(records=r))
    state=bench.run({'session_seconds':2,'jobs':[]},tmp_path,backend=backend)
    assert state['status']=='completed' and calls==['verify','verify']
    assert json.loads((tmp_path/'report.json').read_text())=={'records':[]}
