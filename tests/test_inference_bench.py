import copy
from types import SimpleNamespace

import pytest
import torch

from src.research import inference_bench as bench
from src.models.gpt import GPT
from src.research.protocol import digest
from tests.test_gqa_study import gqa_config


def job():
    return dict(width=128,batch=8,layers=2,prompt=32,decode=32,capacity=128,
                repeats=9,first_role_offset=0,models={'reference':{'family':'mha_tuned'},'candidate':{'family':'gqa_half'}})


def result(j):
    measurements=[]
    for i in range(9):
        measurements.append(dict(order=['reference','candidate'] if i%2==0 else ['candidate','reference'],
             reference=dict(prefill_seconds=.01,decode_seconds=.1),
             candidate=dict(prefill_seconds=.02,decode_seconds=.08)))
    return dict(job_sha256=digest(j),device='mps',torch=str(torch.__version__),cpu_threads=4,
                dtype='float32',both_models_resident=True,holdout_consumed=False,
                peak_device_memory_bytes=None,mps_current_allocated_bytes=123,
                correctness={'reference':dict(cache_allocated_bytes=2097152,cache_logical_used_bytes=1048576,max_absolute_logit_error=1e-6),
                             'candidate':dict(cache_allocated_bytes=1048576,cache_logical_used_bytes=524288,max_absolute_logit_error=1e-6)},
                measurements=measurements,summary=bench.summarize_pair(j,measurements))


def test_paired_timing_units_and_cache_contract():
    j=job();r=result(j);bench.validate_result(j,r)
    assert r['summary']['decode']['paired_ratio_median']==pytest.approx(.8)
    assert r['summary']['decode']['candidate_ms_per_step']==pytest.approx(2.5)
    assert r['summary']['decode']['candidate_batch_tokens_per_second']==pytest.approx(3200)
    assert r['summary']['prefill']['paired_ratio_median']==2


@pytest.mark.parametrize('mutation',['timing','order','cache','used','runtime','peak','holdout','missing','aggregate'])
def test_result_rejects_corruption(mutation):
    j=job();r=result(j)
    if mutation=='timing':r['measurements'][0]['candidate']['decode_seconds']=float('nan')
    elif mutation=='order':r['measurements'][0]['order'].reverse()
    elif mutation=='cache':r['correctness']['candidate']['cache_allocated_bytes']*=2
    elif mutation=='used':r['correctness']['candidate']['cache_logical_used_bytes']+=1
    elif mutation=='runtime':r['device']='cpu'
    elif mutation=='peak':r['peak_device_memory_bytes']=123
    elif mutation=='holdout':r['holdout_consumed']=True
    elif mutation=='missing':r['correctness'].pop('reference')
    elif mutation=='aggregate':r['summary']['decode']['paired_ratio_median']=.5
    with pytest.raises(ValueError):bench.validate_result(j,r)


def test_cached_pass_matches_full_and_counts_buffers():
    torch.set_num_threads(1);torch.manual_seed(10)
    cfg=SimpleNamespace(model=gqa_config(64));model=GPT(cfg.model).eval()
    caches=bench.make_caches(model,cfg,2)
    tokens=torch.arange(16).reshape(2,8)%cfg.model.vocab_size
    with torch.no_grad():full=model(tokens)[0]
    times,cached=bench.cached_pass(model,tokens,caches,4,collect=True)
    torch.testing.assert_close(full,cached,atol=2e-6,rtol=1e-5)
    assert all(x>0 for x in times.values())
    assert bench.cache_bytes(caches)==2*2*cfg.model.block_size*64*4
    _,second=bench.cached_pass(model,tokens,caches,4,collect=True)
    torch.testing.assert_close(cached,second,atol=0,rtol=0)


def test_panel_is_fixed_and_paired(tmp_path):
    import json
    for study,family,seed,layers in bench.PANEL:
        directory=tmp_path/'research_runs'/study;directory.mkdir(parents=True)
        choices={f:{'lr':.002} for f in [family,'mha_tuned']}
        rows=[dict(family=f,width=w,seed=s,lr=lr,checkpoint=f'{f}-{w}-{s}-{lr}',checkpoint_sha256='a'*64)
              for f in choices for w in [64,128,256] for s in [seed,seed+1] for lr in [.001,.002]]
        (directory/'report.json').write_text(json.dumps(dict(records=rows,selection=dict(choices=choices))))
    jobs=bench.panel_jobs(tmp_path)
    assert len(jobs)==len({j['id'] for j in jobs})==24
    assert len({m['checkpoint'] for j in jobs for m in j['models'].values()})==12
    assert {(j['batch'],j['prompt'],j['decode']) for j in jobs}=={(1,32,32),(1,64,64),(8,32,32),(8,64,64)}
    assert all(j['seed'] in [444,1011] and j['prompt']+j['decode']<=128 for j in jobs)


def test_incomplete_report_rejected():
    with pytest.raises(ValueError):bench.build_report({},[])
