"""Readonly retained H200 scoring versus independent CPU FP64 logsumexp oracle."""
import copy,gc,json,os,math
from pathlib import Path
import numpy as np
import pytest,torch
from research.block_norm_v1 import validation as v, campaign_contract as c


@pytest.fixture(scope='module',params=['rms_incumbent','layer_matched_epsilon'])
def scored(request):
    plan=json.loads(Path(os.environ['BLOCK_NORM_PLAN_PATH']).read_text())
    binding=plan['bindings'][request.param]
    rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone()
    gpu,data,cfg=v.load(binding,'mps');report=dict(binding=copy.deepcopy(binding),**v.score(gpu,data))
    v.validate_report(report,binding,data)
    del gpu;gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
    cpu,independent,_=v.load(binding,'cpu');ids=independent.val_data;oracle=[];coverage=[]
    with torch.no_grad():
        for start in range(0,len(ids)-1,128):
            stop=min(start+128,len(ids)-1);x=ids[start:stop].unsqueeze(0)
            logits,_=cpu(x);values=logits[0].double();y=ids[start+1:stop+1]
            nll=torch.logsumexp(values,dim=-1)-values[torch.arange(stop-start),y]
            oracle.extend(nll.tolist());coverage.extend(range(start+1,stop+1))
    del cpu,values,logits;gc.collect()
    assert torch.equal(rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
    assert coverage==list(range(1,len(ids)))
    v.artifacts(binding)
    yield dict(arm=request.param,binding=binding,data=data,report=report,oracle=oracle,coverage=coverage)
    v.artifacts(binding);gc.collect();torch.mps.synchronize();torch.mps.empty_cache()


def test_full_validation_matches_independent_cpu_oracle(scored,record_property):
    report=scored['report'];oracle=scored['oracle'];target=report['target']
    assert target['targets']==1598 and target['validation_tokens']==1599
    assert target['window_counts']==[128]*12+[62] and target['block_counts']==[1598]
    total=math.fsum(oracle);windows=[math.fsum(oracle[i:i+128]) for i in range(0,len(oracle),128)]
    errors=[abs(a-b)/n for a,b,n in zip(report['window_loss_sums'],windows,target['window_counts'])]
    assert max(errors)<3e-5 and abs(report['ce']-total/len(oracle))<3e-5
    assert abs(report['block_loss_sums'][0]-total)/1598<3e-5
    assert target['holdout_scored'] is False
    record_property('validation',dict(arm=scored['arm'],report=report,independent_cpu_fp64_ce=total/1598,
        max_window_mean_nll_error=max(errors),mean_nll_error=abs(report['ce']-total/1598),rng_preserved=True,
        complete_target_offsets=True,quality_comparison=False))


@pytest.mark.parametrize('case',['checkpoint','target_hash','missing_tail','target_count','wrong_tokenizer','loss_sum','ce','byte_score','nan','extra','arithmetic'])
def test_corrupted_report_rejected(scored,case):
    x=copy.deepcopy(scored['report']);binding=scored['binding']
    if case=='checkpoint':x['binding']['h200']['sha256']='0'*64
    elif case=='target_hash':x['target']['target_ids_sha256']='0'*64
    elif case=='missing_tail':x['window_loss_sums'].pop()
    elif case=='target_count':x['target']['targets']-=1
    elif case=='wrong_tokenizer':x['target']['tokenizer']['name']='character'
    elif case=='loss_sum':x['block_loss_sums'][0]+=1
    elif case=='ce':x['ce']+=.1
    elif case=='byte_score':x['bits_per_byte']+=.1
    elif case=='nan':x['window_loss_sums'][0]=float('nan')
    elif case=='extra':x['holdout_ce']=0.
    else:x['loss_dtype']='float16'
    with pytest.raises(ValueError):v.validate_report(x,binding,scored['data'])


@pytest.mark.parametrize('kind',['config','h200','parent'])
def test_artifact_hash_mismatch_rejected_before_load(scored,kind):
    b=copy.deepcopy(scored['binding']);b[kind]['sha256']='0'*64
    with pytest.raises(ValueError,match='Artifact hash'):v.load(b,'cpu')


def test_evaluation_api_replay_exact(scored,record_property):
    got=v.evaluate(scored['binding'],'mps')
    assert c.canonical(got)==c.canonical(scored['report'])
    record_property('exact_mps_evaluation_replay',True)


def test_multi_block_tail_with_analytic_uniform_logits(record_property):
    from types import SimpleNamespace
    class Uniform(torch.nn.Module):
        def __init__(self):
            super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.cfg=SimpleNamespace(block_size=128);self.windows=[]
        def forward(self,x):
            self.windows.append(x[0].tolist())
            return torch.zeros((1,x.shape[1],50257)),None
    ids=torch.arange(8328,dtype=torch.int64)%100
    data=SimpleNamespace(val_data=ids,tokenizer_contract={'name':'gpt2'})
    model=Uniform().eval();result=v.score(model,data)
    assert result['target']['window_counts']==[128]*65+[7]
    assert result['target']['block_counts']==[2048]*4+[135]
    assert [t for window in model.windows for t in window]==ids[:-1].tolist()
    assert abs(result['ce']-math.log(50257))<2e-6
    for value,count in zip(result['block_loss_sums'],result['target']['block_counts']):assert abs(value/count-math.log(50257))<2e-6
    record_property('analytic_multiblock',dict(targets=8327,windows=66,blocks=5,window_tail=7,block_tail=135,mean_loss_error=abs(result['ce']-math.log(50257))))
