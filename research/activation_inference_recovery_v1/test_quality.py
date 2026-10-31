"""Disposable imported-row transport tests; frozen statistical engine reused."""
import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from research.activation_inference_recovery_v1 import quality
from src.research.tokenizer_activation_rules import confirmation_specs
from src.research.tokenizer_activation_final import CONTRACT,PARAMETERS


def ref(name):return dict(path=name,size=1,sha256='a'*64)


def score(ce):
    counts=[min(2048,34278-i) for i in range(0,34278,2048)]
    return dict(context=128,eval_batch_size=8,block_tokens=2048,target_count=34278,target_bytes=111536,
        window_count=math.ceil(34278/128),parameters=PARAMETERS,block_target_counts=counts,
        block_loss_sums=[n*ce for n in counts],mean_ce=ce,bits_per_byte=ce*34278/111536/math.log(2),perplexity=math.exp(ce))


@pytest.fixture
def fixture(monkeypatch):
    selection={'silu':.00125,'gelu_exact':.00125};docs={};root=Path('/disposable-original')
    imported=dict(selection=selection,pair_bindings={str(s):{'bindings':{}} for s in (3611,3612,3613)},
                  files={},final_contract=CONTRACT,plan=ref('plan'))
    for i,spec in enumerate(confirmation_specs(selection),6):
        stem=f'evidence/quality/{i:02d}';request=ref(stem+'/request.json')
        weights={role:ref(stem+'/checkpoints/'+role+'.pt') for role in ('raw','h200')}
        docs[request['path']]={'spec':spec}
        pair=dict(request=request,acceptance=ref(stem+'/accepted.json'),weights=weights)
        imported['pair_bindings'][str(spec['seed'])]['bindings'][spec['family']]=pair
        measured={role:dict(score(5+(0.01 if spec['family']=='gelu_exact' else 0)),
            checkpoint=str(root/weights[role]['path']),checkpoint_sha256=weights[role]['sha256']) for role in ('raw','h200')}
        values={'measurements':measured,'training-summary':dict(training_time_sec=100.),
                'resource-audit':dict(samples=[dict(elapsed_seconds=150.,peak_rss_bytes=1000000000,mps_driver_bytes=2000000000)])}
        for name,value in values.items():
            r=ref(stem+'/'+name+'.json');docs[r['path']]=value;imported['files'][r['path']]=r
    ledger=SimpleNamespace(root=root,document=lambda r:copy.deepcopy(docs[r['path']]))
    calls=[]
    def verify(led,expected):
        assert led is ledger and expected==imported
        calls.append(True);return copy.deepcopy(imported)
    monkeypatch.setattr(quality,'verify_source_import',verify)
    return ledger,imported,docs,calls


def test_quality_only_report_cannot_claim_inference_or_closure(fixture):
    ledger,imported,_,calls=fixture
    result=quality.analyze_imported_quality(ledger,imported)
    assert calls==[True,True]
    assert result['status']=='quality_reviewed_inference_pending'
    assert result['analysis']['strict_quality'] is False
    assert result['analysis']['statistics']['mean_delta']==pytest.approx(.01)
    assert len(result['fresh_provenance'])==6
    assert result['adopted'] is False and result['formal_final_completed'] is False
    assert result['accepted_inference_pairs']==0 and result['holdout_scored'] is False


@pytest.mark.parametrize('bad',['request','path','raw_lineage','h200_lineage','score_arithmetic','resource','contract'])
def test_altered_imported_endpoint_rejected(fixture,bad):
    ledger,imported,docs,_=fixture;stem='evidence/quality/06'
    if bad=='request':docs[stem+'/request.json']['spec']['seed']=999
    elif bad=='path':imported['files'][stem+'/measurements.json']['path']='wrong/measurements.json'
    elif bad in ('raw_lineage','h200_lineage'):docs[stem+'/measurements.json'][bad.split('_')[0]]['checkpoint_sha256']='b'*64
    elif bad=='score_arithmetic':docs[stem+'/measurements.json']['h200']['mean_ce']-=1
    elif bad=='resource':docs[stem+'/resource-audit.json']['samples'][0]['peak_rss_bytes']=4*1024**3
    else:imported['final_contract']={}
    with pytest.raises(ValueError):quality.analyze_imported_quality(ledger,imported)


def test_post_arithmetic_provenance_failure_propagates(fixture,monkeypatch):
    ledger,imported,_,_=fixture;calls=[]
    def verify(*args):
        calls.append(True)
        if len(calls)==2:raise ValueError('Frozen original import differs')
        return imported
    monkeypatch.setattr(quality,'verify_source_import',verify)
    with pytest.raises(ValueError,match='import differs'):quality.analyze_imported_quality(ledger,imported)
