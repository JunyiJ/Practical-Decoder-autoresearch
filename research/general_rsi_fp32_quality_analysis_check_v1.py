"""Synthetic checks for the new symmetric analysis adapter. No training/scoring."""
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
from research import general_rsi_fp32_quality_analysis_v1 as a
from research import general_rsi_fp32_quality_generation_analysis_v1 as g
from research.general_rsi_equal_time_analysis_v1 import bootstrap


def main():
    c=a.read(a.PANEL/'plan.json')['acceptance']; counts=np.array([1,1,2]); sources=['fineweb_edu','fineweb_edu','wikipedia']
    def decide(ces): return a.decide(np.array(ces)[:,None]*counts,counts,sources,c)
    checks={}
    assert decide([-.04]*3)['selected_arm']=='byte_composer_fp32'
    assert decide([.04]*3)['selected_arm']=='existing_byte'
    assert decide([-.02]*3)['selected_arm']=='byte_composer_fp32'
    assert decide([.02]*3)['selected_arm']=='existing_byte'
    checks['symmetric_winners_and_inclusive_mean_threshold']=True
    assert decide([-.01]*3)['selected_arm'] is None
    assert decide([.01]*3)['selected_arm'] is None
    assert decide([-.1,-.1,0])['selected_arm'] is None
    assert decide([-.1,-.1,.01])['selected_arm'] is None
    assert decide([0,0,0])['selected_arm'] is None
    checks['small_mixed_zero_deltas_rejected']=True
    for sign in [1,-1]:
        d=a.decide(sign*np.tile([-10.,-10.,.06],(3,1)),[100,100,1],sources,c)
        assert d['selected_arm'] is None
    checks['both_source_regression_guards']=True
    noisy=a.decide(np.tile([-.2,-.2,.1],(3,1)),[1,1,1],sources,c)
    assert noisy['directions']['byte_composer_fp32']['bootstrap_upper_99']>0 and noisy['selected_arm'] is None
    checks['uncertain_bound_rejected']=True
    mean,draws=bootstrap(np.tile([-1.,-1.,-6.],(3,1)),[1,1,100],773,32)
    rng=np.random.Generator(np.random.PCG64(773)); oracle=[]
    for _ in range(32):
        ids=rng.integers(0,3,size=3)
        oracle.append(sum([-1.,-1.,-6.][i] for i in ids)/sum([1,1,100][i] for i in ids))
    assert abs(mean+8/102)<1e-12 and np.array_equal(draws,np.array(oracle))
    checks['unequal_target_count_bootstrap_oracle']=True
    for bad,n in [(np.full((3,3),np.nan),counts),(np.zeros((3,3)),[1,0,1])]:
        try:a.decide(bad,n,sources,c)
        except AssertionError:pass
        else:raise AssertionError('Invalid inputs accepted')
    checks['invalid_input_rejection']=True
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory);(p/'plan.json').write_bytes((a.PANEL/'plan.json').read_bytes())
        with patch.object(a,'PANEL',p):
            try:a.analyze()
            except AssertionError as e:assert str(e)=='Six accepted endpoints required'
            else:raise AssertionError('Missing endpoints accepted')
    checks['incomplete_panel_rejected_before_scores']=True
    # Exercise aggregation on synthetic144 rows, separate arms/decoders/seeds,
    # unassessable values and equal prompt weighting. Never read real mappings.
    with tempfile.TemporaryDirectory() as directory:
        p=Path(directory);out=p/'generation-evaluation';out.mkdir()
        (p/'plan.json').write_bytes((a.PANEL/'plan.json').read_bytes())
        blind=[];mapping=[];metrics=[];annotations=[]
        for arm in ['existing_byte','byte_composer_fp32']:
            for bundle in [1,2,3]:
                for decoder in ['greedy','temperature_0.8_top_k_40']:
                    for prompt in range(12):
                        key=f'b{len(blind):03d}';blind.append(dict(id=key,prompt={'id':f'p{prompt:02d}'},continuation='synthetic text'))
                        mapping.append(dict(id=key,arm=arm,bundle=bundle,decoder=decoder))
                        metrics.append(dict(id=key,content_tokens=3,repeated_trigram_fraction=0.,terminal_loop=False,stopped_eos=False,hit_token_cap=True,unexpected_early_stop=False))
                        annotations.append(dict(id=key,relevance=2,consistency='unassessable' if prompt==0 else 1,reason='Synthetic fixture',evidence_quote='synthetic',annotator_identity='synthetic fixture'))
        for name,data in [('blind',blind),('mapping',mapping),('metrics',metrics),('annotations',annotations)]:g.put(out/f'{name}.json',data)
        g.put(out/'preparation.json',dict(blind_sha256=g.sha(out/'blind.json'),mapping_sha256=g.sha(out/'mapping.json'),metrics_sha256=g.sha(out/'metrics.json'),plan_sha256=g.BINDING,input_sha256={}))
        with patch.object(g,'P',p),patch.object(g,'OUT',out),patch('builtins.print'):
            g.aggregate()
        s=g.read(out/'summary.json');assert s['count']==144
        for arm in s['groups'].values():
            for d in arm.values():
                assert d['all_seeds']['count']==36 and d['all_seeds']['consistency']['ordinal_counts']['unassessable']==3
                assert d['all_seeds']['relevance']['equal_weight_prompt_mean']==2
                assert all(x['count']==12 for x in d['per_seed'].values())
    checks['synthetic144_generation_aggregation']=True
    g.put(a.PANEL/'analysis-check.json',dict(qualified=True,checks=checks,source_sha256={str(p.relative_to(a.ROOT)):a.sha(p) for p in [Path(__file__),a.ROOT/'research/general_rsi_fp32_quality_analysis_v1.py',a.ROOT/'research/general_rsi_fp32_quality_generation_analysis_v1.py',a.ROOT/'research/general_rsi_equal_time_analysis_v1.py']},scientific_endpoint_scores_read=False,test_opened=False))
    print(json.dumps(checks))

if __name__=='__main__':main()
