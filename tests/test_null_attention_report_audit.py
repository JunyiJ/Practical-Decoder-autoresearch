"""Independent report reconstruction on synthetic nonuniform text blocks only."""
import copy
import math
import pytest

from src.research import null_attention_study as study
from tests.test_null_attention_study import plan, complete

from src.research import null_attention_audit as audit_module


def nonuniform(input_scale=1.,winner='null_zero',alias=True):
    p=plan(input_scale); state=complete(p,.005 if alias else .0025,winner)
    for row in state['records']:
        family=(study.FAMILIES+[study.REFERENCE]).index(row['family'])
        for raw in [False,True]:
            value=study.endpoint(p,row,raw)
            base=value['mean_ce']
            sums=[count*(base+.07*math.sin(i+.13*row['seed'])+.006*family*math.cos(i/7))
                  for i,count in enumerate(value['block_target_counts'])]
            value.update(block_loss_sums=sums,mean_ce=sum(sums)/111538)
    state['selection']=study.freeze_selection(p,state['records'][:18])
    return p,state,study.summarize(p,state)


@pytest.mark.parametrize('input_scale',[1.,.5,.25])
@pytest.mark.parametrize('alias',[False,True])
@pytest.mark.parametrize('winner',study.FAMILIES)
def test_independent_nonuniform_raw_h200_and_bootstrap(input_scale,winner,alias):
    p,state,report=nonuniform(input_scale,winner,alias)
    result=audit_module.audit(p,state,report)
    assert result['physical_trajectories']==(36 if alias else 42) and result['correlated_outputs']==(72 if alias else 84)


@pytest.mark.parametrize('damage',['mean','interval','raw_interval','seed_delta','count',
    'draws','bootstrap_seed','missing_comparison','duplicate_comparison','quality',
    'adoption','screen','alias','metric','manifest','local_mean','winner','holdout'])
def test_independent_audit_rejects_corruption(damage):
    p,state,report=nonuniform()
    report=copy.deepcopy(report)
    c=report['comparisons'][0]; v=c['sensitivity'][0]
    if damage=='mean': v['mean_delta']+=.1
    elif damage=='interval': v['conditional_text_bootstrap_95pct'][1]+=.1
    elif damage=='raw_interval': c['raw_sensitivity'][2]['conditional_text_bootstrap_95pct'][0]+=.1
    elif damage=='seed_delta': v['paired_seed_deltas']['2531']+=.1
    elif damage=='count': v['text_block_count']-=1
    elif damage=='draws': v['bootstrap_draws']=1000
    elif damage=='bootstrap_seed': v['bootstrap_seed']=1
    elif damage=='missing_comparison': report['comparisons'].pop()
    elif damage=='duplicate_comparison': report['comparisons'][1]=copy.deepcopy(c)
    elif damage=='quality': c['quality_improvement']=not c['quality_improvement']
    elif damage=='adoption': report['quality_decision']['qualifies_for_incumbent_replacement']=not report['quality_decision']['qualifies_for_incumbent_replacement']
    elif damage=='screen': report['fixed_lr_screen'][0]['primary']['mean_delta']+=.1
    elif damage=='alias': report['reference_aliases']={'fixed':'standard'}
    elif damage=='metric': report['records'][0]['averaged_outputs']['200']['coverage']['evaluations']['128']['mean_ce']+=.1
    elif damage=='manifest': report['selection']['confirmation']['jobs'][0][3]=999
    elif damage=='local_mean': report['selection']['choices']['prefix_control']['mean_ce']+=.1
    elif damage=='winner': report['selection']['winner']='prefix_control'
    elif damage=='holdout': report['holdout_consumed']=True
    # Keep copied state/report payloads mutually consistent, so rejection must
    # come from reconstruction rather than a trivial equality mismatch.
    state=copy.deepcopy(state)
    state['records']=copy.deepcopy(report['records'])
    state['selection']=copy.deepcopy(report['selection'])
    with pytest.raises(AssertionError): audit_module.audit(p,state,report)
