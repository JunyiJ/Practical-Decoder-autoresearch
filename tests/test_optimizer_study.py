import copy
import pytest
from src.research.optimizer_study import FAMILIES, local_jobs, freeze_selection, summarize


def plan():
    return {'families':FAMILIES,'learning_rates':[.001,.002,.003],'seeds':[11,22,33],
            'widths':[64,128,256],'min_improvement':.005,'scope':'development'}


def rows():
    return [{'family':f,'lr':lr,'width':w,'seed':s,'metric':2.+abs(lr-.002)*100,
             'status':'completed'} for f,lr,w,s in local_jobs(plan())]


def test_all_local_grid_cells_have_paired_seeds_and_fixed_reference():
    jobs=local_jobs(plan()); assert len(jobs)==57==len(set(jobs))
    assert {w for f,lr,w,s in jobs}=={64}
    assert {lr for f,lr,w,s in jobs if f=='reference_zero_decay'}=={.002}


def test_frozen_selection_uses_local_only_and_rejects_incomplete_cells():
    evidence=rows(); selected=freeze_selection(plan(),evidence)
    assert all(c['lr']==.002 for c in selected['choices'].values())
    corrupted=copy.deepcopy(evidence);corrupted[0]['status']='failed'
    with pytest.raises(ValueError,match='incomplete'):freeze_selection(plan(),corrupted)
    with pytest.raises(ValueError,match='incomplete'):freeze_selection(plan(),evidence[:-1])


def test_summary_does_not_promote_mean_only_success():
    evidence=[]
    for family in FAMILIES:
        for width in [64,128,256]:
            for seed,delta in zip([11,22,33],[-.04,-.04,.01]):
                evidence.append({'family':family,'lr':.002,'width':width,'seed':seed,
                    'metric':2.+(delta if family!='reference_zero_decay' else 0.),
                    'training_time_sec':3,'rss_mb':100})
    choices={f:{'lr':.002,'at_grid_boundary':False} for f in FAMILIES}
    report=summarize(plan(),{'records':evidence,'selection':{'choices':choices}})
    r=report['families']['adam_mup_independent']['widths']['256']
    assert r['mean_delta_vs_reference']<-.005
    assert not r['supported_vs_reference']
