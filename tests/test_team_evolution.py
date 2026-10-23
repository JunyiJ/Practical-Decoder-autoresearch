"""Synthetic evolutionary controls; no model training or external agents."""
from copy import deepcopy

import pytest

from src.research.team import coordinator
from src.research.team.evolution import (plan_generation, population, validate_child,
                                         compare_parents, validate_settings)
from test_team_coordinator import (campaign, read, save, install_fake_handoffs,
                                   advance_to, responses)


def state():
    return {'config': {'evolution': {'population_size': 4, 'max_mutations': 2, 'explore_every': 4}},
            'baseline_config': {'training': {'learning_rate': .0006, 'weight_decay': .1, 'grad_clip': 1},
                                'model': {'dropout': 0}},
            'plan': {'widths': [64, 128], 'seeds': [11, 22, 33]}, 'iterations': []}


def member(s, lr=.002, decay=.1, loss=2., transfer='contradicted'):
    cfg = deepcopy(s['baseline_config']);cfg['training'].update(learning_rate=lr, weight_decay=decay)
    n=len(s['iterations'])+1
    item={'id': f'iteration-{n:03d}', 'stage':'completed', 'result_id': f'result-{n}',
          'idea': {'change_kind':'config'}, 'candidate_config':cfg,
          'report':{'local_outcome':'supported', 'transfer_outcome':transfer, 'evidence_integrity':True,
                    'performance':[{'target':'width-64','metric':'validation_ce_delta','complete':True,'candidate_mean':loss}],
                    'records':[{'target':f'width-{w}','seed':seed,'arm':'candidate','status':'completed','metric':loss}
                               for w in [64,128] for seed in [11,22,33]]}}
    s['iterations'].append(item);return item


def test_population_uses_only_small_width_and_deduplicates():
    s=state();member(s,loss=1.9);member(s,lr=.001,loss=2.0,transfer='supported');member(s,loss=2.1)
    p=population(s)
    assert len(p)==2 and p[0]['iteration']=='iteration-001'
    s['iterations'][0]['report']['transfer_outcome']='supported'
    assert [m['iteration'] for m in population(s)]==[m['iteration'] for m in p]
    s['iterations'][0]['report']['evidence_integrity']=False
    assert population(s)[0]['iteration']=='iteration-002'


def test_real_crossover_and_exploration():
    s=state();member(s,lr=.002,decay=.03,loss=1.9);member(s,lr=.001,loss=2.)
    p=plan_generation(s)
    assert p['mode']=='crossover' and len(p['parents'])==2
    assert p['inherited_genotype']=={'training.learning_rate':.001,'training.weight_decay':.03}
    member(s,lr=.0015)
    assert plan_generation(s)['mode']=='explore'
    assert plan_generation(s)['parents']==[]


def test_same_locus_does_not_create_fake_combination():
    s=state();member(s,lr=.002);member(s,lr=.001,loss=2.1)
    assert plan_generation(s)['mode']=='mutate'


def test_child_budget_counts_effective_inherited_changes():
    s=state();member(s)
    i={'evolution':plan_generation(s),'idea':{'change_kind':'config'},'candidate_config':deepcopy(s['iterations'][0]['candidate_config'])}
    with pytest.raises(ValueError,match='unchanged parent'):validate_child(s,i,[])
    i['candidate_config']['training']['weight_decay']=.03
    validate_child(s,i,[])
    assert i['evolution_child']['mutated_paths']==['training.weight_decay']
    i['candidate_config']['training']['grad_clip']=.5
    i['candidate_config']['model']['dropout']=.1
    with pytest.raises(ValueError,match='bound exceeded'):validate_child(s,i,[])
    with pytest.raises(ValueError,match='source edits'):validate_child(s,i,[{'path':'src/models/gpt.py'}])


def test_crossover_requires_exact_recipe():
    s=state();member(s,decay=.03,loss=1.9);member(s,lr=.001,loss=2.)
    i={'evolution':plan_generation(s),'idea':{'change_kind':'config'},'candidate_config':deepcopy(s['iterations'][1]['candidate_config'])}
    i['candidate_config']['training']['weight_decay']=.03
    validate_child(s,i,[])
    i['candidate_config']['training']['grad_clip']=.5
    with pytest.raises(ValueError,match='bound exceeded'):validate_child(s,i,[])


def test_parent_comparison_missing_pairs_and_tampering():
    s=state();old=member(s)
    i={'evolution':plan_generation(s),'evolution_child':{}}
    report=deepcopy(old['report'])
    for r in report['records']:r['metric']-=.1
    comparison=compare_parents(s,i,report)
    assert comparison['comparisons'][0]['mean_child_minus_parent']==pytest.approx(-.1)
    report['records'].pop(0)
    assert compare_parents(s,i,report)['comparisons'][0]['mean_child_minus_parent'] is None
    old['report']['records'][0]['metric']=100
    with pytest.raises(ValueError,match='evidence changed'):compare_parents(s,i,report)


def test_generation_is_registered_and_coder_cannot_overmutate(campaign):
    s=read(campaign);s['config']['evolution']=state()['config']['evolution'];save(campaign,s)
    r=responses();r['coder']['config_overrides'] += [
        {'path':'training.weight_decay','value_json':'0.03'},
        {'path':'training.grad_clip','value_json':'0.5'}]
    install_fake_handoffs(campaign,replies=r)
    result=advance_to(campaign,'coder')
    assert result['iterations'][0]['evolution']['mode']=='explore'
    result=coordinator.advance(campaign['output'],run_agents=True,allow_training=True)
    assert result['status']=='needs_attention' and 'bound exceeded' in result['iterations'][0]['error']
    assert campaign['events']['suites']==[]


def test_generation_tampering_invalidates_prediction(campaign):
    s=read(campaign);s['config']['evolution']=state()['config']['evolution'];save(campaign,s)
    install_fake_handoffs(campaign);s=advance_to(campaign,'coder')
    s['iterations'][0]['evolution']['max_mutations']=99;save(campaign,s)
    s=coordinator.advance(campaign['output'],run_agents=True,allow_training=True)
    assert s['status']=='needs_attention' and 'Prediction no longer matches' in s['iterations'][0]['error']


@pytest.mark.parametrize('key,value',[('population_size',0),('max_mutations',3),('explore_every',2),('population_size',True)])
def test_invalid_settings(key,value):
    settings=state()['config']['evolution'];settings[key]=value
    with pytest.raises(ValueError):validate_settings(settings)


def test_two_generations_preserve_baseline_and_record_parent_comparison(campaign):
    from src.research.team.knowledge import KnowledgeBank
    s=read(campaign);s['config']['evolution']=state()['config']['evolution'];save(campaign,s)
    records=[{'trial_id':f'{width}_{seed}_{arm}', 'target':f'width-{width}', 'width':width,
              'stage':'local' if width==16 else 'transfer', 'seed':seed, 'arm':arm,
              'status':'completed', 'metric':2. if arm=='baseline' else 1.9,
              'tokens_processed':32,'training_time_sec':1.,'rss_mb':100,'duration_seconds':1.}
             for width in [16,32] for seed in [1,2] for arm in ['baseline','candidate']]
    report={'local_outcome':'supported','transfer_outcome':'supported','evidence_integrity':True,
            'holdout_consumed':False,'records':records}
    replies=install_fake_handoffs(campaign,report=report)
    s=advance_to(campaign,'completed')
    first=s['iterations'][0]
    replies['coder']['config_overrides']=[{'path':'training.learning_rate','value_json':'0.0002'}]
    coordinator.advance(campaign['output'],run_agents=True,allow_training=True)
    s=advance_to(campaign,'completed');second=s['iterations'][1]
    assert second['evolution']['parents'][0]['result_id']==first['result_id']
    assert second['evolution_child']['mutated_paths']==['training.learning_rate']
    assert second['report']['evolution']['comparisons'][0]['mean_child_minus_parent']==0.
    assert second['suite']['baseline']['config']==first['suite']['baseline']['config']
    bank=KnowledgeBank(s['config']['knowledge_bank'])
    assert bank.get(second['result_id'])['details']['evolution']['plan']==second['evolution']
    assert len(campaign['events']['suites'])==2


def test_crossover_does_not_retest_existing_genotype():
    s=state();member(s,lr=.002,decay=.03,loss=1.9);member(s,lr=.002,loss=2.)
    assert plan_generation(s)['mode']=='mutate'


def test_deleting_generation_cannot_bypass_registered_controls(campaign):
    s=read(campaign);s['config']['evolution']=state()['config']['evolution'];save(campaign,s)
    install_fake_handoffs(campaign);s=advance_to(campaign,'coder')
    del s['iterations'][0]['evolution'];save(campaign,s)
    s=coordinator.advance(campaign['output'],run_agents=True,allow_training=True)
    assert s['status']=='needs_attention' and campaign['events']['suites']==[]
