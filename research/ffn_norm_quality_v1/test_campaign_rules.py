"""Bounded synthetic rules/oracle tests; fixture seeds are not campaign reservations."""
import copy
import math
import pytest
import numpy as np
from research.ffn_norm_quality_v1 import campaign_rules as rules

SEEDS=dict(pilot=373301,screen=373302,fresh=[373311,373312,373313],evaluation=373303,bootstrap=373304)


def record(spec,ce=5.):
    return dict(spec,status='completed',**({} if spec['stage']=='pilot' else {'h200_ce':ce}))


def screens():return [record(s,5.+i/10) for i,s in enumerate(rules.screen_specs(SEEDS))]
def pilots():return [record(s) for s in rules.pilot_specs(SEEDS)]


def test_exact_finite_orders_and_parameter_scope():
    p=rules.pilot_specs(SEEDS);s=rules.screen_specs(SEEDS)
    assert len(p)==2 and len(s)==6
    assert [(v['lr'],v['arm']) for v in s]==[(lr,arm) for lr,arms in zip(rules.RATES,[rules.ARMS,rules.ARMS[::-1],rules.ARMS]) for arm in arms]
    selected={arm:rules.RATES[i] for i,arm in enumerate(rules.ARMS)}
    c=rules.confirmation_specs(SEEDS,selected)
    assert [(x['seed'],x['arm']) for x in c]==[(seed,arm) for seed,arms in zip(SEEDS['fresh'],[rules.ARMS,rules.ARMS[::-1],rules.ARMS]) for arm in arms]
    assert [x['steps'] for x in p+s+c]==[20]*2+[1200]*12
    assert all(x['width']==256 and x['batch_size']==8 and x['context']==128 and x['policy']==rules.POLICY for x in p+s+c)


@pytest.mark.parametrize('case',['missing','extra','bool','negative','large','short_fresh','tuple_fresh','duplicate_fresh','overlap_screen','overlap_eval'])
def test_seed_reservation_rejections(case):
    s=copy.deepcopy(SEEDS)
    if case=='missing':del s['bootstrap']
    elif case=='extra':s['test']=1
    elif case=='bool':s['pilot']=True
    elif case=='negative':s['pilot']=-1
    elif case=='large':s['pilot']=2**31
    elif case=='short_fresh':s['fresh'].pop()
    elif case=='tuple_fresh':s['fresh']=tuple(s['fresh'])
    elif case=='duplicate_fresh':s['fresh'][1]=s['fresh'][0]
    elif case=='overlap_screen':s['fresh'][0]=s['screen']
    else:s['evaluation']=s['bootstrap']
    with pytest.raises(ValueError):rules.pilot_specs(s)


def test_ties_choose_lower_lr_and_keep_each_arm_independent():
    rows=[record(s,5.) for s in rules.screen_specs(SEEDS)]
    assert rules.choose_rates(SEEDS,rows)==dict.fromkeys(rules.ARMS,.000625)
    # Minimum candidate LR differs; no historical control or pooled-LR selector.
    for row in rows:
        if row['arm']==rules.ARMS[1] and row['lr']==.0025:row['h200_ce']=4.
    assert rules.choose_rates(SEEDS,rows)==dict(zip(rules.ARMS,[.000625,.0025]))


@pytest.mark.parametrize('case',['missing','extra','order','duplicate','failed','bool_ce','nan','inf','negative','wrong_seed','wrong_policy','wrong_step','bool_width','extra_field','wrong_stage'])
def test_bad_screen_endpoints_rejected(case):
    rows=screens()
    if case=='missing':rows.pop()
    elif case=='extra':rows.append(copy.deepcopy(rows[0]))
    elif case=='order':rows[0],rows[1]=rows[1],rows[0]
    elif case=='duplicate':rows[1]=copy.deepcopy(rows[0])
    elif case=='failed':rows[0]['status']='failed'
    elif case in ['bool_ce','nan','inf','negative']:rows[0]['h200_ce']={'bool_ce':True,'nan':float('nan'),'inf':float('inf'),'negative':-1.}[case]
    elif case=='wrong_seed':rows[0]['seed']+=1
    elif case=='wrong_policy':rows[0]['policy']='tokenizer_mla_block_norm_v1'
    elif case=='wrong_step':rows[0]['steps']=1199
    elif case=='bool_width':rows[0]['width']=True
    elif case=='extra_field':rows[0]['test_ce']=4.
    else:rows[0]['stage']='confirm'
    with pytest.raises(ValueError):rules.choose_rates(SEEDS,rows)


def test_every_finite_prefix_and_boundary_is_serial_and_inputs_unchanged():
    p=[];rows=[];choice=None;pilot_release=False;loser_release=False;actions=[]
    for _ in range(18):
        before=copy.deepcopy((SEEDS,p,rows,choice))
        a=rules.next_action(SEEDS,p,rows,choice,pilots_released=pilot_release,losers_released=loser_release)
        assert before==(SEEDS,p,rows,choice);actions.append(a['kind'])
        if a['kind']=='pilot':p.append(record(a['spec']))
        elif a['kind']=='quality':rows.append(record(a['spec']))
        elif a['kind']=='review_pilot_boundary':pilot_release=True
        elif a['kind']=='save_selection':choice=a['value']
        elif a['kind']=='review_screen_boundary':loser_release=True
        else:assert a==dict(kind='review_final',adopted=False);break
    assert actions==['pilot']*2+['review_pilot_boundary']+['quality']*6+['save_selection','review_screen_boundary']+['quality']*6+['review_final']
    assert len(p)==2 and len(rows)==12 and {r['seed'] for r in rows[6:]}==set(SEEDS['fresh'])


@pytest.mark.parametrize('case',['quality_before_pilots','pilot_release_early','quality_before_release','selection_early','losers_early','fresh_before_selection','selection_changed','fresh_before_release','extra_fresh','wrong_fresh_order','bool_flags'])
def test_invalid_transitions_rejected(case):
    p=pilots();rows=screens();choice=rules.choose_rates(SEEDS,rows);pr=True;lr=False
    if case=='quality_before_pilots':p=p[:1];pr=False
    elif case=='pilot_release_early':p=p[:1];rows=[];choice=None
    elif case=='quality_before_release':pr=False
    elif case=='selection_early':rows.pop()
    elif case=='losers_early':rows.pop();choice=None;lr=True
    elif case=='fresh_before_selection':rows.append(record(rules.confirmation_specs(SEEDS,choice)[0]));choice=None
    elif case=='selection_changed':choice[rules.ARMS[0]]=.0025
    elif case=='fresh_before_release':rows.append(record(rules.confirmation_specs(SEEDS,choice)[0]))
    elif case in ['extra_fresh','wrong_fresh_order']:
        rows += [record(s) for s in rules.confirmation_specs(SEEDS,choice)];lr=True
        if case=='extra_fresh':rows.append(copy.deepcopy(rows[-1]))
        else:rows[-1],rows[-2]=rows[-2],rows[-1]
    else:pr=1
    with pytest.raises(ValueError):rules.next_action(SEEDS,p,rows,choice,pilots_released=pr,losers_released=lr)


def test_retention_only_identifies_losing_screens_and_never_authorizes_deletion():
    rows=screens();choice=rules.choose_rates(SEEDS,rows);r=rules.screening_retention(SEEDS,rows,choice)
    assert r==dict(selected_indices=[0,1],losing_indices=[2,3,4,5],fresh_pairs_always_protected=True,deletion_authorized=False)
    with pytest.raises(ValueError):rules.screening_retention(SEEDS,rows+[record(rules.confirmation_specs(SEEDS,choice)[0])],choice)
    with pytest.raises(ValueError):rules.screening_retention(SEEDS,rows,{rules.ARMS[0]:.0025,rules.ARMS[1]:.0025})


def oracle(loss,counts):
    # Python scalar sums and explicit per-seed ratios; numpy supplies only the
    # declared PCG64 draws and a separate linear-quantile implementation.
    ds=[sum(float(loss[s,1,j])-float(loss[s,0,j]) for j in range(len(counts)))/sum(counts) for s in range(3)]
    bounds={}
    for size in [2048,4096,8192]:
        groups=[list(range(i,min(i+size//2048,len(counts)))) for i in range(0,len(counts),size//2048)]
        rng=np.random.Generator(np.random.PCG64(SEEDS['bootstrap']))
        indices=rng.integers(0,len(groups),size=(2000,len(groups)))
        draws=[]
        for sampled in indices:
            token_count=sum(int(counts[j]) for k in sampled for j in groups[k])
            values=[sum(float(loss[s,1,j])-float(loss[s,0,j]) for k in sampled for j in groups[k])/token_count for s in range(3)]
            draws.append(sum(values)/3)
        bounds[str(size)]=dict(gate_upper=float(np.quantile(draws,1-.05/3,method='linear')),
                              descriptive_95pct=np.quantile(draws,[.025,.975],method='linear').tolist())
    return dict(seed_deltas=ds,mean_delta=sum(ds)/3,bounds=bounds)


def panel():
    counts=np.array([2048]*6+[137],dtype=np.int64)
    loss=np.zeros((3,2,7),dtype=np.float64)
    for s in range(3):
        loss[s,0]=counts*(5.+s/16)
        loss[s,1]=loss[s,0]+np.array([-80,-30,40,-90,-12,11,60],dtype=np.float64)*(1+s/8)
    return loss,counts


def test_paired_bootstrap_independent_oracle_tail_weights_and_repeatability(record_property):
    loss,counts=panel();before=loss.copy();n=counts.copy()
    got=rules.paired_statistics(SEEDS,loss,counts);expected=oracle(loss,counts)
    assert got==rules.paired_statistics(SEEDS,loss,counts)
    assert np.array_equal(before,loss) and np.array_equal(n,counts)
    np.testing.assert_allclose(got['seed_deltas'],expected['seed_deltas'],atol=5e-14,rtol=0)
    assert abs(got['mean_delta']-expected['mean_delta'])<5e-14
    error=0.
    for size in expected['bounds']:
        a=[got['bounds'][size]['gate_upper'],*got['bounds'][size]['descriptive_95pct']]
        b=[expected['bounds'][size]['gate_upper'],*expected['bounds'][size]['descriptive_95pct']]
        error=max(error,max(abs(x-y) for x,y in zip(a,b)))
        np.testing.assert_allclose(a,b,atol=5e-14,rtol=0)
    assert abs(got['mean_delta']-float(((loss[:,1]-loss[:,0])/counts).mean()))>.01
    record_property('max_independent_bootstrap_oracle_error',error)
    record_property('bootstrap_draws_per_block_size',2000)


@pytest.mark.parametrize('case',['few_counts','too_many','float_counts','bool_counts','short_interior','zero_tail','large_tail','two_seeds','one_arm','negative_loss','nan_loss','bool_loss','overflow'])
def test_statistics_bad_panels_rejected(case):
    loss,counts=panel()
    if case=='few_counts':counts=counts[:3];loss=loss[:,:,:3]
    elif case=='too_many':counts=np.full(513,2048);loss=np.ones((3,2,513))
    elif case=='float_counts':counts=counts.astype(float)
    elif case=='bool_counts':counts=counts.astype(bool)
    elif case=='short_interior':counts[0]=100
    elif case=='zero_tail':counts[-1]=0
    elif case=='large_tail':counts[-1]=2049
    elif case=='two_seeds':loss=loss[:2]
    elif case=='one_arm':loss=loss[:,:1]
    elif case=='negative_loss':loss[0,0,0]=-1
    elif case=='nan_loss':loss[0,0,0]=np.nan
    elif case=='bool_loss':loss=loss.astype(bool)
    else:loss[:,0,:]=1e308;loss[:,1,:]=0
    with pytest.raises(ValueError):rules.paired_statistics(SEEDS,loss,counts)


def stats(ds=(-.03,-.03,-.03),upper=-.005):
    return dict(seed_deltas=list(ds),mean_delta=sum(ds)/3,bounds={str(b):dict(gate_upper=upper,descriptive_95pct=[-.08,min(-.01,upper)]) for b in rules.BLOCKS})


@pytest.mark.parametrize('ds,upper,passed',[((-.03,-.03,-.03),-.005,True),((-.02,-.02,-.02),-.005,False),((-.019,-.019,-.019),-.005,False),((-.06,-.06,0.),-.005,False),((-.06,-.06,.001),-.005,False),((-.03,-.03,-.03),0.,False)])
def test_strict_gates_never_adopt(ds,upper,passed):
    assert rules.quality_gate(stats(ds,upper))==dict(strict_quality=passed,adopted=False)


@pytest.mark.parametrize('case',['missing_size','mean_mismatch','nan_bound','bad_interval','bool_delta','extra','upper_below_descriptive'])
def test_malformed_gate_summaries_rejected(case):
    s=stats()
    if case=='missing_size':del s['bounds']['8192']
    elif case=='mean_mismatch':s['mean_delta']=-.5
    elif case=='nan_bound':s['bounds']['2048']['gate_upper']=float('nan')
    elif case=='bad_interval':s['bounds']['4096']['descriptive_95pct']=[-.01,-.08]
    elif case=='bool_delta':s['seed_deltas'][0]=True
    elif case=='extra':s['test_improved']=True
    else:s['bounds']['8192']['gate_upper']=-.09
    with pytest.raises(ValueError):rules.quality_gate(s)
