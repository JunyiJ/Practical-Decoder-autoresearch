"""Independent nonconstant byte allocation/bootstrap/LR-selection contracts,no model scores."""
import copy,json,math
from pathlib import Path
import numpy as np
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.gpt2_prefix_worker_v1.execution import sources as predecessor_sources,checked
from research.gpt2_prefix_data_v1.data import config,load
from research.tokenizer_common_bytes_v1.metric import loss_summary
from research.gpt2_prefix_statistics_v1.statistics import select,review,report_contract,WIDTHS,DRAWS,QUANTILE,TARGET_BYTES
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent

def publish(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def reject(fn):
    try:fn()
    except (ValueError,RuntimeError,KeyError,TypeError):return
    raise AssertionError('Partial/foreign/nonfinite statistical input accepted')
def synthetic(plan,target,byte_nats):
    values=[];offset=0
    for size in plan['target_token_byte_lengths']:values.append(math.fsum(byte_nats[offset:offset+size]));offset+=size
    require(offset==TARGET_BYTES,'Fixture byte coverage differs')
    result=dict(loss_summary(plan,values),target=copy.deepcopy(target),window_loss_sums=[math.fsum(values[i:i+128]) for i in range(0,len(values),128)],holdout_scored=False)
    # A separate explicit byte expansion checks the uniform token-allocation convention.
    expanded=[]
    for value,size in zip(values,plan['target_token_byte_lengths']):expanded.extend([value/size]*size)
    for width in WIDTHS:
        expected=[math.fsum(expanded[i:i+width]) for i in range(0,len(expanded),width)];observed=result['raw_byte_blocks'][str(width)]['loss_sums']
        require(max(abs(a-b) for a,b in zip(expected,observed))<1e-8,'Independent explicit byte grouping differs')
    return result

def independent(panel,seed):
    result={}
    for width in WIDTHS:
        counts=panel[0]['native']['raw_byte_blocks'][str(width)]['byte_counts'];rng=np.random.Generator(np.random.PCG64(seed));samples=[]
        for _ in range(DRAWS):
            draw=rng.integers(0,len(counts),size=len(counts));den=sum(counts[int(i)] for i in draw);values=[]
            for pair in panel:
                a=pair['native']['raw_byte_blocks'][str(width)]['loss_sums'];b=pair['prefix8192']['raw_byte_blocks'][str(width)]['loss_sums'];values.append(math.fsum(b[int(i)]-a[int(i)] for i in draw)/(den*math.log(2)))
            samples.append(math.fsum(values)/len(values))
        ordered=sorted(samples);position=QUANTILE*(len(samples)-1);lo=int(math.floor(position));hi=int(math.ceil(position));fraction=position-lo;result[str(width)]=ordered[lo]*(1-fraction)+ordered[hi]*fraction
    return result
with session_lock(GLOBAL_LOCK.parent):
    torch.set_num_threads(4);idle();env=environment();require(not(E/'statistics-contracts001-plan.json').exists(),'New statistical contract fixture only')
    predecessor=checked(reference(E/'prerequisite-close001.json'));require(predecessor['status']=='actual_native_and8K_workers_pilots_and_byte_exposure_closed','Actual prerequisite closure required')
    pins=predecessor_sources()
    for p in (ROOT/'research/gpt2_prefix_statistics_v1').glob('*.py'):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
    require(len(pins)==995,'Exact additive statistics source count required')
    for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Proven source changed')
    publish(E/'statistics-contracts001-plan.json',dict(source=reference(__file__),sources=pins,prerequisite_close=reference(E/'prerequisite-close001.json'),synthetic_nonconstant_fixture_only=True,no_training_or_model_scoring=True,no_scientific_seed=True))
    try:
        with CpuPrerequisiteGuard(E/'statistics-contracts001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            original=checked(reference(E/'worker001-review.json'));targets={r['arm']:r['acceptance']['full_validation']['h200']['mps']['target'] for r in original['rows']};plans={a:load(config(a)).common_byte_plan for a in ['native','prefix8192']};panel=[]
            for pair in range(3):
                base=[1.4+.003*pair+.08*math.sin(i/2047+.1*pair) for i in range(TARGET_BYTES)];candidate=[v-(.025+.004*pair)+.01*math.sin(i/997) for i,v in enumerate(base)]
                panel.append(dict(seed=600440+pair,native=synthetic(plans['native'],targets['native'],base),prefix8192=synthetic(plans['prefix8192'],targets['prefix8192'],candidate)))
            actual=review(panel,600443);oracle=independent(panel,600443);maximum=max(abs(actual['intervals'][w]['upper']-upper) for w,upper in oracle.items());require(maximum<1e-12 and actual['quality_gates_pass'],'Independent nonconstant bootstrap/linear quantile or gates differ')
            screens=[]
            for arm in ['native','prefix8192']:
                for lr,factor in [(.000625,1.03),(.00125,1.),(.0025,1.02)]:
                    values=[factor*(1.4+.08*math.sin(i/2047)) for i in range(TARGET_BYTES)];screens.append(dict(arm=arm,peak_lr=lr,status='accepted',h200=synthetic(plans[arm],targets[arm],values)))
            require(select(screens)=={'native':.00125,'prefix8192':.00125},'Independent minimum-LR choice differs')
            tie=copy.deepcopy(screens)
            for arm in ['native','prefix8192']:
                low=next(r for r in tie if r['arm']==arm and r['peak_lr']==.000625);mid=next(r for r in tie if r['arm']==arm and r['peak_lr']==.00125);low['h200']=copy.deepcopy(mid['h200'])
            require(select(tie)=={'native':.000625,'prefix8192':.000625},'Lower-LR exact tie differs')
            bad_panel=[];bad_panel.append(panel[:2]);bad=copy.deepcopy(panel);bad[2]['seed']=bad[0]['seed'];bad_panel.append(bad)
            bad=copy.deepcopy(panel);bad[0]['native'],bad[0]['prefix8192']=bad[0]['prefix8192'],bad[0]['native'];bad_panel.append(bad)
            for field,value in [('bits_per_byte',float('nan')),('target_sha256','0'*64)]:bad=copy.deepcopy(panel);bad[0]['native'][field]=value;bad_panel.append(bad)
            bad=copy.deepcopy(panel);bad[0]['native']['target']['prefix_bytes']=63;bad_panel.append(bad)
            bad=copy.deepcopy(panel);bad[0]['prefix8192']['raw_byte_blocks']['4096']['byte_counts'][0]-=1;bad_panel.append(bad)
            bad=copy.deepcopy(panel);bad[0]['native']['raw_byte_blocks']['8192']['loss_sums'][0]=float('inf');bad_panel.append(bad)
            for bad in bad_panel:reject(lambda bad=bad:review(bad,600443))
            bad_screens=[screens[:5],screens+[screens[0]]];bad=copy.deepcopy(screens);bad[1]=copy.deepcopy(bad[0]);bad_screens.append(bad)
            bad=copy.deepcopy(screens);bad[0]['status']='partial';bad_screens.append(bad)
            bad=copy.deepcopy(screens);bad[0]['peak_lr']=.0001;bad_screens.append(bad)
            for bad in bad_screens:reject(lambda bad=bad:select(bad))
            positive=copy.deepcopy(panel)
            for pair in positive:
                pair['prefix8192']=synthetic(plans['prefix8192'],targets['prefix8192'],[1.6+.08*math.sin(i/2047) for i in range(TARGET_BYTES)])
            require(not review(positive,600443)['quality_gates_pass'],'Losing synthetic candidate must not pass')
            for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Bound source changed')
        publish(E/'statistics-contracts001.json',dict(status='passed_independent_common_byte_grouping_selection_and_bootstrap_contracts',source=reference(__file__),sources=pins,total_source_bindings=995,environment=env,nonconstant_fixture=True,independent_maximum_upper_difference=maximum,independent_upper_bounds=oracle,positive_and_negative_gate_examples_checked=True,invalid_panels_rejected=len(bad_panel),invalid_screen_sets_rejected=len(bad_screens),exact_lower_lr_ties_checked=True,
            training=False,model_scoring=False,scientific_seed_used=False,holdout_encoded_or_scored=False,weight_files=0,next_action='Implement additive full1200 quality worker/controller/all-six-fresh inference/independent final review with this common-byte statistical contract;perform full finite source/seed/capacity gates then execute under existing human authorization. No quality panel frozen yet.'))
        print(json.dumps(dict(status='passed_independent_common_byte_grouping_selection_and_bootstrap_contracts',maximum_upper_difference=maximum,invalid_panels=len(bad_panel),invalid_screen_sets=len(bad_screens),total_sources=995)),flush=True)
    except Exception as exc:publish(E/'STOP-statistics-contracts001',dict(error=repr(exc),closed=True,no_retry=True));raise
