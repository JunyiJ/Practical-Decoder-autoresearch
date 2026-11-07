"""NEW literal full-config/selection/conditional-statistics proof; no model/data calls."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.relu_squared_quality_v1 import contract as Q
from research.relu_squared_quality_v1 import statistics as T
from research.relu_squared_quality_v1.final_review import verify
from research.relu_squared_quality_v1.training import validate_config
from research.relu_squared_quality_v1.benchmark import measure
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock

R=Path(__file__).with_suffix('')

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(name,v):
    raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=8*1024**2,'Metadata cap')
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def np_state():
    v=np.random.get_state();return hashlib.sha256(v[0].encode()+v[1].tobytes()+repr(v[2:]).encode()).hexdigest()

def report(target,level,localized=False):
    sums=[]
    for start in range(0,34278,128):
        count=min(128,34278-start);value=level+(.25 if start<8192 else -.125) if localized else level
        sums.append(value*count)
    total=math.fsum(sums)
    return dict(target=copy.deepcopy(target),window_loss_sums=sums,block_loss_sums=[math.fsum(sums[i:i+16]) for i in range(0,len(sums),16)],
        total_loss=total,ce=total/34278,bits_per_byte=total/math.log(2)/111536,
        loss_dtype='FP32_own_distribution_NLL_then_CPU_FP64_aggregation',holdout_scored=False,aux_loss_included=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    require(sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json','source-review.json'},'One exact original new config/statistics plan required')
    plan=json.loads((R/'plan.json').read_text());census=Q.checked(plan['census'])
    require(census['status']=='new_full_horizon_fixture_roles_unused_before_construction' and not census['prior_matches'],'New25fixture-role census required')
    for p,h in plan['sources'].items():require(sha(ROOT/p)==h,'Frozen source changed')
    torch.set_num_threads(4);Q.environment()
    with session_lock(GLOBAL_LOCK.parent):
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                cpu=torch.get_rng_state().clone();numpy=np_state();roles=census['roles'];saved=copy.deepcopy(roles);Q.roles_valid(roles)
                screens=Q.screen_specs(roles);require(len(screens)==9,'Complete new source screen configs');neg=[]
                def reject(name,call):
                    try:call()
                    except (ValueError,KeyError,TypeError):neg.append(name)
                    else:raise AssertionError('Invalid literal case accepted: '+name)
                for row in screens:
                    from omegaconf import OmegaConf
                    cfg=OmegaConf.create(row['config']);before=plain(cfg);require(validate_config(cfg)==1200,'Complete own scientific config rejected')
                    for case in ['horizon4','horizon20','horizon1199','ema_interval','ema_start','clip','lr','binding','legacy','global_bool']:
                        bad=copy.deepcopy(cfg)
                        if case.startswith('horizon'):bad.training.max_iters=int(case[7:])
                        elif case=='ema_interval':bad.training.weight_averaging.interval=1
                        elif case=='ema_start':bad.training.weight_averaging.start_step=799
                        elif case=='clip':bad.training.grad_clip=.5
                        elif case=='lr':bad.training.learning_rate=.001875
                        elif case=='binding':bad.relu_squared_quality_execution.kind='foreign'
                        elif case=='legacy':bad.relu_squared_execution={}
                        else:bad.rng_identity.global_cpu=True
                        reject('config_'+case,lambda:validate_config(bad))
                    require(before==plain(cfg),'Caller config changed')
                target=Q.checked(plan['physical_target_receipt'])['rows'][0]['acceptance']['full_validation']['raw']['cpu']['target']
                levels={'native_swiglu':[4.03125,4.,4.015625],'ungated_silu':[4.125,4.125,4.15625],'ungated_relu_squared':[4.25,4.234375,4.21875]}
                records=[dict(status='accepted',arm=a,peak_lr=lr,h200=report(target,levels[a][i])) for i,lr in enumerate(T.LRS) for a in T.ARMS]
                choice=T.select(records);require(choice==dict(native_swiglu=.00125,ungated_silu=.000625,ungated_relu_squared=.0025),'Independent finite choice/tie differs')
                for case in ['missing','duplicate','forbidden_lr','foreign_arm','status','nonfinite','foreign_target']:
                    bad=copy.deepcopy(records)
                    if case=='missing':bad.pop()
                    elif case=='duplicate':bad[-1]=copy.deepcopy(bad[0])
                    elif case=='forbidden_lr':bad[0]['peak_lr']=.001875
                    elif case=='foreign_arm':bad[0]['arm']='foreign'
                    elif case=='status':bad[0]['status']='partial'
                    elif case=='nonfinite':bad[0]['h200']['total_loss']=float('nan')
                    else:bad[0]['h200']['target']['policy']='foreign'
                    reject('selection_'+case,lambda:T.select(bad))
                fresh=Q.fresh_specs(roles,choice);require(len(fresh)==9 and [r['arm'] for r in fresh]==[a for p in range(3) for a in T.ARMS[p:]+T.ARMS[:p]],'Fresh role/order/choice differs')
                for row in fresh:require(validate_config(OmegaConf.create(row['config']))==1200,'Fresh1200 config rejected')
                panels=[]
                for scenario in ['strong','one_control_only','one_seed_worse','small_margin','upper_failure','tie']:
                    panel=[]
                    for i,cohort in enumerate(roles['fresh']):
                        native=4.;silu=4.015625;candidate=3.9375;localized=False
                        if scenario=='one_control_only':silu=3.875
                        elif scenario=='one_seed_worse':candidate=4.03125 if i==2 else 3.875
                        elif scenario=='small_margin':candidate=3.9921875
                        elif scenario=='upper_failure':candidate=4.;localized=True
                        elif scenario=='tie':candidate=4.
                        panel.append(dict(seed=cohort['initializer'],native_swiglu=report(target,native),ungated_silu=report(target,silu),ungated_relu_squared=report(target,candidate,localized)))
                    primary=T.review(panel,roles['bootstrap']);independent=verify(panel,[c['initializer'] for c in roles['fresh']],roles['bootstrap'],primary)
                    require(primary['quality_gates_pass']==(scenario=='strong'),'Literal joint quality gate differs: '+scenario)
                    if scenario=='one_seed_worse':require(primary['gates']['both_mean_ce_below_minus_point02'] and not primary['gates']['every_paired_seed_delta_negative'],'All-seed discriminant missing')
                    if scenario=='upper_failure':require(primary['gates']['both_mean_ce_below_minus_point02'] and primary['gates']['every_paired_seed_delta_negative'] and not primary['gates']['all_six_simultaneous_conditional_text_upper_bounds_negative'],'All-upper discriminant missing')
                    panels.append(dict(scenario=scenario,primary=primary,independent=independent))
                good=panel
                reject('incomplete_panel',lambda:T.review(good[:2],roles['bootstrap']))
                reject('bool_bootstrap',lambda:T.review(good,True))
                bad=copy.deepcopy(good);bad[1]['seed']=bad[0]['seed'];reject('duplicate_panel_seed',lambda:T.review(bad,roles['bootstrap']))
                # Invalid serving lists must fail BEFORE MPS/RNG/model entry.
                reject('serving_missing',lambda:measure([]))
                reject('serving_incomplete',lambda:measure([{}]*8))
                require(roles==saved and torch.equal(cpu,torch.get_rng_state()) and np_state()==numpy,'Caller roles/CPU/NumPy RNG changed')
                for p,h in plan['sources'].items():require(sha(ROOT/p)==h,'Source changed')
                publish('result.json',dict(status='passed_new_full_horizon_config_selection_statistics_fixture',valid_configs=18,negative_cases=neg,choice=choice,panels=panels,
                    synthetic_loss_metadata_only=True,physical_target_metadata_reference=plan['physical_target_receipt'],no_model_optimizer_corpus_scoring_checkpoint_serialization_calls=True,
                    bootstrap_fixture_role=roles['bootstrap'],caller_cpu_numpy_rng_unchanged=True,source_pins_unchanged=True,
                    serving_scope='Only incomplete-input rejection before any MPS/model call;not full nine-serving success',runtime_sources=Q.sources(),scientific_roles_assigned=False))
            publish('STOP',dict(status='complete_new_full_config_selection_statistics_fixture',no_replay=True,no_retry=True))
        except BaseException as exc:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
            raise

if __name__=='__main__':main()
