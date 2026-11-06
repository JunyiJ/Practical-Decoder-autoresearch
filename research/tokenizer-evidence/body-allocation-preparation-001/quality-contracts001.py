"""Independent nonconstant statistical oracle and finite scientific contracts."""
import copy,json,math
from pathlib import Path
import numpy as np
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_quality_v1.contract import quality_sources,screen_specs,fresh_specs,select,weight_credit,FLOOR,TARGET
from research.body_allocation_quality_v1.statistics import review
from research.body_allocation_quality_v1.execution import independent_statistics
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
B=Path(__file__).resolve().parent;OUT=B/'quality-contracts001.json';torch.set_num_threads(4)
def rejected(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return
    raise AssertionError('Invalid finite scientific contract accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not OUT.exists(),'Closed contract fixture cannot rerun')
    with CpuPrerequisiteGuard(B/'quality-contracts001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        seeds=checked(reference(B/'census001/result.json'))['seeds'];screen=screen_specs(seeds)
        require(len(screen)==9 and [r['slot'] for r in screen]==list(range(9)),'Exact screen slots')
        rows=[dict(slot=s['slot'],arm=s['arm'],lr=s['config']['training']['learning_rate'],ce=4.5-float(s['slot']//3)*.01) for s in screen]
        choices,audits=select(rows);require(all(v==.0025 for v in choices.values()),'Independent minimum-LR-selection oracle')
        ties=[dict(r,ce=4.) for r in rows];require(all(v==.000625 for v in select(ties)[0].values()),'Lower-rate tie oracle')
        rejected(lambda:select(rows[:-1]));bad=copy.deepcopy(rows);bad[0]['ce']=float('nan');rejected(lambda:select(bad))
        fresh=fresh_specs(seeds,choices);require(len(fresh)==9 and [r['slot'] for r in fresh]==list(range(9,18)),'Exact fresh slots')
        require([r['arm'] for r in fresh]==['incumbent','shallow','deeper','shallow','deeper','incumbent','deeper','incumbent','shallow'],'Prospective rotated order')
        require([r['config']['training']['seed'] for r in fresh]==[s for s in seeds['fresh'] for _ in range(3)],'Exact shared fresh seeds')
        for spec in screen+fresh:
            cfg=spec['config'];require(cfg['training']['max_iters']==1200 and cfg['training']['lr_schedule']['start_step']==800
                and cfg['training']['weight_averaging']['interval']==10 and cfg['training']['grad_clip']==1.,'Full strict numerical horizon/recipe')
        require(weight_credit({'accepted.pt':120},{'accepted.pt':120})==120,'Actual accepted bytes credited once')
        rejected(lambda:weight_credit({'accepted.pt':120,'extra.pt':5},{'accepted.pt':120}))
        rejected(lambda:weight_credit({'accepted.pt':240},{'accepted.pt':120}))
        pilot=checked(reference(B/'pilot001-review.json'));target=checked(pilot['rows'][0]['accepted'])['acceptance']['full_validation']['target'];counts=target['block_counts']
        pairs=[]
        for pair in range(3):
            values={}
            for arm,shift in [('incumbent',.08),('shallow',.035),('deeper',0.)]:
                block=[float(c*(4.1+shift+.01*pair+.008*math.sin(i*1.7+pair)+(0.006*math.cos(i+pair) if arm=='deeper' else 0.))) for i,c in enumerate(counts)]
                total=sum(block);values[arm]=dict(target=copy.deepcopy(target),block_loss_sums=block,total_loss=total,ce=total/34278)
            pairs.append(values)
        result=review(pairs,713);independent_statistics(pairs,713,result);require(result['all_strict_gates_pass'],'Nonconstant positive-control statistical oracle')
        for control in ('incumbent','shallow'):
            expected=sum(p['deeper']['ce']-p[control]['ce'] for p in pairs)/3
            require(abs(expected-result['contrasts'][control]['mean_delta'])<1e-15,'Independent paired mean differs')
            require(len({round(b['upper'],12) for b in result['contrasts'][control]['conditional_text_upper_bounds']})>1,'Synthetic block bounds must be nonconstant')
        lose=copy.deepcopy(pairs)
        for p in lose:
            r=p['deeper'];r['block_loss_sums']=[v+.2*c for v,c in zip(r['block_loss_sums'],counts)];r['total_loss']=sum(r['block_loss_sums']);r['ce']=r['total_loss']/34278
        require(not review(lose,713)['all_strict_gates_pass'],'Negative-control rejection')
        rejected(lambda:review(pairs[:2],713));bad=copy.deepcopy(pairs);del bad[0]['shallow'];rejected(lambda:review(bad,713))
        bad=copy.deepcopy(pairs);bad[0]['deeper']['block_loss_sums'][0]=float('nan');rejected(lambda:review(bad,713))
        bad=copy.deepcopy(pairs);bad[1]['deeper']['target']['targets']=34277;rejected(lambda:review(bad,713))
        payload=dict(status='passed',quality_sources=quality_sources(),worker_sources=sources(),source=reference(__file__),
            exact_screen_configs=9,exact_fresh_configs=9,independent_nonconstant_bootstrap_atol=1e-12,
            minimum_and_tie_selection=True,extra_weight_credit_rejected=True,partial_missing_arm_nonfinite_wrong_target_rejected=True,
            required_free=FLOOR,cleanup_target=TARGET,scientific_seeds_used=False,bootstrap_fixture_seed=713,holdout_scored=False,
            scope='Configuration/statistical synthetic fixture only; no model construction, checkpoint loading, training or scientific bootstrap seed consumption.')
        with OUT.open('x') as f:json.dump(payload,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('quality_sources','worker_sources')}))
