"""Actual pilot dependency closure plus independent finite selection/retention/statistics contracts."""
import copy,gc,json,math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.embedding_input_dropout_v1.model import GPT
from research.embedding_input_dropout_resume_guard_v1.checkpoint import validate_resume
from research.embedding_input_dropout_worker_v2.execution import sources,checked
from research.embedding_input_dropout_worker_v2.validation import verify_artifact,replay_dropout_rng
from research.embedding_input_dropout_worker_v1.data import load
from research.embedding_input_dropout_campaign_v1.pilot import campaign_sources
from research.embedding_input_dropout_resume_v1.contract import config
from research.embedding_input_dropout_quality_v1.contract import B,RUN,PLAN,FREEZE,FLOOR,TARGET,ENTRY,quality_sources,screen_specs,fresh_specs,select,weight_credit,ARMS,PAIR_CAP
from research.embedding_input_dropout_quality_v1.statistics import review
from research.embedding_input_dropout_quality_v1.execution import independent_statistics,remaining
from research.embedding_input_dropout_quality_v1.retention import retirement_scope
from research.final_ln_scale_worker_v1.validation import validate_report
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
OUT=B/'quality-contracts001.json';torch.set_num_threads(4)
def rejected(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return
    raise AssertionError('Invalid finite contract accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not OUT.exists() and not(B/'quality-contracts001-guard.json').exists(),'Closed finite contracts cannot rerun');q=quality_sources();w=sources();p=campaign_sources()
    with StrictMpsGuard(B/'quality-contracts001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
        census=checked(reference(B/'census001/result.json'));seeds=census['seeds'];screen=screen_specs(seeds);require(len(screen)==6 and [s['slot'] for s in screen]==list(range(6)),'Exact six screen slots')
        rows=[dict(slot=s['slot'],arm=s['arm'],lr=s['config']['training']['learning_rate'],ce=4.5-float(s['slot']//2)*.01) for s in screen]
        choice,audits=select(rows);require(all(v==.0025 for v in choice.values()),'Independent minimum selection oracle')
        require(all(v==.000625 for v in select([dict(r,ce=4.) for r in rows])[0].values()),'Lower-LR tie oracle')
        rejected(lambda:select(rows[:-1]));bad=copy.deepcopy(rows);bad[0]['ce']=float('nan');rejected(lambda:select(bad))
        fresh=fresh_specs(seeds,choice);require(len(fresh)==6 and [s['slot'] for s in fresh]==list(range(6,12)),'Complete three fresh pairs')
        require([s['arm'] for s in fresh]==[ARMS[0],ARMS[1],ARMS[1],ARMS[0],ARMS[0],ARMS[1]],'Frozen paired rotated order')
        require([s['config']['training']['seed'] for s in fresh]==[seed for seed in seeds['fresh'] for _ in range(2)],'Exact shared fresh seeds')
        for spec in screen+fresh:
            cfg=spec['config'];require(cfg['training']['max_iters']==1200 and cfg['training']['lr_schedule']['start_step']==800 and cfg['training']['weight_averaging']['interval']==10
                and cfg['training']['grad_clip']==1. and cfg['training']['adamw_epsilon']==1e-6 and cfg['model']['dim']==256 and spec['weight_cap']==PAIR_CAP,'Original fixed numerical recipe/caps differ')
        require(weight_credit({'accepted.pt':120},{'accepted.pt':120})==120,'Actual bytes credited once')
        rejected(lambda:weight_credit({'accepted.pt':120,'extra.pt':1},{'accepted.pt':120}));rejected(lambda:weight_credit({'accepted.pt':240},{'accepted.pt':120}))
        pilot_paths=[dict(path=str(ROOT/f'research_runs/tokenizer-embedding-input-dropout-pilot-v1-001/evidence/pilots/{i:02d}/checkpoints/{n}')) for i in range(2) for n in ('parent.pt','h200.pt')]
        retirement_scope(pilot_paths,'pilots');rejected(lambda:retirement_scope(pilot_paths[:-1],'pilots'))
        selected=[str(RUN/f'evidence/quality/{i:02d}/checkpoints/{n}') for i in (2,3) for n in ('parent.pt','h200.pt')]
        losers=[dict(path=str(RUN/f'evidence/quality/{i:02d}/checkpoints/{n}')) for i in (0,1,4,5) for n in ('parent.pt','h200.pt')]
        retirement_scope(losers,'losing_screens',selected);bad=copy.deepcopy(losers);bad[0]['path']=selected[0];rejected(lambda:retirement_scope(bad,'losing_screens',selected))
        bad=copy.deepcopy(losers);bad[0]['path']=str(RUN/'evidence/quality/06/checkpoints/parent.pt');rejected(lambda:retirement_scope(bad,'losing_screens',selected))
        rejected(lambda:retirement_scope(losers,'losing_screens',selected[:2]));bad=copy.deepcopy(losers);bad[1]=bad[0];rejected(lambda:retirement_scope(bad,'losing_screens',selected))
        # Model-free full-horizon mask replay guard,not a scientific training run.
        long_rng=[]
        for prob in (0.,.1):
            cfg=config(prob,steps=1200,seed=600410);torch.mps.manual_seed(600410);initial=torch.mps.get_rng_state().clone();expected=replay_dropout_rng(cfg,initial,1200)
            require(torch.equal(torch.mps.get_rng_state(),initial),'Parent RNG replay leaked state')
            with torch.no_grad():
                ones=torch.ones((8,128,256),device='mps',dtype=torch.float32)
                for _ in range(1200):unused=F.dropout(ones,p=prob,training=True)
                actual=torch.mps.get_rng_state().clone();del ones,unused
            require(torch.equal(expected,actual),'Independent1200-mask RNG oracle differs');long_rng.append(dict(probability=prob,steps=1200,exact=True));torch.mps.empty_cache()
        pilot=checked(reference(B/'pilot001-review.json'));require(pilot['status']=='two_resource_pilots_passed','Actual complete resource pair required');pilot_rows=[];target=None
        for rr in pilot['rows']:
            row=checked(rr['accepted']);worker=checked(row['worker']);cfg=OmegaConf.create(worker['config'])
            for n in ('parent','h200'):verify_artifact(worker[n])
            m=GPT(plain(cfg.input_embedding_regularization));o=build_optimizer(m,cfg);e=TailWeightAverages(m,cfg.training);raw=torch.load(worker['parent']['path'],map_location='cpu',weights_only=True)
            require(validate_resume(m,o,e,raw,cfg)==20 and raw['ema']['snapshots']==8 and len(raw['ema']['parameter_names'])==88 and len(raw['clipping_history'])==20,'Actual pilot raw/optimizer/EMA/RNG/history coverage differs')
            h=torch.load(worker['h200']['path'],map_location='cpu',weights_only=True);require(h['kind']=='h200_embedding_input_dropout_v1' and h['parent']==worker['parent'] and h['architecture']==m.signature,'Actual pilot H200 binding differs')
            for n,t in h['model_state'].items():require(torch.equal(t,raw['ema']['averages']['200'].get(n,raw['model_state'][n])),'Pilot H200/rawEMA mismatch')
            data=load(cfg);a=row['acceptance'];validate_report(a['full_validation'],a['binding'],data);require(a['cpu_mps_ce_difference']<3e-5 and [c['batch'] for c in a['cache']]==[1,8] and all(c['cache_bytes_per_item']==196608 for c in a['cache']),'Original pilot full validation/cache acceptance differs')
            for n in ('worker_guard','parent_guard'):
                g=checked(row[n]);require(g['status']=='finished' and g['reason'] is None,'Original actual pilot guard required')
            target=a['full_validation']['target'];pilot_rows.append(dict(accepted=rr['accepted'],worker=row['worker'],weights={n:worker[n] for n in ('parent','h200')},raw_h200_optimizer_ema_rng_clipping_checked=True,original_full_validation_cache_reports_checked_without_rescoring=True))
            del m,o,e,raw,h,data;gc.collect();torch.mps.empty_cache()
        counts=target['block_counts'];pairs=[]
        for i in range(3):
            values={}
            for arm,shift in [(ARMS[0],.06),(ARMS[1],0.)]:
                block=[float(c*(4.1+shift+.01*i+.008*math.sin(j*1.7+i)+(.006*math.cos(j+i) if arm==ARMS[1] else 0.))) for j,c in enumerate(counts)];total=sum(block)
                values[arm]=dict(target=copy.deepcopy(target),block_loss_sums=block,total_loss=total,ce=total/34278)
            pairs.append(values)
        stats=review(pairs,713);independent_statistics(pairs,713,stats);require(stats['all_strict_gates_pass'],'Nonconstant statistical positive-control failed')
        expected=sum(v[ARMS[1]]['ce']-v[ARMS[0]]['ce'] for v in pairs)/3;require(abs(expected-stats['contrasts'][ARMS[0]]['mean_delta'])<1e-15,'Independent paired mean differs')
        require(len({round(v['upper'],12) for v in stats['contrasts'][ARMS[0]]['conditional_text_upper_bounds']})>1,'Nonconstant block-width oracle required')
        bad=copy.deepcopy(pairs)
        for pair in bad:
            v=pair[ARMS[1]];v['block_loss_sums']=[x+.2*c for x,c in zip(v['block_loss_sums'],counts)];v['total_loss']=sum(v['block_loss_sums']);v['ce']=v['total_loss']/34278
        require(not review(bad,713)['all_strict_gates_pass'],'Negative statistical control failed')
        rejected(lambda:review(pairs[:2],713));bad=copy.deepcopy(pairs);del bad[0][ARMS[1]];rejected(lambda:review(bad,713))
        bad=copy.deepcopy(pairs);bad[0][ARMS[1]]['block_loss_sums'][0]=float('nan');rejected(lambda:review(bad,713))
        bad=copy.deepcopy(pairs);bad[1][ARMS[1]]['target']['targets']=34277;rejected(lambda:review(bad,713))
        # Enumerate the prospective high-water,without crediting any future deletion.
        stage_pair_counts=[2+i for i in range(7)]+[2+i for i in range(7)];require(max(stage_pair_counts)==8 and FLOOR==14892462080 and TARGET==17039945728,'Complete tested stage high-water arithmetic differs')
        require(not RUN.exists() and not PLAN.exists() and not FREEZE.exists(),'Scientific production root/freeze must remain absent')
        require(sources()==w and campaign_sources()==p and quality_sources()==q,'Frozen old/new source changed')
        result=dict(status='passed',source=reference(__file__),worker_sources=w,pilot_sources=p,quality_sources=q,total_unique_sources=len({**w,**p,**q}),
            exact_screen_configs=6,exact_fresh_configs=6,minimum_and_tie_selection=True,retention_scope_selected_fresh_duplicate_partial_rejection=True,
            stage_peak_pairs=8,prospective_peak_weight_caps=8*PAIR_CAP,required_free=FLOOR,cleanup_target=TARGET,pilot_dependencies_closed=True,pilot_rows=pilot_rows,
            no_training_or_model_rescoring=True,independent_nonconstant_bootstrap_atol=1e-12,bootstrap_fixture_seed=713,full_horizon_rng_replay=long_rng,
            quality_screen_fresh_bootstrap_seeds_consumed=False,holdout_scored=False,guard=reference(B/'quality-contracts001-guard.json'))
    with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('worker_sources','pilot_sources','quality_sources','pilot_rows','source','guard')}))
