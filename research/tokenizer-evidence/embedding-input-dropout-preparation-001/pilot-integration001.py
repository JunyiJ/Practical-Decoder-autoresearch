"""Actual generated RAW/H200 dependency audit and finite future-pilot contracts;no rescoring."""
import copy,gc,json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.embedding_input_dropout_v1.model import GPT
from research.embedding_input_dropout_resume_v1.contract import config
from research.embedding_input_dropout_resume_guard_v1.checkpoint import validate_resume
from research.embedding_input_dropout_worker_v2.execution import sources,checked
from research.embedding_input_dropout_worker_v2.validation import verify_artifact,replay_dropout_rng
from research.embedding_input_dropout_worker_v1.data import load
from research.embedding_input_dropout_campaign_v1.pilot import campaign_sources
from research.embedding_input_dropout_campaign_v1.rules import ARMS,PROBABILITIES,PAIR_CAP,RAW_CAP,H200_CAP,capacity,DESIGN
from research.final_ln_scale_worker_v1.validation import validate_report
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;torch.set_num_threads(4)
def rejected(fn):
    try:fn()
    except (TypeError,ValueError):return
    raise AssertionError('Forged raw dependency accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();out=E/'pilot-integration001.json';guard=E/'pilot-integration001-guard.json';require(not out.exists() and not guard.exists(),'Closed integration cannot retry')
    reviewref=reference(E/'worker002-review.json');review=checked(reviewref);require(review['status']=='two_generated_workers_independently_accepted','Both generated endpoints required')
    pinned=sources();campaign=campaign_sources();census=checked(reference(E/'census001/result.json'));require(census['status']=='reserved_unused','New actual seed census required')
    with StrictMpsGuard(guard,ROOT),torch.random.fork_rng(devices=[]):
        rows=[];forgeries=0
        for row,prob in zip(review['rows'],PROBABILITIES):
            a=row['acceptance'];w=checked(row['worker']);cfg=OmegaConf.create(w['config']);require(float(cfg.input_embedding_regularization.probability)==prob,'Exact generated arm route differs')
            for n,cap in (('parent',RAW_CAP),('h200',H200_CAP)):verify_artifact(w[n]);require(w[n]['size']<=cap,'Generated artifact exceeds original cap')
            torch.manual_seed(cfg.training.seed);torch.mps.manual_seed(cfg.training.seed);m=GPT(plain(cfg.input_embedding_regularization));initial_cpu=torch.get_rng_state();initial_mps=torch.mps.get_rng_state()
            o=build_optimizer(m,cfg);e=TailWeightAverages(m,cfg.training);raw=torch.load(w['parent']['path'],map_location='cpu',weights_only=True)
            require(validate_resume(m,o,e,raw,cfg)==4 and torch.equal(initial_cpu,raw['cpu_rng']) and torch.equal(replay_dropout_rng(cfg,initial_mps,4),raw['mps_rng']),'Actual full raw/CPU-MPS RNG contract differs')
            require(raw['ema']['snapshots']==3 and len(raw['ema']['parameter_names'])==88 and len(raw['clipping_history'])==4,'Complete generated EMA/history required')
            h=torch.load(w['h200']['path'],map_location='cpu',weights_only=True)
            require(h['kind']=='h200_embedding_input_dropout_v1' and h['cfg']==plain(cfg) and h['architecture']==m.signature and h['parent']==w['parent'] and h['model_state'].keys()==raw['model_state'].keys(),'Actual H200 provenance differs')
            for n,t in h['model_state'].items():require(torch.equal(t,raw['ema']['averages']['200'].get(n,raw['model_state'][n])),'H200 is not exact raw EMA '+n)
            data=load(cfg);validate_report(a['full_validation'],a['binding'],data)
            require(a['step']==4 and a['targets']==4096 and a['ema_snapshots']==3 and a['parameter_tensors']==88 and a['cpu_mps_ce_difference']<3e-5
                and [c['batch'] for c in a['cache']]==[1,8] and all(c['tokens']==128 and c['cache_bytes_per_item']==196608 for c in a['cache']),'Original independent full validation/cache acceptance differs')
            for n in ('worker_guard','parent_guard'):
                g=checked(row[n]);require(g['status']=='finished' and g['reason'] is None,'Actual original resource guard required')
            bad=dict(raw,architecture=dict(raw['architecture'],input_dropout=1.-prob));rejected(lambda:validate_resume(m,o,e,bad,cfg));forgeries+=1
            bad=dict(raw,clipping_history=raw['clipping_history'][:-1]);rejected(lambda:validate_resume(m,o,e,bad,cfg));forgeries+=1
            bad=dict(raw,ema=dict(raw['ema'],parameter_names=raw['ema']['parameter_names'][:-1]));rejected(lambda:validate_resume(m,o,e,bad,cfg));forgeries+=1
            bad=dict(raw,optimizer_state=dict(raw['optimizer_state'],optimizers=raw['optimizer_state']['optimizers'][:-1]));rejected(lambda:validate_resume(m,o,e,bad,cfg));forgeries+=1
            bad=dict(raw,mps_rng=raw['mps_rng'][:1]);rejected(lambda:validate_resume(m,o,e,bad,cfg));forgeries+=1
            rows.append(dict(arm=row['arm'],worker=row['worker'],accepted_step=4,targets=4096,original_acceptance=row,weights={n:w[n] for n in ('parent','h200')},raw_h200_optimizer_ema_rng_clipping_reaudited=True,original_full_validation_cache_reports_checked_without_rescoring=True))
            del m,o,e,raw,h,data,bad;gc.collect();torch.mps.empty_cache()
        require(capacity(0)['capacity_pass'] is False and capacity(2**40)['capacity_pass'] is True,'Capacity false/pass routing differs')
        rejected(lambda:capacity(2**40,2*PAIR_CAP+1));rejected(lambda:capacity(2**40,-1));forgeries+=2
        future=[]
        for prob in PROBABILITIES:
            for steps in (20,1200):
                for lr in (.000625,.00125,.0025):
                    cfg=config(prob,steps=steps,seed=census['seeds']['pilot'] if steps==20 else census['seeds']['fresh'][0],lr=lr)
                    require(cfg.training.batch_size==8 and cfg.model.block_size==128 and cfg.training.max_iters==steps and cfg.training.adamw_epsilon==1e-6,'Finite inherited future recipe differs')
                    future.append(dict(probability=prob,steps=steps,lr=lr))
        require(sources()==pinned and campaign_sources()==campaign,'Frozen input or new pilot sources changed')
    result=dict(status='passed_generated_dependencies_closed_for_finite_resource_pilots',source=reference(__file__),sources=pinned,campaign_sources=campaign,generated_review=reviewref,
        rows=rows,forgeries_rejected=forgeries,future_configurations=future,census=reference(E/'census001/result.json'),design=DESIGN,
        zero_training_updates=True,zero_model_rescoring=True,weight_copies=0,scientific_quality=False,
        generated_tensor_dependencies_closed=True,future_consumers='Only this immutable integration/acceptance metadata and a later exact retirement manifest. Pilot constructors/trainers/acceptance never load four-update payloads.',
        next='NEW exact4payload cross-root generated-retirement audit,then complete live finite-pilot pair capacity and preparation. No selected/fresh scientific weight is eligible.',guard=reference(guard))
    payload=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();require(len(payload)<=512*1024,'Bounded integration output')
    with out.open('xb') as f:f.write(payload)
    print(json.dumps(dict(status=result['status'],endpoints=len(rows),forgeries=forgeries,future_configs=len(future),new_updates=0)))
