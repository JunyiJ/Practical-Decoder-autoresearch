"""Full train-split fit decomposition on retained weights; no training/selection."""
import gc,hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_model_v1.model import GPT,load_weights
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_worker_v1.validation import verify_artifact
from research.body_allocation_worker_v1.data import load as data_view
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001';torch.set_num_threads(4)
def publish(p,d):
    payload=(json.dumps(d,indent=2,allow_nan=False)+'\n').encode();require(len(payload)<=512*1024,'Bounded metadata')
    with p.open('xb') as f:f.write(payload)
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not(E/'plan.json').exists(),'Closed diagnostic cannot retry')
    closure=checked(reference(B/'quality001-closure001.json'));rows=[r for r in closure['trajectories'] if r['slot']>=9 and r['arm']=='incumbent'];require(len(rows)==3,'All original incumbent models required')
    plan=dict(kind='train_split_fit_decomposition_v1',source=reference(__file__),sources=sources(),closure=reference(B/'quality001-closure001.json'),models=[r['accepted'] for r in rows],
        question='How well do retained incumbent models fit the known train stream, and which frequency/within-segment strata carry its loss? Read original validation128 CE only as a descriptive reference.',
        controls='All267687train next-token targets once per fixed model,128causal resets including tail. No val rescoring,holdout,training,seed,tuning,bootstrap,model/serving selection or adoption.',
        strata='Train-target token occurrence counts1-5/6-50/51-500/501+,and within-segment offsets0-31/32-63/64-95/96-127. Fixed before execution,all strata including empty groups reported.',
        resources='Three serial retained H200 models;zero weights output;600s/RSS4GiB/MPS6GiB/reserve8GiB,512KiB per plan/result. The closed body capacity floor is not reused.',
        inference_limits='Train-versus-validation content differs;this gap does not alone prove overfitting,an optimal horizon or a causal bottleneck. No automatic duration/regularization experiment.')
    publish(E/'plan.json',plan);reports=[];started=time.monotonic()
    try:
        with StrictMpsGuard(E/'guard.json',ROOT),torch.random.fork_rng(devices=[]),torch.no_grad():
            for row in rows:
                accepted=checked(row['accepted']);w=checked(accepted['worker']);ref=w['h200'];verify_artifact(ref);cfg=OmegaConf.create(w['config'])
                m=GPT(cfg.model).to('mps').eval();saved=torch.load(ref['path'],map_location='cpu',weights_only=True);load_weights(m,saved);del saved
                data=data_view(cfg,'mps');tokens=data.train_data;require(tokens.numel()==267688,'Original full train stream required')
                freq=torch.bincount(tokens,minlength=50257);target_freq=freq[tokens[1:]];chunks=[]
                for start in range(0,len(tokens)-1,128):
                    n=min(128,len(tokens)-1-start);x=tokens[start:start+n].unsqueeze(0).to('mps');y=tokens[start+1:start+n+1].to('mps')
                    logits,_=m(x);losses=F.cross_entropy(logits[0],y,reduction='none').cpu().to(torch.float64)
                    require(bool(torch.isfinite(losses).all()),'Finite train losses required');chunks.append(losses);require(time.monotonic()-started<600,'Diagnostic time cap')
                losses=torch.cat(chunks);require(losses.numel()==267687,'Complete train targets required');groups=[]
                for lo,hi in ((1,5),(6,50),(51,500),(501,267688)):
                    mask=(target_freq>=lo)&(target_freq<=hi);n=int(mask.sum());groups.append(dict(min_occurrences=lo,max_occurrences=hi,targets=n,total_loss=float(losses[mask].sum()),ce=float(losses[mask].mean()) if n else None))
                positions=torch.arange(len(losses))%128;position_groups=[]
                for lo in (0,32,64,96):
                    mask=(positions>=lo)&(positions<lo+32);position_groups.append(dict(first_offset=lo,last_offset=lo+31,targets=int(mask.sum()),ce=float(losses[mask].mean())))
                require(sum(r['targets'] for r in groups)==sum(r['targets'] for r in position_groups)==267687,'Strata coverage differs')
                ce=float(losses.mean());val=accepted['acceptance']['full_validation']['ce']
                reports.append(dict(seed=cfg.training.seed,accepted=row['accepted'],h200=ref,train_ids_sha256=data.data_identity['train_ids_sha256'],train_targets=267687,
                    train_target_ids_sha256=hashlib.sha256(tokens[1:].numpy().tobytes()).hexdigest(),ce=ce,recorded_validation128_ce=val,validation_minus_train_ce=val-ce,
                    frequency_strata=groups,within_segment_strata=position_groups))
                del m,data,logits,x,y,losses,chunks;gc.collect();torch.mps.empty_cache();verify_artifact(ref)
            require(sources()==plan['sources'],'Frozen sources changed')
        result=dict(status='completed_train_split_fit_diagnostic',plan=reference(E/'plan.json'),reports=reports,
            mean_train_ce=sum(r['ce'] for r in reports)/3,mean_recorded_validation128_ce=sum(r['recorded_validation128_ce'] for r in reports)/3,
            mean_validation_minus_train_ce=sum(r['validation_minus_train_ce'] for r in reports)/3,
            training=False,adopted=False,new_scientific_seed=False,validation_rescored=False,holdout_scored=False,guard=reference(E/'guard.json'))
        publish(E/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('reports','plan','guard')}))
    except Exception as exc:publish(E/'STOP',dict(error=repr(exc),closed=True));raise
