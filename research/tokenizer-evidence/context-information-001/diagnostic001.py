"""New fixed-weight conditioning diagnostic; no old command or native128 replay."""
import gc,json,math,time
from pathlib import Path
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_model_v1.model import GPT,load_weights
from research.body_allocation_worker_v1.data import load as data_view
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_worker_v1.validation import verify_artifact
from research.final_ln_scale_worker_v1.validation import target_contract
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001'
torch.set_num_threads(4)
def publish(path,value,limit=512*1024):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=limit,'Bounded diagnostic output exceeded')
    with path.open('xb') as f:f.write(raw)
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not (E/'plan.json').exists() and not (E/'result.json').exists(),'Closed diagnostic cannot retry')
    closure=checked(reference(B/'quality001-closure001.json'));require(closure['status']=='CLOSED_NOT_ADOPTED','Complete original panel required')
    rows=[r for r in closure['trajectories'] if r['slot']>=9 and r['arm']=='incumbent'];require(len(rows)==3,'All three already-trained incumbent fresh models required')
    plan=dict(kind='fixed_weight_context_information_diagnostic_v1',source=reference(__file__),closure=reference(B/'quality001-closure001.json'),sources=sources(),
        models=[r['accepted'] for r in rows],conditioning_windows=[32,64],reference_window=128,
        reference_source='Immutable original full-validation128 scores; never rerun original closed inference/final commands or rescore128.',
        hypothesis='Do the existing incumbent weights gain predictive information between32,64 and128 conditioning tokens? This does not prove a benefit from training/evaluating at256.',
        selection=False,adoption_permitted=False,training=False,new_scientific_seeds=False,holdout_scored=False,output_cap=512*1024,
        full_resource_budget='Three serial existing H200 models, two changed conditioning passes each, no new weight/checkpoint output;512KiB result+metadata,600sec/RSS4GiB/MPS6GiB/reserve8GiB.',
        conditioning_scope='Same34278 validation targets and same learned weights. Different causal context resets, so CE changes are information/conditioning diagnostics, not a new trained-model quality comparison.')
    publish(E/'plan.json',plan);results=[];started=time.monotonic()
    try:
        with StrictMpsGuard(E/'guard.json',ROOT),torch.random.fork_rng(devices=[]):
            for row in rows:
                a=checked(row['accepted']);worker=checked(a['worker']);cfg=OmegaConf.create(worker['config']);require(cfg.model.body_allocation_arm=='incumbent','Original incumbent required')
                verify_artifact(worker['h200']);model=GPT(cfg.model).to('mps').eval();saved=torch.load(worker['h200']['path'],map_location='cpu',weights_only=True);load_weights(model,saved);del saved
                data=data_view(cfg,'mps');identity=target_contract(data);reference_report=a['acceptance']['full_validation'];require(reference_report['target']==identity,'Original fixed validation identity differs')
                reports=[]
                for context in (32,64):
                    chunks=[];counts=[]
                    with torch.no_grad():
                        for start in range(0,len(data.val_data)-1,context):
                            count=min(context,len(data.val_data)-1-start);x=data.val_data[start:start+count].unsqueeze(0).to('mps');y=data.val_data[start+1:start+count+1].to('mps')
                            logits,_=model(x);require(logits.shape==(1,count,50257),'Exact original output layout')
                            losses=F.cross_entropy(logits[0],y,reduction='none').to('cpu').to(torch.float64);require(bool(torch.isfinite(losses).all()),'Nonfinite conditioning loss')
                            chunks.append(losses);counts.append(count);require(time.monotonic()-started<600,'Bounded inference diagnostic timed out')
                    losses=torch.cat(chunks);require(len(losses)==34278 and sum(counts)==34278,'Full identical target coverage required')
                    total=float(losses.sum());reports.append(dict(conditioning_window=context,targets=34278,tail_retained=True,window_counts=counts,
                        block_loss_sums=[float(losses[i:i+2048].sum()) for i in range(0,34278,2048)],total_loss=total,ce=total/34278,reference128_ce=reference_report['ce'],
                        diagnostic_ce_minus_reference128=total/34278-reference_report['ce']))
                results.append(dict(seed=cfg.training.seed,original_accepted=row['accepted'],h200=worker['h200'],target=identity,reports=reports,reference128_score_source=a['acceptance']['binding']))
                del model,data,losses,chunks,logits,x,y;gc.collect();torch.mps.empty_cache();verify_artifact(worker['h200'])
            require(sources()==plan['sources'],'Frozen source changed during diagnostic')
        means={str(c):sum(next(v for v in r['reports'] if v['conditioning_window']==c)['ce'] for r in results)/3 for c in (32,64)}
        baseline=sum(r['reports'][0]['reference128_ce'] for r in results)/3
        result=dict(status='completed_fixed_weight_conditioning_diagnostic',plan=reference(E/'plan.json'),rows=results,mean_ce={**means,'128_original_reference':baseline},
            mean_gain32_to64=means['32']-means['64'],mean_gain64_to128=means['64']-baseline,training=False,new_scientific_seeds=False,adoption=False,holdout_scored=False,
            interpretation='Changed conditioning on already-trained fixed incumbent weights, adaptively reused validation. Positive gains motivate a controlled future context study, not a256-context gain claim or fresh confirmation. No new model selected.',
            guard=reference(E/'guard.json'))
        publish(E/'result.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ('rows','plan','guard')}))
    except Exception as exc:
        publish(E/'STOP',dict(error=repr(exc),no_retry=True));raise
