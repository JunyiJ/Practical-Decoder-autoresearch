"""Matched-initial input motion and explicitly derived inference counterfactuals.

No original checkpoint is modified. Derived models are not original H200 results,
quality confirmations, exported models or adoption candidates. No caches are used.
"""
import gc,json,math,statistics
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,plain,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_v1.model import GPT,load_weights
from research.embedding_input_dropout_worker_v2.execution import checked,Guard
from research.embedding_input_dropout_worker_v2.validation import verify_artifact
from research.embedding_input_dropout_worker_v2.scoring import score
from research.embedding_input_dropout_worker_v1.data import load
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/embedding-input-dropout-preparation-001'
def initial(policy,seed):
 before=torch.random.get_rng_state().clone()
 with torch.random.fork_rng(devices=[]):
  # CPU default-generator seeding reproduces CPU construction in original run;
  # unlike torch.manual_seed this does not reseed other device generators.
  torch.random.default_generator.manual_seed(seed);model=GPT(policy).eval();tables={n:model.state_dict()[n].clone() for n in ['transformer.wte.weight','lm_head.weight']};del model
 require(torch.equal(before,torch.random.get_rng_state()),'CPU RNG changed');return tables

def motion(final,start,ids,counts):
 a=final.to(torch.float64);z=start.to(torch.float64);delta=(a-z).square().sum(1);norm=z.square().sum(1);f=counts[ids].to(torch.float64)
 return dict(relative_L2_all=float(torch.sqrt(delta.sum()/norm.sum())),relative_L2_observed=float(torch.sqrt(delta[ids].sum()/norm[ids].sum())),relative_L2_train_frequency=float(torch.sqrt((delta[ids]*f).sum()/(norm[ids]*f).sum())),initial_RMS=float(z.square().mean().sqrt()),final_RMS=float(a.square().mean().sqrt()),mean_observed_row_cosine=float(torch.nn.functional.cosine_similarity(a[ids],z[ids],dim=1).mean()))
def publish(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment();require(not(E/'initial-counterfactual001.json').exists() and not(E/'initial-counterfactual001-plan.json').exists(),'One new bounded diagnostic only')
 with Guard(E/'initial-counterfactual001-guard.json',600):
  require(checked(reference(E/'result.json'))['status']=='completed_train_only_representation_capacity_analysis','Actual geometry evidence required');closure=checked(reference(B/'quality001-closure001.json'));require(closure['status']=='CLOSED_NOT_ADOPTED','Actual closed panel required');rows=[r for r in closure['trajectories'] if r['slot']>=6];require(len(rows)==6,'All fresh endpoint movement required')
  protocol=dict(policy='input_table_initialization_counterfactual_diagnostic_v1',source=reference(__file__),closure=reference(B/'quality001-closure001.json'),geometry=reference(E/'result.json'),environment=env,
   movement='All6fresh H200 input/readout matrices versus exact frozen CPU native initialization from their ALREADY USED original seeds;no repeated optimization. Relative change is not functional importance.',
   derived_inference='All3contemporaneous control H200s only;two predeclared in-memory input replacements:exact native initial table and independent native table from fixture713417. Body/readout remain exactly H200;no table rescaling/sweep/cache/checkpoint output.',
   baseline='Use immutable same-engine full34278targetH200 acceptance reports;do not rescore originals. Each derived result must have identical target identity/count/arithmetic. Existing CPU/MPS validation tolerance is context,not a new population confidence interval.',
   falsifier='If original-initial replacement loses materially,small parameter movement does not imply learning is unnecessary. A fresh random-ID table is a discriminating negative control:Gaussian covariance alone must not imply arbitrary word-code interchangeability.',
   scope='Derived inference on adaptively reused validation text;not fresh training,model adoption,holdout evaluation or a frozen input/factorization training claim. No next scientific seed is assigned.',fixture_seed=713417)
  publish(E/'initial-counterfactual001-plan.json',protocol);movements=[];counterfactuals=[]
  for row in rows:
   accepted=checked(row['accepted']);worker=checked(row['worker']);cfg=OmegaConf.create(worker['config']);policy=plain(cfg.input_embedding_regularization);original=worker['h200'];verify_artifact(original);start=initial(policy,row['seed']);saved=torch.load(original['path'],map_location='cpu',weights_only=True)
   data=load(cfg,'cpu');counts=torch.bincount(data.train_data,minlength=50257);ids=torch.nonzero(counts>0).flatten();require(len(ids)==10653,'Train census differs')
   movements.append(dict(arm=row['arm'],seed=row['seed'],h200=original,matrices={n:motion(saved['model_state'][n],start[n],ids,counts) for n in start}))
   if row['arm']=='input_zero':
    random_table=initial(policy,713417)['transformer.wte.weight'];baseline=accepted['acceptance']['full_validation']
    for label,table in [('restore_exact_initial_input',start['transformer.wte.weight']),('independent_native_random_input',random_table)]:
     with torch.random.fork_rng(devices=[]):model=GPT(policy).to('mps').eval()
     load_weights(model,saved)
     with torch.no_grad():model.transformer.wte.weight.copy_(table.to('mps'))
     model.validate();derived=score(model,data);require(derived['target']==baseline['target'],'Derived target identity differs');require(abs(sum(derived['block_loss_sums'])-derived['total_loss'])<1e-8,'Derived aggregation differs')
     counterfactuals.append(dict(derived_policy=protocol['policy'],derivation=label,seed=row['seed'],original_h200=original,original_acceptance=row['accepted'],baseline_ce=baseline['ce'],derived_full_validation=derived,delta_from_saved_baseline=derived['ce']-baseline['ce'],original_checkpoint_modified=False,exported_model=False));del model;gc.collect();torch.mps.empty_cache()
    del random_table
   verify_artifact(original);del saved,start,data,counts,ids;gc.collect()
  means={label:statistics.mean(r['delta_from_saved_baseline'] for r in counterfactuals if r['derivation']==label) for label in ['restore_exact_initial_input','independent_native_random_input']}
  result=dict(status='completed_explicit_input_initialization_counterfactuals',plan=reference(E/'initial-counterfactual001-plan.json'),movement=movements,counterfactuals=counterfactuals,mean_derived_deltas=means,
   no_training=True,no_new_scientific_seed=True,original_h200_files_unchanged=True,weight_outputs=0,holdout_encoded_or_scored=False,original_validation_rescored=False,
   interpretation='Single-site inference perturbations do not estimate training with frozen embeddings or a lower rank. Random-ID control distinguishes code identity from covariance. No adoption or quality-panel pooling.')
  publish(E/'initial-counterfactual001.json',result);print(json.dumps(dict(status=result['status'],mean_derived_deltas=means,movement=movements)),flush=True)
