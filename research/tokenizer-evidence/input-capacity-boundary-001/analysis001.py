"""Train-only representation geometry and existing curves; no model scoring/training."""
import gc,json,math,statistics
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_worker_v2.execution import checked
from research.embedding_input_dropout_worker_v2.validation import verify_artifact
from research.embedding_input_dropout_resume_v1.contract import config
from research.embedding_input_dropout_worker_v1.data import load
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/embedding-input-dropout-preparation-001';RANKS=(64,128,192,256)
def gram(x,w):
 require(x.ndim==2 and w.ndim==1 and len(w)==len(x) and bool(torch.isfinite(x).all()) and bool((w>0).all()),'Finite positive weighted observations required')
 w=w/w.sum();mean=(x*w[:,None]).sum(0);z=x-mean;return z.T@(z*w[:,None])
def describe(g):
 require(g.shape==(256,256) and g.dtype==torch.float64 and bool(torch.isfinite(g).all()),'Exact finite covariance required')
 eig=torch.linalg.eigvalsh(g);tol=max(float(g.trace()),1.)*1e-12;require(float(eig.min())>=-tol,'Covariance not positive semidefinite');eig=eig.clamp_min(0).flip(0);total=float(eig.sum());require(total>0 and abs(total-float(g.trace()))<1e-9,'Trace identity differs')
 require(abs(float((eig*eig).sum())-float((g*g).sum()))<1e-8,'Frobenius identity differs')
 p=eig/total;positive=p[p>0];return dict(trace=total,effective_rank_entropy=float(torch.exp(-(positive*positive.log()).sum())),participation_rank=float(1/(p*p).sum()),best_rank_covariance_energy={str(k):float(eig[:k].sum()/total) for k in RANKS},tail_energy_interpretation='Best centered squared reconstruction energy on this train-only covariance;not validation loss,model equivalence,trainability or causal regularization evidence.')
def publish(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment();require(not(E/'result.json').exists(),'New bounded analysis only')
 with CpuPrerequisiteGuard(E/'guard.json',ROOT,timeout_seconds=300,poll_seconds=.25):
  closure=checked(reference(B/'quality001-closure001.json'));require(closure['status']=='CLOSED_NOT_ADOPTED' and closure['all_972_bindings_unchanged'],'Closed actual panel required')
  # Independent tiny nonconstant arithmetic oracle precedes actual covariance use.
  x=torch.tensor([[math.sin(i+j*.7)+i*j*.11 for j in range(4)] for i in range(7)],dtype=torch.float64);w=torch.arange(1,8,dtype=torch.float64);m=[sum(float(w[i])*float(x[i,j]) for i in range(7))/float(w.sum()) for j in range(4)];oracle=torch.tensor([[sum(float(w[i])*(float(x[i,j])-m[j])*(float(x[i,k])-m[k]) for i in range(7))/float(w.sum()) for k in range(4)] for j in range(4)],dtype=torch.float64);torch.testing.assert_close(gram(x,w),oracle,rtol=0,atol=1e-12);del x,w,oracle
  view=load(config(0.,steps=4,seed=600417),'cpu');counts=torch.bincount(view.train_data,minlength=50257);identity=view.data_identity;del view
  ids=torch.nonzero(counts>0).flatten();freq=counts[ids].to(torch.float64);require(len(ids)==10653 and int(counts.sum())==267688,'Original train census differs')
  rows=[r for r in closure['trajectories'] if r['slot']>=6];require(len(rows)==6,'All fresh endpoints,not selected screens,required')
  protocol=dict(source=reference(__file__),closure=reference(B/'quality001-closure001.json'),environment=env,models=[r['worker'] for r in rows],scope='All three fresh control/candidate H200 pairs;train-only token census/weight geometry;existing immutable sampled curves. No quality seed,model construction,forward/scoring,holdout,training,weight output or source modification.',
   matrices=['transformer.wte.weight','lm_head.weight'],ranks=list(RANKS),covariances=['uniform_all_vocabulary','uniform_train_observed','train_token_frequency'],null='IIDGaussian on observed train IDs with same global matrix standard deviation;fixtureCPUgenerator713416 reset for every model. Same frequency weights expose concentration caused by Zipf counts alone.',
   prediction='If train-weighted input covariance concentrates but no more than the same-frequency Gaussian,that does not establish learned low-rank structure. Uniform observed covariance and output matrix provide contrary controls. Neither concentration nor parameter savings alone admits a quality campaign.',
   statistical_scope='Descriptive census of these six trained checkpoints. No intervals,rank/arm selection or pooled quality inference. Train-stream frequency is not exact random-window sampler exposure.',
   future_controls_required='Native incumbent plus full-rank factorized control before any low-rank candidate;check changed initialization,projection optimizer membership/width rates and equal tuning. Factorization is distinct from tying input/output tables. No rank or numerical panel admitted here.')
  publish(E/'plan.json',protocol);reports=[]
  for row in rows:
   accepted=checked(row['accepted']);worker=checked(row['worker']);ref=worker['h200'];verify_artifact(ref);saved=torch.load(ref['path'],map_location='cpu',weights_only=True);require(saved['kind']=='h200_embedding_input_dropout_v1','Original own-policy export required');state=saved['model_state'];require(sum(t.numel() for n,t in state.items() if not n.endswith(('freq_cos','freq_sin')))==28492416,'Learned inventory differs')
   matrices={}
   for name in protocol['matrices']:
    full=state[name].to(torch.float64);require(full.shape==(50257,256),'Original table shape required');observed=full[ids];one=torch.ones(len(full),dtype=torch.float64);null=torch.randn(len(ids),256,generator=torch.Generator().manual_seed(713416),dtype=torch.float64)*float(full.std(unbiased=False))
    matrices[name]=dict(uniform_all_vocabulary=describe(gram(full,one)),uniform_train_observed=describe(gram(observed,torch.ones(len(ids),dtype=torch.float64))),train_token_frequency=describe(gram(observed,freq)),same_frequency_gaussian=describe(gram(null,freq)),uniform_observed_gaussian=describe(gram(null,torch.ones(len(ids),dtype=torch.float64))))
    del full,observed,one,null;gc.collect()
   verify_artifact(ref);reports.append(dict(slot=row['slot'],arm=row['arm'],seed=row['seed'],h200=ref,matrices=matrices));del saved,state;gc.collect()
  curves={};stability={}
  for arm in ['input_zero','input_feature_p10']:
   group=[r for r in rows if r['arm']==arm];curves[arm]=[dict(seed=r['seed'],points=[{k:v for k,v in m.items() if k in ['step','targets','train_ce','validation_ce']} for m in r['metrics']]) for r in group]
   stability[arm]=dict(clipped_updates=[r['clipped_updates'] for r in group],training_seconds=[r['training_seconds'] for r in group],mean_last_saved_interval_sampled_validation_change=statistics.mean(r['metrics'][-1]['validation_ce']-r['metrics'][-2]['validation_ce'] for r in group))
  summaries={}
  for arm in ['input_zero','input_feature_p10']:
   group=[r for r in reports if r['arm']==arm];summaries[arm]={}
   for name in protocol['matrices']:
    summaries[arm][name]={scope:dict(mean_effective_rank=statistics.mean(r['matrices'][name][scope]['effective_rank_entropy'] for r in group),mean_rank128_energy=statistics.mean(r['matrices'][name][scope]['best_rank_covariance_energy']['128'] for r in group)) for scope in ['uniform_all_vocabulary','uniform_train_observed','train_token_frequency','same_frequency_gaussian','uniform_observed_gaussian']}
  capacity=[dict(rank=k,input_parameters=50257*k+k*256,total_parameters=28492416-50257*256+50257*k+k*256,untied_output_unchanged_parameters=50257*256,extra_lookup_projection_MACs_per_token=k*256) for k in RANKS]
  result=dict(status='completed_train_only_representation_capacity_analysis',plan=reference(E/'plan.json'),original_data=identity,train_tokens=267688,train_observed_ids=len(ids),census_is_not_sampler_exposure=True,fixture_seed=713416,
   reports=reports,summary=summaries,existing_curves=curves,existing_stability=stability,prospective_input_only_factorization_arithmetic=capacity,
   no_training=True,no_model_construction_or_scoring=True,no_new_scientific_seed=True,validation_rescored=False,holdout_encoded_or_scored=False,weight_outputs=0,
   caution='Mask-on sampled trainCE and mask-off validationCE are not a same-distribution generalization gap. Covariance/reconstruction and frequency correlations do not establish a quality benefit from training with a bottleneck. Existing table-sharing negative/control changes and primary-paper applicability gaps must be reviewed before admission.')
  publish(E/'result.json',result);print(json.dumps(dict(status=result['status'],summary=summaries,prospective_input_only_factorization_arithmetic=capacity,existing_stability=stability)),flush=True)
