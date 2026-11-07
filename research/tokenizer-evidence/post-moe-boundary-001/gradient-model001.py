"""Own fixed-weight native gradient integration, generated oracle and3real CPU/MPS pilot draws."""
import copy,gc,json,math,hashlib,shutil
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_model_v1.model import GPT,load_weights
from research.tokenizer_moe_model_v1.contract import model_config
from research.tokenizer_moe_resume_v1.contract import runtime as environment
from research.tokenizer_moe_worker_v1.execution import idle
from research.tokenizer_gradient_noise_v1.component import Moments
from research.tokenizer_gradient_probe_v1 import runtime as C
from research.tokenizer_gradient_probe_v1.data import load_train
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent

def publish(n,d):
 raw=(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=16*1024**2,'Bounded metadata proof output required')
 with (E/n).open('xb') as f:f.write(raw)
def reject(fn):
 try:fn()
 except (ValueError,RuntimeError,TypeError,KeyError):return
 raise AssertionError('Foreign/invalid diagnostic input accepted')
def compare(a,b):
 require(a.keys()==b.keys(),'Complete gradient keys differ');error=0.
 for n in a:
  torch.testing.assert_close(a[n].double(),b[n].double(),atol=4e-5,rtol=4e-4);error=max(error,float((a[n].double()-b[n].double()).abs().max()))
 return error

torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 idle();env=environment();require(not(E/'gradient-model001-plan.json').exists() and not(ROOT/'research_runs/tokenizer-gradient-noise-diagnostic-v1-001').exists(),'New integration before unattempted diagnostic only')
 pins=C.sources();bound=C.bindings();roles=C.checked(reference(E/'fixture-role-census001.json'));require(roles['no_prior_matches'] is True,'Private unused fixture role audit required')
 floor=8*1024**3+16*1024**2+1267904512+1458176;free=shutil.disk_usage(ROOT).free;require(free>=floor+2*1024**3,'Complete3endpoint metadata budget plus2GiB required')
 publish('gradient-model001-plan.json',dict(source=reference(__file__),sources=pins,environment=env,roles=reference(E/'fixture-role-census001.json'),bound_raw_endpoints=[dict(slot=b['slot'],worker=b['worker_ref'],raw=b['worker']['parent']) for b in bound],full_floor=floor,target=floor+2*1024**3,free_bytes=free,atol=4e-5,rtol=4e-4,generated_targets=42,real_gradient_calls_cpu=3,real_gradient_calls_mps=3,real_target_evaluations=6144,optimizer_steps=0,validation_encoded=False,holdout_encoded=False,weight_files_written=0))
 try:
  mps_before=torch.mps.get_rng_state().clone();cpu_before=torch.get_rng_state().clone()
  with StrictMpsGuard(E/'gradient-model001-guard.json',ROOT):
   with torch.random.fork_rng(devices=[]):
    torch.default_generator.manual_seed(600494);m=GPT(model_config('dense')).train()
   initial=C.state_hash(m);gen=torch.Generator().manual_seed(600491);x=torch.randint(1,400,(2,7),generator=gen);y=torch.randint(1,400,(2,7),generator=gen)
   cpu_grad,cpu_ce=C.gradients(m,x,y);m.zero_grad(set_to_none=True);m=m.to('mps');xx=x.to('mps');yy=y.to('mps');mps_grad,mps_ce=C.gradients(m,xx,yy)
   error=compare(cpu_grad,mps_grad);require(abs(cpu_ce-mps_ce)<3e-5,'Generated complete CE agreement failed');del cpu_grad
   moments=Moments();norms=[]
   for i in range(2):
    g,ce=C.gradients(m,xx[i:i+1],yy[i:i+1]);moments.update(g);norms.append(C.gradient_norms(g));del g
   linearity_error=compare(moments.means,mps_grad);r=moments.report()
   for n,stat in r['parameter_statistics'].items():
    independent=math.fsum(v[n] for v in norms)-2*stat['mean_gradient_squared_norm'];require(math.isclose(independent,stat['trace_sample_covariance'],rel_tol=1e-8,abs_tol=1e-10),'Independent generated moment identity differs')
   require(C.state_hash(m)==initial,'Generated gradients mutated model weights/buffers')
   grad_before={n:C.tensor_hash(p.grad) for n,p in m.named_parameters()};cases=[]
   bads=[(xx.float(),yy),(xx,yy[:,:-1]),(xx-50257,yy),(xx,yy+50257),(xx[:0],yy[:0]),(torch.zeros(9,7,dtype=torch.int64,device='mps'),torch.zeros(9,7,dtype=torch.int64,device='mps')),(torch.zeros(1,129,dtype=torch.int64,device='mps'),torch.zeros(1,129,dtype=torch.int64,device='mps')),(x,y)]
   for bx,by in bads:reject(lambda bx=bx,by=by:C.gradients(m,bx,by));cases.append('invalid_input_before_gradient_mutation')
   m.eval();reject(lambda:C.gradients(m,xx,yy));m.train();cases.append('eval_mode_rejection')
   reject(lambda:C.gradients(object(),xx,yy));cases.append('foreign_model_rejection')
   require(all(p.grad is not None and C.tensor_hash(p.grad)==grad_before[n] for n,p in m.named_parameters()),'Invalid call mutated existing gradients')
   require(C.state_hash(m)==initial,'Invalid input mutated model state')
   del m,moments,mps_grad,r,x,y,xx,yy,bads,bx,by
   gc.collect();torch.mps.empty_cache()
   actual=[];forgeries=[]
   for item in bound:
    w=item['worker'];raw=torch.load(w['parent']['path'],map_location='cpu',weights_only=True)
    for mode in ['kind','architecture','cfg','step','missing','extra']:
     bad=dict(raw)
     if mode=='kind':bad['kind']='h200_tokenizer_moe_v1'
     elif mode=='architecture':bad['architecture']=dict(raw['architecture'],arm='moe')
     elif mode=='cfg':bad['cfg']=dict(raw['cfg'],model={})
     elif mode=='step':bad['step']=1199
     elif mode=='missing':bad.pop('cpu_rng')
     else:bad['foreign']=True
     reject(lambda bad=bad:C.raw_weights(bad,w));forgeries.append(dict(slot=item['slot'],case=mode))
    del bad,raw;gc.collect()
    model=C.load_dense(w,'cpu',600494);before=C.state_hash(model)
    view=load_train(w,'cpu',600491);require(set(view.splits)==set(view.generators)=={'train'} and view.identity['validation_encoded'] is False and view.identity['holdout_encoded'] is False,'Train-only data contract differs')
    replay=torch.Generator().manual_seed(600491);starts=torch.randint(267688-512,(2,),generator=replay);x,y=view.next_train()
    expected_x=torch.stack([view.train_data[i:i+512] for i in starts]).reshape(8,128);expected_y=torch.stack([view.train_data[i+1:i+513] for i in starts]).reshape(8,128)
    require(torch.equal(x,expected_x) and torch.equal(y,expected_y) and torch.equal(replay.get_state(),view.generators['train'].get_state()),'Independent native paired512 replay differs')
    sampler=view.generators['train'].get_state().clone();ids=view.train_data;view.train_data=ids[:-1];reject(view.next_train);view.train_data=ids;require(torch.equal(sampler,view.generators['train'].get_state()),'Invalid data mutated private sampler')
    cpu_grad,ce_cpu=C.gradients(model,x,y);model.zero_grad(set_to_none=True);model=model.to('mps');g,ce_mps=C.gradients(model,x.to('mps'),y.to('mps'));difference=compare(cpu_grad,g);require(abs(ce_cpu-ce_mps)<3e-5,'Actual fixed-weight CPU/MPS training CE differs')
    # Two references to the SAME already-produced gradient stress the full FP64 tree; they are NOTtwo independent draws/noise evidence.
    stress=Moments();stress.update(g);stress.update(g);sr=stress.report();require(sr['trace_sample_covariance']==0.,'Exact repeated-vector moment identity differs')
    require(C.state_hash(model)==before,'Actual gradients mutated fixed RAW model/buffers')
    actual.append(dict(slot=item['slot'],raw=w['parent'],cpu_ce=ce_cpu,mps_ce=ce_mps,cpu_mps_ce_difference=abs(ce_cpu-ce_mps),maximum_gradient_difference=difference,gradients=len(g),fp64_mean_bytes=sum(v.numel()*v.element_size() for v in stress.means.values()),raw_payload_released_before_gradients=True,sampler_replay=True,invalid_sampler_before_mutation=True,model_unchanged=True,one_train_batch_targets=1024,repeated_gradient_buffer_stress_NOT_independent_draws=True,validation_encoded=False,holdout_encoded=False))
    del model,view,cpu_grad,g,stress,sr,x,y,expected_x,expected_y,ids;gc.collect();torch.mps.empty_cache();C.verify_artifact(w['parent'])
   require(torch.equal(cpu_before,torch.get_rng_state()) and torch.equal(mps_before,torch.mps.get_rng_state()),'CPU/MPS global RNG changed')
   require(C.sources()==pins,'Pinned numerical sources changed')
  result=dict(status='own_fixed_weight_gradient_integration_cpu_mps_and_actual_payloads_passed',source=reference(__file__),sources=pins,generated_maximum_gradient_difference=error,generated_gradient_linearity_maxerror=linearity_error,generated_invalid_cases=cases,raw_metadata_forgeries=forgeries,actual=actual,global_cpu_mps_rng_unchanged=True,optimizer_steps=0,weights_written=0,validation_encoded=False,holdout_encoded=False,quality_result=False,guards=[reference(E/'gradient-model001-guard.json')],next_action='Verify actual guards/proof/source/roles/livefull3endpointbudget;invoke NEW research.tokenizer_gradient_probe_v1.execution ONCE. Keep32newprivateB8draws×all3fixedRAWcontrols,98304targets,nooptimizer/validation/holdout/bootstrap/weights. Never rerun this integration or any closed source command.')
  publish('gradient-model001.json',result);print(json.dumps(dict(status=result['status'],sources=len(pins),generated_max_gradient_error=error,linearity_error=linearity_error,actual=actual)),flush=True)
 except BaseException as exc:publish('STOP-gradient-model001',dict(error=repr(exc),closed=True,no_retry=True));raise
