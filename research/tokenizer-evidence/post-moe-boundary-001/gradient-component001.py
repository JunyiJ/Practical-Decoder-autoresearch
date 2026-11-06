"""Generated independent moment oracle and transactional forgeries; no corpus/model/quality RNG."""
import json,hashlib,copy,shutil
from pathlib import Path
import torch
from research.tokenizer_gradient_noise_v1.component import Moments
from research.body_allocation_model_v1.contract import ROOT,require
from research.tokenizer_moe_resume_v1.contract import runtime
from research.tokenizer_moe_worker_v1.execution import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;OLD=E.parent/'post-prefix-boundary-001'
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def publish(n,d):
 with (E/n).open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
def reject(fn):
 try:fn()
 except (ValueError,TypeError):return
 raise AssertionError('Malformed moment input accepted')
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
 idle();env=runtime();assert not(E/'gradient-component001-plan.json').exists()
 plan=json.loads((OLD/'quality001-plan.json').read_text());pins=dict(plan['sources'])
 for p,h in pins.items():assert ref(ROOT/p)['sha256']==h
 for p in sorted((ROOT/'research/tokenizer_gradient_noise_v1').glob('*.py')):pins[str(p.relative_to(ROOT))]=ref(p)['sha256']
 pins[str(Path(__file__).relative_to(ROOT))]=ref(Path(__file__))['sha256'];assert len(pins)==1038
 floor=8*1024**3+16*1024**2+1267904512+1458176;assert shutil.disk_usage(ROOT).free>=floor+2*1024**3
 publish('gradient-component001-plan.json',dict(source=ref(Path(__file__)),sources=pins,fixture_seed=600490,environment=env,full_floor=floor,target=floor+2*1024**3,output_metadata_cap=16*1024**2,training=False,corpus=False,model=False,weights_written=0,scientific_quality_seed=False))
 try:
  before=torch.get_rng_state().clone()
  with CpuPrerequisiteGuard(E/'gradient-component001-guard.json',ROOT,timeout_seconds=600,poll_seconds=.25):
   gen=torch.Generator().manual_seed(600490);cases=[];maxerror=0.
   for dtype in [torch.float64,torch.float32]:
    for mode in ['ordinary','constant','zero_mean']:
     n=16;draws=[]
     for i in range(n):
      if mode=='ordinary':draws.append({'matrix':torch.randn(3,7,generator=gen,dtype=dtype)+2.,'vector':torch.randn(5,generator=gen,dtype=dtype)+1.})
      elif mode=='constant':draws.append({'matrix':torch.full((3,7),.25,dtype=dtype),'vector':torch.full((5,),.5,dtype=dtype)})
      else:draws.append({'matrix':torch.full((3,7),(-1.)**i,dtype=dtype),'vector':torch.full((5,),(-1.)**i,dtype=dtype)})
     m=Moments()
     for d in draws:m.update(d)
     result=m.report();vectors=torch.stack([torch.cat([d[k].flatten().double() for k in ['matrix','vector']]) for d in draws]);mu=vectors.mean(0);cov=float((vectors-mu).square().sum())/(n-1);norm=float(mu.square().sum());signal=norm-cov/n
     error=max(abs(cov-result['trace_sample_covariance']),abs(norm-result['mean_gradient_squared_norm']),abs(signal-result['signal_squared_unbiased']));maxerror=max(maxerror,error);assert error<1e-12
     assert result['positive_signal_estimate']==(signal>0)
     if signal>0:assert abs(result['ratio_in_independent_B8_draw_units']-cov/signal)<1e-12
     else:assert result['ratio_in_independent_B8_draw_units'] is None
     if mode=='constant':assert result['trace_sample_covariance']==0.
     # Equation A.2 for1 independent B8 draw versus their16-draw mean.
     q=float(vectors.square().sum(1).mean());assert abs((n*norm-q)/(n-1)-signal)<1e-12 and abs((q-norm)/(1-1/n)-cov)<1e-12
     reject(lambda:Moments().report());snapshot=copy.deepcopy(m.__dict__)
     for bad in [{},{'matrix':draws[0]['matrix']},{'matrix':torch.zeros(7,dtype=dtype),'vector':draws[0]['vector']},{'matrix':draws[0]['matrix'].long(),'vector':draws[0]['vector']},{'matrix':torch.full((3,7),float('nan'),dtype=dtype),'vector':draws[0]['vector']}]:
      reject(lambda bad=bad:m.update(bad));assert m.count==snapshot['count'] and m.schema==snapshot['schema'] and m.m2==snapshot['m2'] and all(torch.equal(v,snapshot['means'][k]) for k,v in m.means.items())
     cases.append(dict(dtype=str(dtype),mode=mode,draws=n,error=error,invalid_updates_rejected_before_mutation=5))
   assert torch.equal(before,torch.get_rng_state())
   for p,h in pins.items():assert ref(ROOT/p)['sha256']==h
  result=dict(status='generated_gradient_moment_oracle_and_transactional_contracts_passed',source=ref(Path(__file__)),sources=pins,cases=cases,maxerror=maxerror,global_cpu_rng_unchanged=True,corpus=False,model=False,training=False,optimizer_steps=0,weight_files_written=0,quality_result=False,next_action='Implement own fixed-weight native dense RAW-endpoint gradient-sampling integration for all3fresh controls with no optimizer update/validation. Verify source/data/weights/RNG immutability,strictMPS,complete metadata/memory budget and private new fixture roles before actual diagnostic. No batch-quality panel or inferred optimum.')
  publish('gradient-component001.json',result);print(json.dumps(dict(status=result['status'],sources=len(pins),cases=len(cases),maxerror=maxerror)))
 except BaseException as exc:publish('STOP-gradient-component001',dict(error=repr(exc),closed=True,no_retry=True));raise
