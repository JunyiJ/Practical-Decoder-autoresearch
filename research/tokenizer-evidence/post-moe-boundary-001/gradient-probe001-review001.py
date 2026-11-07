"""Independent saved-metadata identities and terminal closure; no gradient/model replay."""
import hashlib,json,math,shutil
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.body_allocation_model_v1.contract import ROOT,require
from research.tokenizer_moe_worker_v1.execution import idle
from research.tokenizer_gradient_probe_v1 import runtime as C
E=Path(__file__).resolve().parent;R=ROOT/'research_runs/tokenizer-gradient-noise-diagnostic-v1-001'
def ref(p):return dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(r):
 p=Path(r['path']);require(ref(p)=={k:r[k] for k in ['path','sha256']},'Immutable reference differs');return json.loads(p.read_text())
def publish(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
 idle();require(not(E/'gradient-probe001-review001.json').exists(),'New terminal review only')
 with CpuPrerequisiteGuard(E/'gradient-probe001-review001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
  terminal=ref(R/'STOP');stop=checked(terminal);require(stop['status']=='complete_training_free_gradient_diagnostic' and stop['never_rerun'] is True,'Original diagnostic terminal required')
  result=checked(stop['result']);plan=checked(ref(R/'plan.json'));guard=checked(ref(R/'guard.json'));pins=C.sources()
  require(result['sources']==plan['sources']==pins and len(pins)==1043,'Exact1043frozen source bindings required')
  require(result['status']=='complete_training_free_three_endpoint_gradient_diagnostic' and result['total_target_evaluations']==98304 and result['optimizer_steps']==result['weight_files_written']==0 and result['global_cpu_mps_rng_unchanged'] and not result['validation_encoded'] and not result['holdout_encoded'],'Complete train-only fixed-weight contract required')
  require(guard['status']=='finished' and guard['reason'] is None,'Original600s/4GiB/6GiB/8GiBguard failed')
  require(plan['draws_per_endpoint']==32 and plan['private_train_seed']==600492 and plan['initializer_seed']==600493,'Original new fixture roles required');rows=[];streams=[];maxerror=0.
  for item in result['endpoints']:
   d=checked(item);r=d['statistics'];n=32;require(len(d['draws'])==r['draws']==n and len(r['parameter_statistics'])==88 and d['optimizer_steps']==0 and d['model_unchanged'],'Complete fixed-weight gradient inventory required')
   require(d['training_only_identity']['validation_encoded'] is False and d['training_only_identity']['holdout_encoded'] is False,'Training prefix only required')
   for name,stat in r['parameter_statistics'].items():
    q=math.fsum(x['gradient_squared_norms'][name] for x in d['draws']);cov=(q-n*stat['mean_gradient_squared_norm'])/(n-1);signal=stat['mean_gradient_squared_norm']-cov/n
    error=max(abs(cov-stat['trace_sample_covariance']),abs(signal-stat['signal_squared_unbiased']));maxerror=max(maxerror,error);require(error<1e-10,'Independent scalar moment identity differs')
   norm=math.fsum(s['mean_gradient_squared_norm'] for s in r['parameter_statistics'].values());cov=math.fsum(s['trace_sample_covariance'] for s in r['parameter_statistics'].values());signal=norm-cov/n
   require(math.isclose(cov,r['trace_sample_covariance'],abs_tol=1e-12) and math.isclose(signal,r['signal_squared_unbiased'],abs_tol=1e-12),'Independent complete aggregate differs')
   ratio=cov/signal if signal>0 else None;require(ratio==r['ratio_in_independent_B8_draw_units'],'Unclamped signal/noise ratio differs')
   require(sum(v['parameters'] for v in d['groups'].values())==88,'Complete input/readout/body grouping required')
   streams.append([(x['input_sha256'],x['target_sha256']) for x in d['draws']]);C.verify_artifact(d['raw'])
   rows.append(dict(slot=d['slot'],raw=d['raw'],draws=n,mean_gradient_squared_norm=norm,trace_sample_covariance=cov,signal_squared_unbiased=signal,ratio_in_independent_B8_draw_units=ratio,groups=d['groups'],noise_correction_fraction=(cov/n/norm),mean_online_train_ce=math.fsum(x['ce'] for x in d['draws'])/n))
  require([r['slot'] for r in rows]==[9,14,16] and streams[0]==streams[1]==streams[2],'All3fixed endpoints/same private stream required')
  mirror=E/'gradient-probe001-metadata-mirror001';mirror.mkdir();files=[]
  for p in [R/'plan.json',R/'guard.json',R/'result.json',R/'STOP',*[R/f'endpoint-{s:02d}.json' for s in [9,14,16]]]:
   before=ref(p);dest=mirror/p.name;shutil.copyfile(p,dest);require(ref(p)==before and ref(dest)['sha256']==before['sha256'],'Lossless metadata copy changed');files.append(dict(source=before,destination=ref(dest),size=p.stat().st_size))
  require(len(files)==7 and not list(R.rglob('*.pt')) and ref(R/'STOP')==terminal,'No tensor output or terminal mutation permitted')
  publish(mirror/'completion.json',dict(status='closed_gradient_diagnostic_metadata_mirror_complete',files=files,weight_copies=0))
 report=dict(status='complete_training_free_gradient_diagnostic_independently_reviewed',source=ref(Path(__file__)),terminal=terminal,result=stop['result'],plan=ref(R/'plan.json'),guard=ref(R/'guard.json'),sources=1043,rows=rows,max_scalar_identity_error=maxerror,closed_metadata_mirror=ref(mirror/'completion.json'),files=7,weight_copies=0,gradients_replayed=False,model_loaded=False,optimizer_steps=0,validation_encoded=False,holdout_encoded=False,adopted=False,quality_result=False,next_action='Raw-gradient variation is high at3fixedRAWendpoints,not anoptimalbatch orqualitygain. Because N32signal subtracts a large noise correction and ratios arebiased,admitonlyNEW independently sampled cohort-stability diagnostic before batch-training feasibility. Preserve original32draws/STOP/source;newroots/seeds/source/gates required,no old diagnostic replay.')
 publish(E/'gradient-probe001-review001.json',report);print(json.dumps(dict(status=report['status'],rows=[dict(slot=r['slot'],ratio=r['ratio_in_independent_B8_draw_units'],noise_correction_fraction=r['noise_correction_fraction']) for r in rows],maxerror=maxerror)))
