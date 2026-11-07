"""One complete fresh-panel metadata analysis; no model/scoring/bootstrap replay."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path(__file__).resolve().parents[3]
B=Path(__file__).resolve().parent
E=B/'readout-quality001-existing-analysis001'
ARMS=['native','context_mix2','probability_mix2']
def ref(p):
 p=Path(p);return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def checked(v):
 assert ref(v['path'])==v;return json.loads(Path(v['path']).read_text())
def publish(name,value):
 raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 assert sum(p.stat().st_size for p in E.iterdir() if p.is_file())+len(raw)<2*1024**2
 with (E/name).open('xb') as f:f.write(raw)
def main(args):
 assert ref(E/'plan.json')['sha256']==args.plan_sha
 plan=checked(ref(E/'plan.json'));assert plan['source']==ref(__file__) and not(E/'attempt.json').exists() and not(E/'STOP').exists()
 for n,h in plan['sources'].items():assert ref(ROOT/n)['sha256']==h,n
 closure=checked(plan['closure']);assert closure['status']=='complete_all_eighteen_readout_quality_closed_not_adopted' and closure['adopted'] is False
 publish('attempt.json',dict(pid=os.getpid(),plan=ref(E/'plan.json')))
 try:
  with CpuPrerequisiteGuard(E/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
   began=time.perf_counter();paired={};inputs=[];curves=[]
   for a in closure['anchors'][9:]:
    row=checked(a['accepted']);worker=checked(row['worker']);accept=row['acceptance'];seed=worker['config']['training']['seed'];arm=row['arm']
    assert accept['status']=='accepted' and accept['step']==1200 and accept['targets']==1228800
    assert [x['step'] for x in worker['metrics']]==[1,301,601,901,1200]
    paired.setdefault(seed,{})[arm]=dict(raw=accept['full_validation']['raw']['mps']['ce'],h200=accept['full_validation']['h200']['mps']['ce'],clipped=accept['clipped_updates'])
    inputs.extend([a['accepted'],row['worker']]);curves.append(dict(seed=seed,arm=arm,points=worker['metrics']))
   assert len(paired)==3 and all(set(v)==set(ARMS) for v in paired.values())
   contrasts={}
   for name,candidate,control in [('MoS-minus-native','probability_mix2','native'),('MoS-minus-MoC','probability_mix2','context_mix2'),('MoC-minus-native','context_mix2','native')]:
    contrasts[name]={}
    for endpoint in ['raw','h200']:
     deltas=[v[candidate][endpoint]-v[control][endpoint] for v in paired.values()]
     contrasts[name][endpoint]=dict(deltas=deltas,mean=math.fsum(deltas)/3,all_negative=all(x<0 for x in deltas))
   primary=checked(closure['primary_statistics'])['statistics']
   for name,control in [('MoS-minus-native','native'),('MoS-minus-MoC','context_mix2')]:
    assert abs(contrasts[name]['h200']['mean']-primary['contrasts'][control]['mean_ce_delta'])<1e-12
   averaging={arm:dict(deltas=[v[arm]['h200']-v[arm]['raw'] for v in paired.values()],clipped_counts=[v[arm]['clipped'] for v in paired.values()]) for arm in ARMS}
   for arm,v in averaging.items():v['mean']=math.fsum(v['deltas'])/3
   for reference in inputs:checked(reference)
   for n,h in plan['sources'].items():assert ref(ROOT/n)['sha256']==h,n
   publish('result.json',dict(status='complete_closed_fresh_readout_metadata_analysis',source=ref(__file__),plan=ref(E/'plan.json'),closure=plan['closure'],seed_order=list(paired),complete_pairs=paired,contrasts=contrasts,h200_minus_raw=averaging,sparse_online_curves=curves,
    input_metadata=inputs,bootstrap_draws=0,model_checkpoint_tensor_loads=0,corpus_scoring_calls=0,optimizer_updates=0,new_quality_seeds=0,holdout_scored=False,all_fresh_arms_and_seeds_retained=True,incumbent_unchanged=True,adopted=False,
    limits='Descriptive existing full-panel metadata only. RAW is not an alternative selector; five sampled online points and clipping are not mechanism/EMA/horizon/overfitting/conditioning diagnoses. MoC-minus-native is descriptive, not a fallback selection or new confidence test. No retrospective quality gate changes.',elapsed_seconds=time.perf_counter()-began))
  publish('STOP',dict(status='complete_existing_readout_metadata_analysis',no_replay=True,no_retry=True));print(json.dumps(dict(contrasts=contrasts,h200_minus_raw=averaging)),flush=True)
 except BaseException as exc:
  if not(E/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
  raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--plan-sha',required=True);main(p.parse_args())
