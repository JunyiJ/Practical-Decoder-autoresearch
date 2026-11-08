"""Independent saved-artifact checks; no model or training function calls."""
import hashlib,json,math
from pathlib import Path

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while x:=f.read(1048576):h.update(x)
 return h.hexdigest()
def check(r):
 p=Path(r['path']);assert p.is_file() and not p.is_symlink() and p.stat().st_nlink==1
 assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
def read(p):return json.loads(p.read_text())

def run(plan,spec,out):
 import torch
 torch.set_num_threads(4)
 result=read(out/'result.json');assert result['accepted'] and result['spec']==spec
 parent=result['parent'];h200=result['h200'];check(parent);check(h200)
 raw=torch.load(parent['path'],map_location='cpu',weights_only=True)
 avg=torch.load(h200['path'],map_location='cpu',weights_only=True)
 identity=raw['identity'];numerical=raw['numerical'];names=list(numerical['parameters'])
 roles=plan['bundles'][spec['bundle']];native=spec['variant']=='native';count=88 if native else 105
 assert raw['kind']=='byte_input_training_context_v2' and avg['kind']=='byte_input_h200_inference_v1'
 assert identity['roles']==roles and identity['binding']==plan['binding'] and identity['data']==plan['data']
 assert identity['recipe']==plan['recipes'][str(spec['learning_rate'])]
 assert identity['model'][1]==spec['variant'] and identity['model'][2]==(None if native else roles['composer_initializer'])
 if spec['variant']=='nonlexical_codes':assert identity['model'][3][2]==roles['nonlexical_key']
 assert numerical['completed_steps']==1200 and numerical['backend']=='mps:0'
 assert len(names)==count and sum(t.numel() for t in numerical['parameters'].values())==(28492416 if native else 16351859)
 assert len(raw['history'])==1200
 for step in (1,301,601,901,1200):
  d=read(out/('diagnostic'+str(step).zfill(4)+'.json'))
  assert d['step']==step and d['targets_seen']==step*1024 and d['eval_iters']==10 and d['diagnostic_only']
  assert d['train_sampler_unchanged'] and d['global_rng_unchanged'] and d['parameters_unchanged']
  assert len(d['individual_batch_losses'])==10 and all(math.isfinite(x) and x>=0 for x in d['individual_batch_losses'])
  assert d['validation_ce']==sum(d['individual_batch_losses'])/10
 times=[];observed_driver=[]
 for i,history in enumerate(raw['history'],1):
  observed=read(out/('step'+str(i).zfill(4)+'.json'))
  assert history=={k:observed[k] for k in history}
  assert history['step']==i and history['active_parameter_names']==names and history['gradient_tensors']==count
  assert history['threshold']==1. and 0<=history['post_norm']<=1.+1e-5
  coefficient=min(1.,1./(history['pre_norm']+1e-6))
  assert 0<=history['coefficient']<=1 and abs(coefficient-history['coefficient'])<2e-5
  assert abs(history['post_norm']-history['pre_norm']*history['coefficient'])<=max(1e-5,history['post_norm']*2e-5)
  assert math.isfinite(history['ce_loss']) and history['ce_loss']>=0
  assert observed['elapsed_seconds']>0 and observed['mps_current_bytes']<=observed['mps_driver_bytes']<6*1024**3
  times.append(observed['elapsed_seconds']);observed_driver.append(observed['mps_driver_bytes'])
 assert len(numerical['children'])==2
 group_names=[]
 for index,child in enumerate(numerical['children']):
  assert list(child['state'])==[g['name'] for g in child['groups']]
  assert all(g['options']['lr']==spec['learning_rate']*(.5 if index==0 else 1.)*.1 for g in child['groups'])
  group_names.extend(child['state'])
  for name,fields in child['state'].items():
   assert set(fields)==({'momentum_buffer'} if index==0 else {'step','exp_avg','exp_avg_sq'})
   for key,t in fields.items():
    assert t.dtype==torch.float32 and bool(torch.isfinite(t).all())
    if key=='step':assert tuple(t.shape)==() and t.item()==1200
    else:assert t.shape==numerical['parameters'][name].shape
    if key=='exp_avg_sq':assert bool((t>=0).all())
 assert len(group_names)==len(set(group_names))==count and set(group_names)==set(names)
 saved=raw['averages'];assert saved['completed_steps']==1200 and saved['snapshots']==41
 assert saved['spec']['start_step']==800 and saved['spec']['interval']==10 and saved['spec']['half_lives']==[200]
 assert avg['parent']==parent and avg['identity']==identity and avg['completed_steps']==1200
 assert avg['averaging']==dict(half_life=200,spec=saved['spec'],snapshots=41) and list(avg['parameters'])==names
 for name,t in avg['parameters'].items():
  assert t.dtype==torch.float32 and t.shape==numerical['parameters'][name].shape and bool(torch.isfinite(t).all())
  assert torch.equal(t,saved['averages']['200'][name])
 samplers={}
 for name,state in raw['samplers'].items():
  assert torch.Generator(device='cpu').set_state(state).initial_seed()==roles['train_sampler' if name=='train' else 'validation_sampler']
  samplers[name]=hashlib.sha256(state.numpy().tobytes()).hexdigest()
 scores=read(out/'validation.json');report=scores['report'];loss=scores['per_target_losses']
 assert len(loss)==34278 and report['target']==plan['target'] and report['binding']==h200
 assert all(type(x)is float and math.isfinite(x) and x>=0 for x in loss)
 assert abs(math.fsum(loss)-report['total_loss'])<1e-8
 for field,width in [('window_loss_sums',128),('block_loss_sums',2048)]:
  expected=[math.fsum(loss[i:i+width]) for i in range(0,len(loss),width)]
  assert len(expected)==len(report[field]) and all(abs(a-b)<1e-8 for a,b in zip(expected,report[field]))
 assert report['ce']==report['total_loss']/34278
 assert report['bits_per_byte']==report['total_loss']/math.log(2)/report['target']['target_bytes']
 serving=read(out/'inference.json')
 assert [r['batch'] for r in serving['correctness']]==[1,8] and len(serving['timings'])==14
 for r in serving['correctness']:
  assert r['cache_bytes']==r['batch']*196608 and r['reset_exact'] and 0<=r['max_gate_ratio']<=1 and 0<=r['max_absolute_error']<math.inf
 for i,r in enumerate(serving['timings']):
  assert (r['batch'],r['repeat'],r['warmup'])==((1 if i<7 else 8),i%7,i%7<2)
  assert all(type(r[k])is float and math.isfinite(r[k]) and r[k]>0 for k in ['full_seconds','prefill96_seconds','decode32_seconds'])
 assert result['completed_steps']==1200 and result['targets_seen']==1228800 and not result['quality_selected']
 assert result['max_update_seconds']==max(times) and result['total_update_seconds']==sum(times) and result['mean_update_seconds']==sum(times)/1200
 assert result['max_mps_driver_bytes']==max(observed_driver)
 check(parent);check(h200)
 return dict(accepted=True,slot=spec['slot'],variant=spec['variant'],stage=spec['stage'],bundle=spec['bundle'],learning_rate=spec['learning_rate'],
  ce=report['ce'],bits_per_byte=report['bits_per_byte'],sampler_hashes=samplers,parent=parent,h200=h200,
  completed_steps=1200,targets_seen=1228800,averaging_snapshots=41,validation_targets=34278,
  total_update_seconds=sum(times),max_update_seconds=max(times),quality_selected=False,
  terminal_guard_and_STOP_required=True,scope='Independent saved RAW/H200, optimizer/EMA/history/data/sampler, complete validation aggregation and inference receipt checks. Requires finished outer guard and terminal STOP before acceptance.')
