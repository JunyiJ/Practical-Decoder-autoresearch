"""Finite paired contracts, non-rescue behavior, and independent statistical checks."""
import copy,json,os
from pathlib import Path
import numpy as np
import pytest
from omegaconf import OmegaConf
from . import campaign_contract as c,campaign_rules as r,quality_execution as q
from research.readout_centering_model_v1.integration import identity
from research.readout_centering_campaign_v2.retention import planned_manifest
from research.readout_centering_model_v1.runtime.train.lr_schedule import schedule_contract,multiplier
from research.readout_centering_model_v1.runtime.train.weight_averaging import averaging_contract

def seeds():return json.loads(Path(os.environ['READOUT_CENTERING_INTEGRATION_PLAN']).read_text())['seeds']
@pytest.mark.parametrize('slot',[-1,6,12,None,True])
def test_invalid_slot(slot):
 with pytest.raises(ValueError):q.run_once(None,slot,'0'*64)
@pytest.mark.parametrize('slot',range(6))
def test_exact_config(slot):
 s=seeds();spec=r.fresh_specs(s)[slot];out=q.ROOT/f'evidence/quality/{slot:02d}';cfg=c.trial_config(out,spec,s);t=OmegaConf.create(cfg).training
 assert identity(OmegaConf.create(cfg))['decay']==r.DECAYS[spec['arm']]
 assert identity(OmegaConf.create(cfg))['grad_clip']==1.
 assert identity(OmegaConf.create(cfg))['readout']['kind']==r.READOUTS[spec['arm']]
 assert cfg['model']['readout_norm_epsilon']==1e-5
 assert t.learning_rate==.00125 and t.max_iters==1200 and t.batch_size==8 and cfg['model']['block_size']==128
 assert t.seed==s['fresh'][slot//2] and t.eval_seed==s['evaluation']
 schedule=schedule_contract(t);ema=averaging_contract(t)
 assert schedule['total_steps']==ema['total_steps']==1200 and schedule['start_step']==r.COOLDOWNS[spec['arm']] and ema['start_step']==800 and ema['interval']==10
 assert 1+(1200-800)//10==41 and multiplier(schedule,0)==multiplier(schedule,spec['cooldown_start'])==1. and multiplier(schedule,1199)==.1
 assert multiplier(schedule,spec['cooldown_start']+1)<1.
 assert abs(sum(multiplier(schedule,i) for i in range(1200))-1020)<1e-9
 for field,value in [('learning_rate',.0025),('weight_decay',0.),('seed',s['pilot']),('max_iters',20)]:
  bad=copy.deepcopy(cfg);bad['training'][field]=value
  with pytest.raises(ValueError):c.validate_trial_config(bad,out,spec,s)
 bad=copy.deepcopy(cfg);bad['model']['readout_norm_kind']='rms_affine_bias' if spec['arm']=='layer' else 'layer_affine'
 with pytest.raises(ValueError):c.validate_trial_config(bad,out,spec,s)
 bad=copy.deepcopy(spec);bad['selection']=.00125
 with pytest.raises(ValueError):c.trial_config(out,bad,s)

def test_pair_order_and_only_final_readout_kind_differs():
 s=seeds();specs=r.fresh_specs(s);assert [x['arm'] for x in specs]==['layer','rms_bias','rms_bias','layer','layer','rms_bias']
 assert not hasattr(r,'screen_specs') and not hasattr(r,'choose_rates') and not hasattr(q,'boundary')
 for pair in range(3):
  configs=[]
  for spec in specs[2*pair:2*pair+2]:
   cfg=c.trial_config(q.ROOT/f"evidence/quality/{spec['slot']:02d}",spec,s)
   for k in ('checkpoint_dir','metrics_file','summary_file'):del cfg['training'][k]
   del cfg['model']['readout_norm_kind']
   configs.append(cfg)
  assert configs[0]==configs[1]
 for n in range(6):assert q.next_spec({'fresh':specs},{'records':[{}]*n})==specs[n]
 with pytest.raises(ValueError):q.next_spec({'fresh':specs},{'records':[{}]*6})
@pytest.mark.parametrize('change',['screen','duplicate','short','bool'])
def test_seed_rejection(change):
 s=seeds()
 if change=='screen':s['screen']=s['pilot']+1
 elif change=='duplicate':s['fresh'][0]=s['pilot']
 elif change=='short':s['fresh'].pop()
 else:s['pilot']=True
 with pytest.raises(ValueError):r.validate_seeds(s)
@pytest.mark.parametrize('boundary',['lr','fresh','rank'])
def test_no_quality_retirement(boundary):
 with pytest.raises(ValueError):planned_manifest(boundary,[],None,{}, {})

def test_final_only_outputs_and_guards():
 s=seeds();spec=r.fresh_specs(s)[0];cfg=c.trial_config(q.ROOT/'evidence/quality/00',spec,s)
 sink=object.__new__(q.Sink);sink.raw=None;sink.cfg=cfg;sink.root=q.ROOT/'evidence/quality/00'
 with pytest.raises(ValueError):sink.checkpoint(sink.root/'checkpoints/final_bad.pt',{'step':20})
 assert q.QualityGuard(Path('/tmp/unused-readout-centering-contract-guard'),c.ROOT).limits['timeout_seconds']==7200
 assert q.FULL_FREE==6*560*1024**2+256*1024**2+8*1024**3
 for seconds in (0,7201):
  with pytest.raises(ValueError):q.QualityGuard(Path('/tmp/unused-readout-centering-contract-guard'),c.ROOT,seconds)

def test_bootstrap_sign_short_tail_and_gate():
 s=dict(pilot=92001,fresh=[92011,92012,92013],evaluation=92003,bootstrap=92004)
 counts=np.array([2048]*8+[17]);control=np.tile(counts*4.,(3,1));candidate=control-np.array([.03,.04,.05])[:,None]*counts
 stats=r.paired_statistics(s,np.stack([control,candidate],axis=1),counts)
 assert np.allclose(stats['seed_deltas'],[-.03,-.04,-.05],atol=1e-12)
 assert r.quality_gate(stats)['strict_quality'] and not r.quality_gate(stats)['adopted']
 for size,b in stats['bounds'].items():assert abs(b['gate_upper']+.04)<1e-12
 worse=r.paired_statistics(s,np.stack([candidate,control],axis=1),counts)
 assert not r.quality_gate(worse)['strict_quality']
 bad=copy.deepcopy(stats);bad['mean_delta']=-.02;bad['seed_deltas']=[-.02]*3
 assert not r.quality_gate(bad)['strict_quality']
 bad=copy.deepcopy(stats);bad['seed_deltas'][0]=0.;bad['mean_delta']=sum(bad['seed_deltas'])/3
 assert not r.quality_gate(bad)['strict_quality']
 bad=copy.deepcopy(stats);bad['bounds']['2048']['gate_upper']=0.
 assert not r.quality_gate(bad)['strict_quality']
 with pytest.raises(ValueError):r.paired_statistics(s,np.ones((2,2,9)),counts)
 with pytest.raises(ValueError):r.paired_statistics(s,np.full((3,2,9),np.nan),counts)

@pytest.mark.parametrize('arm',r.ARMS)
def test_complete_quality_clipping_history(arm):
 threshold=1.
 from .artifact_acceptance import audit_clipping_history
 s=seeds();spec=next(x for x in r.fresh_specs(s) if x['arm']==arm)
 cfg=OmegaConf.create(c.trial_config(q.ROOT/f"evidence/quality/{spec['slot']:02d}",spec,s))
 coef=threshold/(2.+1e-6)
 rows=[dict(step=k+1,threshold=threshold,pre_norm=2.,post_norm=2.*coef,coefficient=coef,gradient_tensors=88) for k in range(1200)]
 audit=audit_clipping_history(cfg,rows);assert audit['updates']==1200 and audit['coefficient_below_one']==1200
 for mutate in [lambda x:x.pop(),lambda x:x[-1].update(step=1199),lambda x:x[-1].update(threshold=2.),lambda x:x[-1].update(coefficient=.9),lambda x:x[-1].update(pre_norm=float('nan'))]:
  bad=copy.deepcopy(rows);mutate(bad)
  with pytest.raises(ValueError):audit_clipping_history(cfg,bad)


def test_bounded_large_report_publication():
 from src.research.tokenizer_activation_entry import session_lock
 from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
 from src.research.tokenizer_wide_hybrid_metadata import snapshot,budgets
 plan=json.loads(Path(os.environ['READOUT_CENTERING_INTEGRATION_PLAN']).read_text())
 root=Path(plan['disposable_root']);assert not root.exists();root.mkdir()
 payload=dict(scope='synthetic publication capacity only',report='x'*(512*1024))
 with session_lock(root) as fd:
  writer=MetadataWriter(root,(int(os.environ['READOUT_CENTERING_GLOBAL_FD']),fd));base='evidence/quality/00'
  worker=q.publish_worker(writer,base,payload);accepted=q.publish_acceptance(writer,base,payload)
  assert worker.name=='resource-audit.json' and accepted.name=='resource-events.jsonl'
  assert json.loads(worker.read_text())==json.loads(accepted.read_text())==payload
  snap=snapshot(root);assert snap['checkpoint_logical_bytes']==0
  assert all(snap['groups'][k]<=v for k,v in budgets().items())
  with pytest.raises(FileExistsError):q.publish_worker(writer,base,payload)
