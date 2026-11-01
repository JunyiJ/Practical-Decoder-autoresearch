"""Narrow actual1200-step contracts and retention adapter tests; no training."""
import copy,json,os
from pathlib import Path
from types import SimpleNamespace
import pytest
from . import campaign_contract as c,campaign_rules as rules,quality_execution as q,quality_retention as r
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import session_lock
from research.local_mix_v1.runtime.train.lr_schedule import schedule_contract
from research.local_mix_v1.runtime.train.weight_averaging import averaging_contract
from omegaconf import OmegaConf

def plan():return read(reference(Path(os.environ['LOCAL_MIX_INTEGRATION_PLAN'])))

@pytest.mark.parametrize('slot',[-1,12,None,True])
def test_invalid_slot(slot):
 with pytest.raises(ValueError):q.run_once(None,slot,'0'*64)

def test_finite_configs():
 seeds=plan()['seeds']
 for a in rules.RATES:
  for b in rules.RATES:
   specs=rules.screen_specs(seeds)+rules.confirmation_specs(seeds,dict(zip(rules.ARMS,(a,b))))
   for i,spec in enumerate(specs):
    out=q.ROOT/f'evidence/quality/{i:02d}';cfg=c.trial_config(out,spec,seeds);c.validate_trial_config(cfg,out,spec,seeds)
    t=OmegaConf.create(cfg).training;assert schedule_contract(t)['total_steps']==averaging_contract(t)['total_steps']==1200
    assert t.lr_schedule.start_step==800 and t.weight_averaging.start_step==800 and t.weight_averaging.interval==10
    assert cfg['model']['attention_logit_scale']==.5 and cfg['model']['ffn_hidden_norm']=='none'
    assert schedule_contract(t)['kind']=='linear_cooldown'
    assert cfg['model']['ffn_input_mixing']==spec['arm']
    bad=copy.deepcopy(cfg);bad['training']['learning_rate']*=2
    with pytest.raises(ValueError):c.validate_trial_config(bad,out,spec,seeds)

def test_sink_and_confirmation_boundaries():
 cfg=c.trial_config(q.ROOT/'evidence/quality/00',rules.screen_specs(plan()['seeds'])[0],plan()['seeds'])
 sink=object.__new__(q.Sink);sink.raw=None;sink.cfg=cfg;sink.root=q.ROOT/'evidence/quality/00'
 with pytest.raises(ValueError):sink.checkpoint(sink.root/'checkpoints/final_bad.pt',dict(step=20))
 with pytest.raises(ValueError):q.next_spec({},dict(records=[{}]*6,cleanup=None))
 guard=q.QualityGuard(Path('/tmp/unused-score-quality-guard'),c.ROOT);assert guard.limits['timeout_seconds']==7200

def test_retention_adapter_on_disposable_files():
 root=Path(plan()['disposable_root']);root.mkdir();e=root/'evidence';requests=[];records=[];seeds=plan()['seeds']
 for slot,spec in enumerate(rules.screen_specs(seeds)):
  out=root/f'evidence/quality/{slot:02d}';(out/'checkpoints').mkdir(parents=True);cfg=c.trial_config(out,spec,seeds);files={}
  for role in ('parent','h200'):
   p=out/'checkpoints'/f'{role}.pt';p.write_bytes(f'disposable {slot} {role}'.encode());files[role]=reference(p)
  q.exclusive(out/'audit.json',dict(disposable=True));requests.append(dict(config=cfg,files=files));records.append(dict(spec,status='completed',h200_ce=3.+slot/100))
 selection=rules.choose_rates(seeds,records);(e/'meta').mkdir();q.exclusive(e/'meta/selection.json',dict(value=selection))
 args=r.inputs(root,seeds,records,selection,requests);assert len(r.tested.planned_manifest(**args)['delete'])==8
 bad=copy.deepcopy(requests);bad[0]['files']['parent']=bad[2]['files']['parent']
 with pytest.raises(ValueError):r.inputs(root,seeds,records,selection,bad)
 context=dict(scope='disposable_score_retention_adapter')
 with session_lock(root) as fd:
  writer=SimpleNamespace(root=root,lock_fds=(int(os.environ['LOCAL_MIX_GLOBAL_FD']),fd))
  ref=r.execute(writer,e/'.retention/lr',seeds,records,selection,requests,context,lambda:context)
  assert len(r.verify(ref,context))==8
  with pytest.raises(FileExistsError):r.tested.execute_boundary(e,'lr',writer.lock_fds,lambda:None,lambda:context)


def test_mixing_is_only_arm_difference():
 seeds=plan()['seeds'];cfgs=[]
 for i,arm in enumerate(rules.ARMS):
  spec=rules.screen_specs(seeds)[i];cfg=c.trial_config(q.ROOT/f'evidence/quality/{i:02d}',spec,seeds)
  from research.local_mix_v1.runtime.train.lr_schedule import multiplier
  schedule=schedule_contract(OmegaConf.create(cfg).training)
  assert multiplier(schedule,0)==1.
  assert multiplier(schedule,99)==multiplier(schedule,799)==multiplier(schedule,800)==1.
  assert multiplier(schedule,1199)==.1
  t=cfg['training']
  for k in ('checkpoint_dir','metrics_file','summary_file'):t.pop(k)
  cfg['model'].pop('ffn_input_mixing')
  cfgs.append(cfg)
 assert cfgs[0]==cfgs[1]
