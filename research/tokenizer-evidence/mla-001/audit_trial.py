"""Read-only numerical/lineage audit of a completed tokenizer trial; no inference."""
import argparse,json,math
from pathlib import Path
import torch
from omegaconf import OmegaConf
from src.research.protocol import file_digest
from src.train.train import _make_loader,_training_contract
from src.utils.checkpoint import read_checkpoint,checkpoint_loader
p=argparse.ArgumentParser();p.add_argument('trial',type=Path);args=p.parse_args()
root=args.trial.resolve();r=json.loads((root/'result.json').read_text());req=json.loads((root/'request.json').read_text())
assert r['request_sha256']==file_digest(root/'request.json')
from src.research.tokenizer_mla_study import config
assert req['config']==config(root,req['spec']['family'],req['spec']['seed'],req['spec']['lr'])
cfg=OmegaConf.create(req['config']);steps=cfg.training.max_iters
assert r['training']['completed_steps']==steps and steps == 1200
raw=read_checkpoint(r['raw']['checkpoint']);avg=read_checkpoint(r['h200']['checkpoint'])
assert raw['cfg']==req['config']==avg['cfg'] and raw['step']==avg['step']==steps
loader=_make_loader(cfg,'cpu')
assert raw['training_state']['contract']==_training_contract(cfg,'mps',loader)
assert torch.equal(raw['training_state']['loader_rng']['test'],loader.generators['test'].get_state())
for _ in range(steps):loader.get_batch('train')
assert torch.equal(raw['training_state']['loader_rng']['train'],loader.generators['train'].get_state())
snapshots=1+(steps-cfg.training.weight_averaging.start_step)//cfg.training.weight_averaging.interval
assert avg['averaged_weights']['snapshots']==snapshots
assert raw['training_state']['weight_averaging_state']['snapshots']==snapshots
assert avg['parent_checkpoint']['sha256']==file_digest(Path(r['raw']['checkpoint']))
assert avg['tokenizer_contract']==loader.tokenizer_contract and 'optimizer_state' not in avg
assert raw['optimizer_state']['independent_decay']==.00006
for optimizer in raw['optimizer_state']['optimizers']:
 for state in optimizer['state'].values():
  for name,value in state.items():
   if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
   if name=='step':assert int(value)==steps
 for group in optimizer['param_groups']:assert group['weight_decay']==0
for name in ['raw','h200']:
 score=r[name];payload=raw if name=='raw' else avg
 assert score['checkpoint_sha256']==file_digest(Path(score['checkpoint']))
 assert all(torch.isfinite(v).all() for v in payload['model_state'].values())
 assert score['target_count']==34278 and sum(score['block_target_counts'])==34278
 expected=[min(2048,34278-i) for i in range(0,34278,2048)]
 assert score['block_target_counts']==expected
 assert math.isclose(sum(score['block_loss_sums'])/34278,score['mean_ce'],rel_tol=1e-12)
 assert math.isclose(score['bits_per_byte'],sum(score['block_loss_sums'])/111536/math.log(2),rel_tol=1e-12)
 assert math.isclose(score['perplexity'],math.exp(score['mean_ce']),rel_tol=1e-12)
checkpoint_loader(OmegaConf.create(avg['cfg']),avg,cfg.data.path,'cpu')
assert r['training']['final_metric']['lr_multiplier']==.1
review=dict(id=r['id'],status='passed',result_sha256=file_digest(root/'result.json'),
            request_sha256=file_digest(root/'request.json'),completed_steps=steps,ema_snapshots=snapshots,
            raw_ce=r['raw']['mean_ce'],h200_ce=r['h200']['mean_ce'],h200_bpb=r['h200']['bits_per_byte'],
            train_sampler_replayed=True,test_rng_untouched=True,
            scope='Same-agent saved-tensor/metadata audit, no training or scoring rerun. Not independent seed evidence.')
output=Path('research_state/tokenizer-mla-validation')/(r['id']+'-review.json')
output.write_text(json.dumps(review,indent=2));print(json.dumps(review,indent=2))
