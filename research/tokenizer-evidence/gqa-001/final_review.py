import json,statistics,shutil
from pathlib import Path
import numpy as np
from src.research.protocol import file_digest,digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
run=Path('research_runs/tokenizer-gqa-001');out=Path('research/tokenizer-evidence/gqa-001');plan=json.loads((run/'plan.json').read_text());state=json.loads((run/'state.json').read_text())
assert state['status']=='completed' and len(state['records'])==20 and state['plan_sha256']==digest(plan)
assert source_fingerprint()==plan['source'] and runtime_identity()==plan['runtime']
assert file_digest(Path('research/TOKENIZER_GQA_PROTOCOL.md'))==plan['protocol_sha256']
assert file_digest(Path(plan['config']['data']['path']))==plan['corpus_sha256']
assert file_digest(Path('research/tokenizer-evidence/quality-001/adopted-recipe.json'))==plan['adopted_recipe_sha256']
for r in state['records']:
 trial=run/r['id'];rev=json.loads((out/r['id']/'review.json').read_text());req=json.loads((trial/'request.json').read_text())
 assert r==json.loads((trial/'result.json').read_text()) and r['status']=='completed' and not r['holdout_consumed']
 assert rev['result_sha256']==file_digest(trial/'result.json') and rev['status']=='passed'
 assert r['request_sha256']==file_digest(trial/'request.json')
 assert req['config']['training']['max_iters']==3000 and req['config']['training']['learning_rate']==r['lr']
 for arm in ('raw','h200'):assert file_digest(Path(r[arm]['checkpoint']))==r[arm]['checkpoint_sha256']
screen=json.loads((out/'screen-review.json').read_text());assert state['selection']==screen['selected']=={'mha':.00125,'gqa':.00125}
comparisons=[];costs=[]
for w in (64,256):
 arms={f:{r['seed']:r['h200'] for r in state['records'] if r['stage']=='confirm' and r['width']==w and r['family']==f} for f in ('mha','gqa')}
 for f in arms:assert sorted(arms[f])==[2711,2712,2713]
 saved=next(c for c in state['conclusion']['comparisons'] if c['width']==w)
 for interval in saved['intervals']:
  counts=arms['mha'][2711]['block_target_counts'];step=interval['text_block_tokens']//2048
  diffs=[[arms['gqa'][s]['block_loss_sums'][i]-arms['mha'][s]['block_loss_sums'][i] for s in (2711,2712,2713)] for i in range(len(counts))]
  avgs=[statistics.mean(v) for v in diffs];sums=[sum(avgs[i:i+step]) for i in range(0,len(counts),step)];weights=[sum(counts[i:i+step]) for i in range(0,len(counts),step)]
  rng=np.random.default_rng(20261024);samples=rng.integers(len(weights),size=(2000,len(weights)))
  estimates=[sum(sums[i] for i in a)/sum(weights[i] for i in a) for a in samples]
  assert np.allclose(np.quantile(estimates,[.025,.975]),interval['conditional_text_bootstrap_95pct'],atol=1e-12,rtol=0)
 means={f:statistics.mean(v['mean_ce'] for v in arms[f].values()) for f in arms}
 deltas=[arms['gqa'][s]['mean_ce']-arms['mha'][s]['mean_ce'] for s in (2711,2712,2713)]
 eligible=statistics.mean(deltas)<=.01 and all(x<=.01 for x in deltas) and all(x['conditional_text_bootstrap_95pct'][1]<=.01 for x in saved['intervals'])
 assert eligible==saved['quality_eligible']
 comparisons.append(dict(**saved,mean_h200_ce=means))
 data=[]
 for s in (2711,2712,2713):
  p=run/'final-inference'/f'w{w}-s{s}.json';d=json.loads(p.read_text());assert d['status']=='completed' and len(d['records'])==32 and len(d['correctness'])==4
  for f in arms:assert d['metadata'][f]['sha256']==arms[f][s]['checkpoint_sha256']
  assert (d['prompt_tokens'],d['decode_tokens'],d['warmups'],d['repeats'])==(32,64,2,8)
  data.append(d)
 for b in (1,8):
  rows=[r for d in data for r in d['records'] if r['batch']==b];summary={}
  for f in arms:
   cells=[r for r in rows if r['family']==f];kv={r['kv_bytes'] for r in cells};assert len(kv)==1 and len(cells)==24
   summary[f]=dict(kv_bytes=kv.pop(),parameter_bytes=data[0]['metadata'][f]['parameter_bytes'],median_prefill_seconds=statistics.median(r['prefill_seconds'] for r in cells),median_decode_seconds=statistics.median(r['decode_seconds'] for r in cells),decode_seconds_range=[min(r['decode_seconds'] for r in cells),max(r['decode_seconds'] for r in cells)])
  reduction=1-summary['gqa']['kv_bytes']/summary['mha']['kv_bytes'];assert reduction==.5
  costs.append(dict(width=w,batch=b,arms=summary,kv_reduction=reduction,decode_median_ratio=summary['gqa']['median_decode_seconds']/summary['mha']['median_decode_seconds']))
assert json.loads((run/'final-inference/completion.json').read_text())['source_unchanged']
review=dict(status='completed',quality=comparisons,inference=costs,adoption={'64':False,'256':True},scope='Adopt width256 half-KV GQA at LR0.00125 as a scoped serving recipe versus equally tuned MHA at LR0.00125. Retain general MHA LR0.0025 incumbent; no fresh LR0.00125-vs0.0025 general baseline adoption. Short context128 Mac MPS FP32 evidence only; test closed.',limitations='Same-agent artifact review and separately recomputed conditional text intervals; adaptive validation, not independent seed-population or test evidence. Timing uses two resident models, fixed training tokens, 32 prefill +64 decode, 24 measured repeats per family/batch across three checkpoints; KV allocation reductions are not total-memory or cloud-cost savings. GQA decodes slightly slower here.',source_runtime_protocol_corpus_unchanged=True,training_seconds=sum(r['training']['training_time_sec'] for r in state['records']))
(out/'final-review.json').write_text(json.dumps(review,indent=2)+'\n');shutil.copyfile(run/'state.json',out/'final-state.json');shutil.copytree(run/'final-inference',out/'final-inference',dirs_exist_ok=False)
r=next(r for r in state['records'] if r['stage']=='confirm' and r['width']==256 and r['family']=='gqa');cfg=json.loads((run/r['id']/'request.json').read_text())['config']
(out/'adopted-width256-serving-recipe.json').write_text(json.dumps(dict(config=cfg,scope=review['scope'],evidence='final-review.json',checkpoints={str(s):next(x['h200']['checkpoint'] for x in state['records'] if x['stage']=='confirm' and x['width']==256 and x['family']=='gqa' and x['seed']==s) for s in (2711,2712,2713)}),indent=2)+'\n')
print(json.dumps({'quality':[{k:v for k,v in c.items() if k!='intervals'} for c in comparisons],'costs':costs,'adoption':review['adoption'],'training_seconds':review['training_seconds']},indent=2))
