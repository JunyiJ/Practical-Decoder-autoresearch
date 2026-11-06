"""Independent common-byte coverage/window/loss oracles;no model scores."""
import copy,json,math
from pathlib import Path
import torch
from research.gpt2_prefix_tokenizer_v1.codec import Codec,SIZES
from research.tokenizer_common_bytes_v1.metric import target_plan,validate,windows,loss_summary
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent
def rejects(fn):
 try:fn()
 except (ValueError,TypeError):return
 raise AssertionError('Invalid target/loss contract accepted')
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();require(not(E/'metric-contracts001.json').exists(),'New finite byte-metric proof only')
 with CpuPrerequisiteGuard(E/'metric-contracts001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
  census=json.loads((E/'result.json').read_text());require(census['status']=='completed_existing_GPT2_prefix_codec_census','Actual codec contracts required')
  for n,h in census['codec_sources'].items():require(reference(ROOT/n)['sha256']==h,'Immutable codec changed')
  text=(ROOT/'data/raw/shakespeare.txt').read_text();a=int(.8*len(text));b=int((.8+.1)*len(text));val=text[a:b];del text
  fixture=('ASCII '+('café 🦊 中文\t<|endoftext|>\n'*1400));reports=[];cases=0
  for size in SIZES:
   codec=Codec(size)
   for data in [fixture,val]:
    plan=target_plan(codec,data);validate(codec,data,plan);collected=[]
    for x,y,start in windows(plan):require(len(x)==len(y)<=128 and x[1:]==y[:-1] and x[0]==(plan['context_ids'][-1] if start==0 else plan['target_ids'][start-1]),'Reset window differs');collected.extend(y)
    require(collected==plan['target_ids'],'Repeated/missing target')
    losses=[.2+(i%11)*.037 for i in range(plan['target_tokens'])];out=loss_summary(plan,losses)
    byte_losses=[value/n for value,n in zip(losses,plan['target_token_byte_lengths']) for _ in range(n)];require(len(byte_losses)==plan['target_bytes'],'Independent byte expansion differs')
    for width in (4096,8192,16384):
     oracle=[math.fsum(byte_losses[i:i+width]) for i in range(0,len(byte_losses),width)];actual=out['raw_byte_blocks'][str(width)]['loss_sums'];require(len(oracle)==len(actual) and max(abs(x-y) for x,y in zip(oracle,actual))<1e-8,'Independent grouped loss differs')
    require(abs(out['bits_per_byte']-math.fsum(losses)/(math.log(2)*len(byte_losses)))<1e-12,'Independent BPB differs')
    for key,new in [('target_sha256','0'*64),('target_bytes',plan['target_bytes']+1),('prefix_bytes',plan['prefix_bytes']+1)]:
     bad=copy.deepcopy(plan);bad[key]=new;rejects(lambda bad=bad:validate(codec,data,bad));cases+=1
    for bad in [losses[:-1],losses+[1.],[float('nan')]*len(losses),[-1.]*len(losses)]:rejects(lambda bad=bad:loss_summary(plan,bad));cases+=1
    if data==val:reports.append(dict(vocab_size=size,target_sha256=plan['target_sha256'],target_bytes=plan['target_bytes'],target_tokens=plan['target_tokens'],prefix_bytes=plan['prefix_bytes'],context_policy=plan['context']))
  require(len({r['target_sha256'] for r in reports})==len({r['target_bytes'] for r in reports})==1,'Cross-tokenizer target bytes differ')
  sources={str(p.relative_to(ROOT)):reference(p)['sha256'] for p in sorted((ROOT/'research/tokenizer_common_bytes_v1').glob('*.py'))};require(len(sources)==2,'Exact additive metric source set required')
  result=dict(status='passed_common_byte_metric_contracts',source=reference(__file__),codec_sources=census['codec_sources'],metric_sources=sources,source_bindings_total=976,proof_cases_rejected=cases,
   independent_grouping_atol=1e-8,independent_bpb_atol=1e-12,reports=reports,codec_census=reference(E/'result.json'),no_model_scoring=True,no_training=True,no_new_scientific_seed=True,holdout_encoded_or_scored=False,
   scope='Evaluation metric/coverage contract only;no candidate size/training horizon/resource/quality panel frozen. Equal target bytes do not make training exposure or raw context equal. A fresh native baseline is required;old CE/H200 scores cannot be pooled into this new target boundary.')
  save(E/'metric-contracts001.json',result);print(json.dumps(result),flush=True)
