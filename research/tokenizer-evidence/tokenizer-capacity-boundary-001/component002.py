"""Native zero-control and8Kmatched-row component correctness;no quality training."""
import copy,gc,json,math
from pathlib import Path
import torch
import torch.nn.functional as F
from research.body_allocation_model_v1.contract import ROOT,require,model_config,config as original_config
from research.body_allocation_model_v1.model import GPT as Original
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.body_allocation_model_v1.evidence import reference,environment
from research.gpt2_prefix_model_v2.contract import config
from research.gpt2_prefix_model_v2.model import GPT,load_weights
from research.embedding_input_dropout_worker_v2.execution import Guard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent

def create(cls,cfg,seed,device):
 with torch.random.fork_rng(devices=[]):
  torch.random.default_generator.manual_seed(seed);m=cls(cfg);post=torch.random.get_rng_state().clone()
 return m.to(device),post
def same(a,b):
 if type(a) is torch.Tensor:require(type(b) is torch.Tensor and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.cpu(),b.cpu()),'Tensor mismatch')
 elif isinstance(a,dict):require(a.keys()==b.keys(),'Key mismatch');[same(a[k],b[k]) for k in a]
 elif isinstance(a,(tuple,list)):require(type(a)==type(b) and len(a)==len(b),'Sequence mismatch');[same(x,y) for x,y in zip(a,b)]
 else:require(a==b,'Scalar mismatch')
def rejected(fn):
 try:fn()
 except (ValueError,AssertionError,IndexError):return
 raise AssertionError('Foreign/invalid contract accepted')
def oracle(m,x,y,v):
 z=F.embedding(x,m.transformer.wte.weight)
 for block in m.transformer.h:z=block(z,None,0)
 logits=F.linear(m.transformer.ln_f(z),m.lm_head.weight[:v])/m.width_mult
 return logits,F.cross_entropy(logits.reshape(-1,v),y.reshape(-1))
def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment();require(not(E/'component002.json').exists() and not(E/'component002-plan.json').exists(),'One new finite component proof only')
 with Guard(E/'component002-guard.json',600):
  codec=json.loads((E/'result.json').read_text());metric=json.loads((E/'metric-contracts001.json').read_text());require(codec['status']=='completed_existing_GPT2_prefix_codec_census' and metric['status']=='passed_common_byte_metric_contracts','Actual codec/metric prerequisites required')
  pins={**codec['codec_sources'],**metric['metric_sources']};old=json.loads((ROOT/'research/tokenizer-evidence/embedding-input-dropout-preparation-001/quality001-closure001.json').read_text());pins.update({**old['sources'],**old['pilot_sources'],**old['quality_sources']})
  for ns in ['gpt2_prefix_model_v1','gpt2_prefix_model_v2']:
   for p in (ROOT/'research'/ns).glob('*.py'):pins[str(p.relative_to(ROOT))]=reference(p)['sha256']
  require(len(pins)==982,'Exact additive source inventory required')
  failed=json.loads((E/'component001-guard.json').read_text());require(failed['status']=='fixture_exception' and not(E/'component001.json').exists(),'Original failed proof remains closed')
  for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Proven source changed')
  plan=dict(source=reference(__file__),sources=pins,environment=env,kind='component_feasibility_not_scientific_panel',candidate='prefix8192',anchor='Largest power-of-two below10653train-observed GPT2IDs;one structural feasibility anchor,not an optimum or validation-quality selection. Other sizes were intrinsic census controls.',
   contracts='Native initialization/RNG/logits/CE/gradients/two optimizer steps exact;8Krow/body bridge and independent sliced-readout equations/gradients;fresh-repeat/own checkpoint/foreign optimizer and caches;CPU/MPS causality/full-cache-reset batches1/8. No corpus training or quality seeds.',
   fixture_seed=600420,prior_closed_failure=[reference(E/n) for n in ['component001.py','component001.log','component001-guard.json','component001-plan.json']],fix='Separate explicit8192x256MPSreduction;original50257guard and allv1sources unchanged.',resource_seconds=600,scientific_panel_frozen=False)
  save(E/'component002-plan.json',plan);reports=[]
  for device in ['cpu','mps']:
   old_model,p=create(Original,model_config('incumbent'),600420,device);native,q=create(GPT,config('native'),600420,device);same(p,q);same(old_model.state_dict(),native.state_dict());cfg=original_config('incumbent',steps=4,seed=600420,lr=.00125);oa=build_optimizer(old_model,cfg);ob=build_optimizer(native,cfg)
   x=(torch.arange(16).reshape(2,8)%7+1).to(device);y=(x+3)%79
   for step in range(2):
    old_model.train();native.train();la,ca=old_model(x,y);lb,cb=native(x,y);same(la,lb);same(ca,cb);oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True);ca.backward();cb.backward()
    for (na,a),(nb,b) in zip(old_model.named_parameters(),native.named_parameters()):require(na==nb,'Parameter name order differs');same(a.grad,b.grad)
    oa.step();ob.step();same(old_model.state_dict(),native.state_dict())
    for a,b in zip(oa.optimizers,ob.optimizers):same(a.state_dict(),b.state_dict())
   del old_model,native,oa,ob,la,lb,ca,cb;gc.collect()
   if device=='mps':torch.mps.empty_cache()
   original,p=create(Original,model_config('incumbent'),600420,device);small,q=create(GPT,config('prefix8192'),600420,device);same(p,q)
   for name,t in small.state_dict().items():same(t,original.state_dict()[name][:8192] if name in ('transformer.wte.weight','lm_head.weight') else original.state_dict()[name])
   original.train();small.train();expected,loss=oracle(original,x,y,8192);actual,ce=small(x,y);torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-5);torch.testing.assert_close(ce,loss,rtol=1e-6,atol=1e-6);loss.backward();ce.backward()
   for name,t in small.named_parameters():
    g=dict(original.named_parameters())[name].grad;g=g[:8192] if name in ('transformer.wte.weight','lm_head.weight') else g;torch.testing.assert_close(t.grad,g,rtol=5e-4,atol=5e-5)
   o=build_optimizer(small,cfg);require(len(o.optimizers[0].param_groups)==44 and len(o.optimizers[1].param_groups)==44,'Genuine inherited optimizer coverage differs');require(sum(t.numel() for t in small.parameters())==6955136,'Exact8Kinventory differs');o.zero_grad(set_to_none=True);del original,expected,loss,actual,ce;gc.collect()
   cache_reports=[];small.eval()
   with torch.no_grad():
    for batch in [1,8]:
     tokens=(torch.arange(batch*128).reshape(batch,128)%503+1).to(device);full=small(tokens)[0];c=small.caches(batch)
     parts=[]
     for start,end in [(0,31),(31,96),(96,128)]:parts.append(small(tokens[:,start:end],cache=c,start_pos=start)[0])
     cached=torch.cat(parts,1);torch.testing.assert_close(full,cached,rtol=3e-4,atol=3e-5)
     for item in c:item.reset()
     altered=tokens.clone();altered[:,64:]=(altered[:,64:]+19)%8192;torch.testing.assert_close(small(altered)[0][:,:64],full[:,:64],rtol=0,atol=0)
     torch.testing.assert_close(small(tokens,cache=c)[0],full,rtol=3e-4,atol=3e-5);positions=[i.position for i in c];rejected(lambda:small(tokens[:,:1],cache=c,start_pos=128));require([i.position for i in c]==positions,'Rejected overflow mutated cache')
     bytes_item=sum(t.numel()*t.element_size() for i in c for t in i.buffers())//batch;require(bytes_item==196608,'Cache allocation differs');cache_reports.append(dict(batch=batch,cache_bytes_per_item=bytes_item,max_absolute_error=float((full-cached).abs().max())));del full,c,cached,parts,altered,tokens
    native,_=create(GPT,config('native'),600420,device);foreign=native.caches(2);rejected(lambda:small(x,cache=foreign));del native,foreign
   own=dict(kind='h200_gpt2_prefix_model_v2',architecture=copy.deepcopy(small.signature),model_state={n:t.detach().cpu().clone() for n,t in small.state_dict().items()});clone,_=create(GPT,config('prefix8192'),600420,device);load_weights(clone,own);same(small.state_dict(),clone.state_dict());bad=dict(own,kind='h200_embedding_input_dropout_v1');rejected(lambda:load_weights(clone,bad));bad=copy.deepcopy(own);bad['architecture']['vocab_size']=50257;rejected(lambda:load_weights(clone,bad));wrong=copy.deepcopy(o.state_dict());wrong['body_allocation_signature']=config('native');rejected(lambda:o.load_state_dict(wrong));rejected(lambda:small(torch.full((1,1),8192,dtype=torch.int64,device=device)))
   reports.append(dict(device=device,native_exact_initial_rng_logits_gradients_two_optimizer_steps=True,prefix_matched_body_rows_and_rng=True,prefix_independent_equations_and_gradients=True,own_checkpoint_and_foreign_rejections=True,groups=[44,44],parameter_tensors=88,parameters=6955136,cache_and_causality=cache_reports))
   del small,clone,o,own,bad,wrong;gc.collect()
   if device=='mps':torch.mps.empty_cache()
  for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Source changed during proof')
  result=dict(status='passed_native_and8K_component_correctness',plan=reference(E/'component002-plan.json'),source_bindings=pins,total_source_bindings=982,reports=reports,no_real_data_training=True,no_scientific_seed_assigned=True,weight_files=0,holdout_encoded_or_scored=False,
   limitations='Synthetic component steps only;real-data repeat/resume with EMA/RNG/sampler/clipping/serialization,worker full common-byte evaluation acceptance,resource pilots,primary resource-axis decision and complete finite scientific contract remain REQUIRED. No quality/model/tokenizer adoption.')
  save(E/'component002.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_bindings']}),flush=True)
