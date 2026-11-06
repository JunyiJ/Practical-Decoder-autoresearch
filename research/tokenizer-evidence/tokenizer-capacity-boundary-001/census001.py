"""Existing-data tokenizer capacity census and codec contracts; no LM training."""
import collections,copy,hashlib,json,statistics
from pathlib import Path
import torch,tiktoken
from research.gpt2_prefix_tokenizer_v1.codec import Codec,SIZES
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent

def publish(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,sort_keys=True);f.write('\n')
def rejects(fn):
 try:fn()
 except (ValueError,UnicodeError):return
 raise AssertionError('Invalid codec contract accepted')
with session_lock(GLOBAL_LOCK.parent):
 torch.set_num_threads(4);idle();env=environment();require(not(E/'result.json').exists() and not(E/'plan.json').exists(),'One new train/validation census only')
 with CpuPrerequisiteGuard(E/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
  old=json.loads((ROOT/'research/tokenizer-evidence/embedding-input-dropout-preparation-001/quality001-closure001.json').read_text());require(old['status']=='CLOSED_NOT_ADOPTED','Actual closed latest panel required')
  pins={**old['sources'],**old['pilot_sources'],**old['quality_sources']};require(len(pins)==972,'Original972sources required')
  for n,h in pins.items():require(reference(ROOT/n)['sha256']==h,'Frozen source changed')
  raw=(ROOT/'data/raw/shakespeare.txt').read_bytes();require(hashlib.sha256(raw).hexdigest()=='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed','Original corpus required');text=raw.decode('utf8');a=int(.8*len(text));b=int((.8+.1)*len(text));parts={'train':text[:a],'validation':text[a:b]};del text,raw
  sources={str(p.relative_to(ROOT)):reference(p)['sha256'] for p in sorted((ROOT/'research/gpt2_prefix_tokenizer_v1').glob('*.py'))};require(len(sources)==2,'Finite additive codec namespace required')
  plan=dict(source=reference(__file__),environment=env,old_sources=pins,new_codec_sources=sources,diagnostic_sizes=list(SIZES),new_tokenizer_fitting=False,tokenizer_policy='Pretrained GPT2 merge-rank prefix,not a newly learned corpus BPE;all256byte primitives retained. Same regex,no normalization,no special-marker interpretation.',
   question='Quantify the vocabulary-capacity versus sequence/raw-context tradeoff before admitting a tokenizer/model study. Intrinsic compression/coverage is not a quality selector.',
   split_contract='Original raw80/10/10 boundaries;encode only train and originalvalidation,never holdout.',
   future_science_requirements='No cross-tokenizer CE comparison. Need identical predeclared raw target bytes and explicit context/compute/data-exposure controls,matched initialization/optimizer rules,equal finite tuning,fresh confirmation and full resource forecast. No scientific seed/model/rate/size selected here.')
  publish(E/'plan.json',plan);fixtures=['','Hello world.','a\t  b\r\n\n café 🦊 中文','<|endoftext|>','\x00'+''.join(chr(i) for i in range(256))];reports=[];forgeries=0
  for size in SIZES:
   c=Codec(size);desc=c.descriptor();require(Codec.from_descriptor(json.loads(json.dumps(desc))).descriptor()==desc,'Descriptor roundtrip differs')
   for value in fixtures:
    ids=c.encode(value);require(c.decode(ids)==value and c.decode_bytes(ids)==value.encode('utf8'),'Exact raw bytes not preserved')
    if size==50257:require(ids==tiktoken.get_encoding('gpt2').encode_ordinary(value),'Full codec differs from incumbent')
   for key,new in [('policy','foreign'),('version','changed'),('mergeable_ranks_sha256','0'*64),('normalization','nfkc'),('special_tokens','interpret_markers')]:
    bad=copy.deepcopy(desc);bad[key]=new;rejects(lambda bad=bad:Codec.from_descriptor(bad));forgeries+=1
   for bad in [[-1],[size],[True],[1.5]]:rejects(lambda bad=bad:c.decode_bytes(bad));forgeries+=1
   rejects(lambda:c.encode('\ud800'));forgeries+=1
   train=c.encode(parts['train']);val=c.encode(parts['validation']);require(c.decode_bytes(train)==parts['train'].encode('utf8') and c.decode_bytes(val)==parts['validation'].encode('utf8'),'Original split roundtrip failed')
   if size==50257:require((len(train),len(val))==(267688,34279),'Original GPT2counts differ')
   counts=collections.Counter(train);unseen=collections.Counter(i for i in val[1:] if i not in counts);lengths=[len(c._engine.decode_single_token_bytes(i)) for i in train];spans=[sum(lengths[i:i+128]) for i in range(0,len(lengths)-127,128)]
   reports.append(dict(vocab_size=size,descriptor=desc,train_tokens=len(train),validation_tokens=len(val),train_bytes=len(parts['train'].encode('utf8')),validation_bytes=len(parts['validation'].encode('utf8')),train_observed_ids=len(counts),unused_train_vocabulary=size-len(counts),validation_target_unseen_ids=len(unseen),validation_target_unseen_tokens=sum(unseen.values()),validation_target_unseen_fraction=sum(unseen.values())/(len(val)-1),first_validation_token_bytes=len(c._engine.decode_single_token_bytes(val[0])),train_128token_byte_span_mean=statistics.mean(spans),train_128token_byte_span_min=min(spans),train_128token_byte_span_max=max(spans),fixed_width256_learned_parameters=2760832+2*size*256))
  baseline=next(v for v in reports if v['vocab_size']==50257)
  for v in reports:v['train_token_inflation_vs_gpt2']=v['train_tokens']/baseline['train_tokens'];v['parameter_ratio_vs_incumbent']=v['fixed_width256_learned_parameters']/28492416
  result=dict(status='completed_existing_GPT2_prefix_codec_census',plan=reference(E/'plan.json'),codec_sources=sources,original972sources_unchanged=True,total_additive_sources=974,codec_roundtrip_fixtures=len(fixtures)*len(SIZES),descriptor_and_id_forgeries_rejected=forgeries,
   reports=reports,no_model_or_tokenizer_training=True,new_scientific_seed=False,model_checkpoint_outputs=0,holdout_encoded_or_scored=False,no_quality_comparison_or_adoption=True,no_size_selected=True,
   interpretation='Existing external GPT2 merges truncated by rank;not corpus-adapted training. Smaller vocabulary changes segmentation,raw context span,init dimensions and target exposure. Counts/cost arithmetic alone establish no model improvement or fair cross-tokenizer score.')
  publish(E/'result.json',result);print(json.dumps(dict(status=result['status'],reports=[{k:v for k,v in r.items() if k!='descriptor'} for r in reports],forgeries=forgeries)),flush=True)
