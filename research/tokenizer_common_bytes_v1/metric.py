"""Explicit common bytes, forced prefix boundary, deterministic token coding.

This does not make context in raw bytes or training exposure equal. It defines
only comparable evaluation target coverage and loss accounting, not a study.
"""
import hashlib,math
POLICY='rsi_common_raw_byte_target_v1'
def target_plan(codec,text,prefix_limit=64):
 if type(text) is not str or type(prefix_limit) is not int or prefix_limit<1:raise ValueError('Text and positive fixed byte prefix required')
 raw=text.encode('utf8',errors='strict')
 if len(raw)<=prefix_limit:raise ValueError('Text must exceed fixed prefix')
 split=prefix_limit
 while True:
  try:prefix=raw[:split].decode('utf8',errors='strict');break
  except UnicodeDecodeError:split-=1
  if split==0:raise ValueError('No complete Unicode prefix')
 suffix=raw[split:].decode('utf8',errors='strict');context=codec.encode(prefix);targets=codec.encode(suffix)
 pieces=[codec._engine.decode_single_token_bytes(i) for i in targets]
 if not context or not targets or any(not b for b in pieces) or b''.join(pieces)!=raw[split:] or codec.decode_bytes(context)!=raw[:split]:raise ValueError('Codec does not preserve declared bytes')
 return dict(policy=POLICY,codec=codec.descriptor(),prefix_limit=prefix_limit,prefix_bytes=split,context_ids=context,target_ids=targets,target_token_byte_lengths=[len(b) for b in pieces],
  full_text_sha256=hashlib.sha256(raw).hexdigest(),target_sha256=hashlib.sha256(raw[split:]).hexdigest(),target_bytes=len(raw)-split,target_tokens=len(targets),
  segmentation='Encode prefix and suffix separately at the common declared Unicode boundary;not global re-encoding.',
  context='Disjoint target windows of at most128tokens,one immediately preceding context token,matching inherited reset geometry. Raw context differs with tokenizer.',holdout_scored=False)
def validate(codec,text,plan):
 if type(plan) is not dict or target_plan(codec,text,plan.get('prefix_limit'))!=plan:raise ValueError('Foreign/mutated target contract')
 return plan
def windows(plan):
 if plan.get('policy')!=POLICY:raise ValueError('Own byte target policy required')
 ids=plan['target_ids'];prior=plan['context_ids'][-1]
 for start in range(0,len(ids),128):
  target=ids[start:start+128];context=prior if start==0 else ids[start-1]
  yield [context,*target[:-1]],target,start

def loss_summary(plan,losses):
 if plan.get('policy')!=POLICY or len(losses)!=plan['target_tokens'] or any(type(x) not in (float,int) or not math.isfinite(x) or x<0 for x in losses):raise ValueError('Complete finite loss vector required')
 sizes=plan['target_token_byte_lengths']
 if len(sizes)!=len(losses) or sum(sizes)!=plan['target_bytes'] or any(type(n) is not int or n<1 for n in sizes):raise ValueError('Exact byte coverage required')
 total=math.fsum(losses);blocks={}
 # Spread each token's negative log likelihood uniformly over its raw bytes.
 # This is an explicit accounting convention, not conditional per-byte entropy.
 for width in (4096,8192,16384):
  sums=[0.]*((plan['target_bytes']+width-1)//width);counts=[min(width,plan['target_bytes']-s) for s in range(0,plan['target_bytes'],width)];offset=0
  for value,size in zip(losses,sizes):
   end=offset+size
   while offset<end:
    index=offset//width;count=min(end-offset,width-offset%width);sums[index]+=float(value)*count/size;offset+=count
  if abs(math.fsum(sums)-total)>max(1e-8,total*1e-12):raise ValueError('Loss conservation failed')
  blocks[str(width)]=dict(byte_counts=counts,loss_sums=sums)
 return dict(policy=POLICY,target_sha256=plan['target_sha256'],target_bytes=plan['target_bytes'],target_tokens=plan['target_tokens'],total_nats=total,bits_per_byte=total/(math.log(2)*plan['target_bytes']),
  raw_byte_blocks=blocks,interpretation='Likelihood of declared deterministic token coding;not marginalization over all tokenizations. Byte-block allocation is bookkeeping,not observed byte-level entropy. No CE-per-token cross-tokenizer comparison.')
