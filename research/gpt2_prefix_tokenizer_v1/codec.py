"""Lossless fixed GPT2 byte-BPE rank prefixes, explicitly distinct identities."""
import copy,hashlib,importlib.metadata,json
import tiktoken
POLICY='rsi_gpt2_merge_prefix_diagnostic_v1';SIZES=(4096,8192,16384,50257)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def vocab_hash(ranks):return digest(sorted((k.hex(),v) for k,v in ranks.items()))
class Codec:
 def __init__(self,size):
  if type(size) is not int or size not in SIZES:raise ValueError('Declared diagnostic size required')
  base=tiktoken.get_encoding('gpt2');self.size=size
  if size==50257:self._engine=base
  else:
   ranks={k:v for k,v in base._mergeable_ranks.items() if v<size}
   if set(ranks.values())!=set(range(size)) or not all(bytes([i]) in ranks for i in range(256)):raise ValueError('Dense byte-complete prefix required')
   self._engine=tiktoken.Encoding(name=f'{POLICY}_{size}',pat_str=base._pat_str,mergeable_ranks=ranks,special_tokens={})
  self._identity=dict(policy=POLICY,kind='full_pretrained_gpt2' if size==50257 else 'pretrained_gpt2_merge_rank_prefix',vocab_size=size,library='tiktoken',version=importlib.metadata.version('tiktoken'),pattern=base._pat_str,mergeable_ranks_sha256=vocab_hash(self._engine._mergeable_ranks),base_mergeable_ranks_sha256=vocab_hash(base._mergeable_ranks),special_tokens='ordinary_literal_text_only',normalization='none',new_tokenizer_training=False)
  self.validate()
 def validate(self):
  e=self._engine
  if type(e) is not tiktoken.Encoding or e.n_vocab!=self.size or e._pat_str!=self._identity['pattern'] or vocab_hash(e._mergeable_ranks)!=self._identity['mergeable_ranks_sha256'] or importlib.metadata.version('tiktoken')!=self._identity['version']:raise ValueError('Codec identity changed')
 def descriptor(self):self.validate();return copy.deepcopy(self._identity)
 def encode(self,text):
  self.validate()
  if type(text) is not str:raise ValueError('Unicode text required')
  text.encode('utf8',errors='strict') # Never silently repair unpaired surrogates.
  ids=self._engine.encode_ordinary(text)
  if any(type(i) is not int or not 0<=i<self.size for i in ids):raise ValueError('ID outside declared vocabulary')
  return ids
 def decode_bytes(self,ids):
  self.validate()
  if type(ids) not in (list,tuple) or any(type(i) is not int or not 0<=i<self.size for i in ids):raise ValueError('Dense integer IDs required')
  return self._engine.decode_bytes(list(ids))
 def decode(self,ids):return self.decode_bytes(ids).decode('utf8',errors='strict')
 def token_bytes(self,idx):
  self.validate()
  if type(idx) is not int or not 0<=idx<self.size:raise ValueError('Integer ID required')
  return self._engine.decode_single_token_bytes(idx)
 @classmethod
 def from_descriptor(cls,value):
  if type(value) is not dict or value.get('policy')!=POLICY:raise ValueError('Own explicit policy required')
  candidate=cls(value.get('vocab_size'))
  if candidate.descriptor()!=value:raise ValueError('Foreign/mutated codec descriptor')
  return candidate
