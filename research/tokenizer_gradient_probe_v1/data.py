"""Exact native training prefix only; no validation/holdout encoding or generator."""
import hashlib,importlib.metadata,json
from pathlib import Path
from types import SimpleNamespace
import torch,tiktoken
from research.body_allocation_model_v1.contract import require
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest
from research.block_norm_v1.runtime.data.loaders import _get_batch
from research.tokenizer_moe_model_v1.contract import signature

class TrainingView(SimpleNamespace):
    def next_train(self):
        require(self.batch_size==8 and self.block_size==128 and self.batch_layout=='paired_blocks512' and set(self.splits)==set(self.generators)=={'train'},'Exact training-only sampler inventory required')
        ids=self.train_data
        require(ids.device.type=='cpu' and ids.dtype==torch.int64 and ids.shape==(267688,) and ids.is_contiguous() and self.splits['train'] is ids,'Native CPU training IDs required')
        require(digest(ids.tolist())==self.identity['train_ids_sha256'],'Training IDs changed before sampler mutation')
        return _get_batch(self,'train')

def load_train(worker,device,seed):
    cfg=worker['config'];require(type(seed) is int and seed>=0 and device in ('cpu','mps'),'Explicit private fixture seed/device required')
    require('model' not in cfg and signature(cfg['moe_model'])['arm']=='dense' and cfg['backbone_template']==cfg['moe_model']['backbone_template'],'Genuine native dense own configuration required')
    require(cfg['data']['tokenizer']=='gpt2' and cfg['data']['train_fraction']==.8 and cfg['training']['batch_size']==8 and cfg['training']['batch_layout']=='paired_blocks512','Original native training geometry required')
    expected=worker['data'];p=Path(cfg['data']['path']);raw=p.read_bytes();corpus=hashlib.sha256(raw).hexdigest();require(corpus==expected['corpus_sha256'],'Original corpus identity required')
    text=raw.decode('utf8');end=int(.8*len(text));require(len(text)==expected['raw_characters'] and end==expected['raw_train_end'],'Original raw train boundary required')
    enc=tiktoken.get_encoding('gpt2');tokenizer=dict(name='gpt2',implementation='tiktoken',version=importlib.metadata.version('tiktoken'),vocabulary_sha256=hashlib.sha256(json.dumps(sorted((token.hex(),rank) for token,rank in enc._mergeable_ranks.items())).encode()).hexdigest(),pattern=enc._pat_str,special_tokens=dict(enc._special_tokens),special_token_policy='ordinary_text',split_unit='raw_text',vocab_size=enc.n_vocab)
    require(tokenizer==expected['tokenizer'],'Original tokenizer/library identity required')
    values=enc.encode(text[:end],disallowed_special=());require(len(values)==267688 and digest(values)==expected['train_ids_sha256'],'Original training IDs required')
    require(hashlib.sha256(p.read_bytes()).hexdigest()==corpus,'Corpus changed during load')
    ids=torch.tensor(values,dtype=torch.int64);identity=dict(corpus_sha256=corpus,raw_characters=len(text),raw_train_end=end,train_ids_sha256=expected['train_ids_sha256'],tokenizer=tokenizer,validation_encoded=False,holdout_encoded=False)
    return TrainingView(train_data=ids,splits={'train':ids},generators={'train':torch.Generator().manual_seed(seed)},batch_size=8,block_size=128,batch_layout='paired_blocks512',vocab_size=50257,device=device,identity=identity)
