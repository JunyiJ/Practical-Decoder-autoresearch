"""Original train/validation identities and sampler, with no holdout encoding."""
import hashlib,importlib.metadata,json
from types import SimpleNamespace
from pathlib import Path
import tiktoken,torch
from research.body_allocation_model_v1.contract import ROOT,require,signature
from research.block_norm_v1.runtime.data.loaders import _get_batch
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest

class View(SimpleNamespace):
    def get_batch(self,split='train'):return _get_batch(self,split)

def load(cfg,device='cpu'):
    signature(cfg.model)
    expected=json.loads((ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json').read_text())['data']
    path=Path(cfg.data.path);before=hashlib.sha256(path.read_bytes()).hexdigest()
    require(before==expected['corpus_sha256'] and cfg.data.tokenizer=='gpt2'
        and cfg.data.train_fraction==.8 and cfg.data.val_fraction==.1,'Frozen corpus/splits required')
    text=path.read_text();a=int(.8*len(text));b=int((.8+.1)*len(text))
    require((len(text),a,b)==(expected['raw_characters'],expected['raw_train_end'],expected['raw_val_end']),'Raw boundaries differ')
    enc=tiktoken.get_encoding('gpt2')
    tokenizer=dict(name='gpt2',implementation='tiktoken',version=importlib.metadata.version('tiktoken'),
        vocabulary_sha256=hashlib.sha256(json.dumps(sorted((token.hex(),rank) for token,rank in enc._mergeable_ranks.items())).encode()).hexdigest(),
        pattern=enc._pat_str,special_tokens=dict(enc._special_tokens),special_token_policy='ordinary_text',split_unit='raw_text',vocab_size=enc.n_vocab)
    require(tokenizer==expected['tokenizer'],'Tokenizer identity differs')
    # Only these two slices are encoded; no test-tail array, text slice or generator.
    values={name:enc.encode(part,disallowed_special=()) for name,part in [('train',text[:a]),('val',text[a:b])]}
    require(len(values['train'])==267688 and len(values['val'])==34279
        and digest(values['train'])==expected['train_ids_sha256']
        and digest(values['val'])==expected['validation_ids_sha256']
        and digest(values['val'][1:])==expected['validation_target_ids_sha256'],'Encoded split identity differs')
    tensors={n:torch.tensor(v,dtype=torch.int64) for n,v in values.items()}
    require(cfg.training.batch_size==8 and cfg.model.block_size==128 and cfg.training.batch_layout=='paired_blocks512','Fixed sampling geometry required')
    require(hashlib.sha256(path.read_bytes()).hexdigest()==before,'Corpus changed')
    return View(train_data=tensors['train'],val_data=tensors['val'],splits=tensors,
        generators={n:torch.Generator().manual_seed(s) for n,s in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]},
        device=device,batch_size=8,block_size=128,batch_layout='paired_blocks512',vocab_size=50257,
        train_fraction=.8,val_fraction=.1,eval_seed=cfg.training.eval_seed,tokenizer_contract=tokenizer,data_identity=expected)
