import copy
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace
import tiktoken
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require,plain,canonical,config as native_config
from research.tokenizer_moe_model_v1.contract import model_config,signature
from research.block_norm_v1.runtime.data.loaders import _get_batch
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest
from research.embedding_input_dropout_resume_guard_v1.checkpoint import cpu_state


def config(arm,steps=4,seed=600460,eval_seed=600461,lr=.00125):
    require(type(eval_seed) is int and eval_seed>=0,'Explicit evaluation fixture seed required')
    cfg=plain(native_config('incumbent',steps,seed,lr));backbone=cfg.pop('model')
    cfg['backbone_template']=backbone;cfg['moe_model']=model_config(arm)
    cfg['training'].update(eval_seed=eval_seed,checkpoint_dir=str(ROOT/'research_runs/UNLAUNCHED-moe-fixture/checkpoints'),
        metrics_file=str(ROOT/'research_runs/UNLAUNCHED-moe-fixture/metrics.jsonl'),summary_file=str(ROOT/'research_runs/UNLAUNCHED-moe-fixture/summary.json'))
    return OmegaConf.create(cfg)


def architecture(cfg):
    require('model' not in cfg,'Own model configuration must not masquerade as legacy cfg.model')
    require(canonical(cfg.backbone_template)==canonical(cfg.moe_model.backbone_template),'Genuine shared native backbone required')
    return signature(cfg.moe_model)


class View(SimpleNamespace):
    def get_batch(self,split='train'):return _get_batch(self,split)


def load(cfg,device='cpu'):
    architecture(cfg)
    require(device in ('cpu','mps') and cfg.data.tokenizer=='gpt2' and cfg.data.train_fraction==.8 and cfg.data.val_fraction==.1,'Native source/splits/device required')
    require(cfg.training.batch_size==8 and cfg.training.batch_layout=='paired_blocks512','Native sampling geometry required')
    expected=json.loads((ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json').read_text())['data']
    path=Path(cfg.data.path);raw=path.read_bytes();before=hashlib.sha256(raw).hexdigest()
    require(before==expected['corpus_sha256'],'Original corpus identity required')
    text=raw.decode('utf8');a=int(.8*len(text));b=int((.8+.1)*len(text))
    require((len(text),a,b)==(expected['raw_characters'],expected['raw_train_end'],expected['raw_val_end']),'Original raw boundaries required')
    enc=tiktoken.get_encoding('gpt2')
    tokenizer=dict(name='gpt2',implementation='tiktoken',version=importlib.metadata.version('tiktoken'),
        vocabulary_sha256=hashlib.sha256(json.dumps(sorted((token.hex(),rank) for token,rank in enc._mergeable_ranks.items())).encode()).hexdigest(),
        pattern=enc._pat_str,special_tokens=dict(enc._special_tokens),special_token_policy='ordinary_text',split_unit='raw_text',vocab_size=enc.n_vocab)
    require(tokenizer==expected['tokenizer'],'Original tokenizer identity required')
    # Deliberately encode only the authorized two slices, never a holdout tail.
    ids={n:enc.encode(t,disallowed_special=()) for n,t in [('train',text[:a]),('val',text[a:b])]}
    require(len(ids['train'])==267688 and len(ids['val'])==34279 and digest(ids['train'])==expected['train_ids_sha256']
        and digest(ids['val'])==expected['validation_ids_sha256'] and digest(ids['val'][1:])==expected['validation_target_ids_sha256'],'Original native IDs required')
    tensors={n:torch.tensor(v,dtype=torch.int64) for n,v in ids.items()}
    require(hashlib.sha256(path.read_bytes()).hexdigest()==before,'Corpus changed during data load')
    return View(train_data=tensors['train'],val_data=tensors['val'],splits=tensors,
                generators={n:torch.Generator().manual_seed(s) for n,s in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]},
                batch_size=8,block_size=128,batch_layout='paired_blocks512',vocab_size=50257,device=device,
                data_identity=copy.deepcopy(expected),tokenizer_contract=tokenizer)


def restore_samplers(view,states):
    require(type(states) is dict and set(states)=={'train','val'},'Exact private sampler inventory required')
    for name,value in states.items():
        cpu_state(value);require(value.shape==view.generators[name].get_state().shape,'Sampler shape differs')
    for name,value in states.items():view.generators[name].set_state(value)
