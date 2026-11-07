"""Native train/validation slices; the holdout tail is never encoded."""
import copy,hashlib,importlib.metadata,json
from pathlib import Path
from types import SimpleNamespace
import tiktoken,torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require,plain,canonical,config as native_config
from research.tokenizer_mla_gate_model_v1.contract import model_config,signature
from research.block_norm_v1.runtime.data.loaders import _get_batch
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest
from research.embedding_input_dropout_resume_guard_v1.checkpoint import cpu_state

DATA=ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json'

def config(arm,steps,seed,eval_seed,lr=.00125):
    require(type(eval_seed) is int and eval_seed>=0,'Explicit private evaluation role required')
    cfg=plain(native_config('incumbent',steps,seed,lr));cfg['backbone_template']=cfg.pop('model');cfg['gate_model']=model_config(arm)
    cfg['training'].update(eval_seed=eval_seed,checkpoint_dir=str(ROOT/'research_runs/UNLAUNCHED-mla-gate-fixture/checkpoints'),metrics_file=str(ROOT/'research_runs/UNLAUNCHED-mla-gate-fixture/metrics.jsonl'),summary_file=str(ROOT/'research_runs/UNLAUNCHED-mla-gate-fixture/summary.json'))
    return OmegaConf.create(cfg)

def architecture(cfg):
    require('model' not in cfg and 'moe_model' not in cfg,'Own gate policy cannot masquerade as legacy/MoE config')
    require(canonical(cfg.backbone_template)==canonical(cfg.gate_model.backbone_template),'Exact genuine native backbone required')
    return signature(cfg.gate_model)

def identity(cfg):
    architecture(cfg)
    require(cfg.data.tokenizer=='gpt2' and cfg.data.train_fraction==.8 and cfg.data.val_fraction==.1 and cfg.training.batch_size==8 and cfg.training.batch_layout=='paired_blocks512','Native codec/splits/paired sampling geometry required')
    expected=json.loads(DATA.read_text())['data']
    require(hashlib.sha256(Path(cfg.data.path).read_bytes()).hexdigest()==expected['corpus_sha256'],'Bound original corpus required')
    return copy.deepcopy(expected)

class View(SimpleNamespace):
    def get_batch(self,split='train'):return _get_batch(self,split)

def load(cfg,device='cpu'):
    expected=identity(cfg);require(device in ('cpu','mps'),'Explicit CPU/MPS data target required')
    raw=Path(cfg.data.path).read_bytes();text=raw.decode('utf8');a=int(.8*len(text));b=int((.8+.1)*len(text))
    require((len(text),a,b)==(expected['raw_characters'],expected['raw_train_end'],expected['raw_val_end']),'Original raw split boundaries required')
    enc=tiktoken.get_encoding('gpt2')
    tokenizer=dict(name='gpt2',implementation='tiktoken',version=importlib.metadata.version('tiktoken'),vocabulary_sha256=hashlib.sha256(json.dumps(sorted((token.hex(),rank) for token,rank in enc._mergeable_ranks.items())).encode()).hexdigest(),pattern=enc._pat_str,special_tokens=dict(enc._special_tokens),special_token_policy='ordinary_text',split_unit='raw_text',vocab_size=enc.n_vocab)
    require(tokenizer==expected['tokenizer'],'Original codec/library contract required')
    ids={n:enc.encode(t,disallowed_special=()) for n,t in [('train',text[:a]),('val',text[a:b])]}
    require(len(ids['train'])==267688 and len(ids['val'])==34279 and digest(ids['train'])==expected['train_ids_sha256'] and digest(ids['val'])==expected['validation_ids_sha256'] and digest(ids['val'][1:])==expected['validation_target_ids_sha256'],'Exact original native IDs required')
    require(hashlib.sha256(Path(cfg.data.path).read_bytes()).hexdigest()==expected['corpus_sha256'],'Corpus changed during load')
    tensors={n:torch.tensor(v,dtype=torch.int64) for n,v in ids.items()}
    return View(train_data=tensors['train'],val_data=tensors['val'],splits=tensors,generators={n:torch.Generator(device='cpu').manual_seed(s) for n,s in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]},batch_size=8,block_size=128,batch_layout='paired_blocks512',vocab_size=50257,device=device,data_identity=expected,tokenizer_contract=tokenizer)

def restore_samplers(view,states):
    require(type(states) is dict and set(states)==set(view.generators)=={'train','val'},'Exact private sampler inventory required')
    for n,v in states.items():cpu_state(v);require(v.shape==view.generators[n].get_state().shape,'Sampler shape differs')
    for n,v in states.items():view.generators[n].set_state(v)
