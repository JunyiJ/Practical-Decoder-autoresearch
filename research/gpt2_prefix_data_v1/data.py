"""Explicit native/8K data contract, private inherited512-token sampling math."""
import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require,plain,canonical,config as original_config
from research.gpt2_prefix_model_v2.contract import config as model_config,signature
from research.gpt2_prefix_tokenizer_v1.codec import Codec
from research.tokenizer_common_bytes_v1.metric import target_plan,validate
from research.block_norm_v1.runtime.data.loaders import _get_batch
from research.final_ln_scale_model_v1.runtime.models.output_bias import digest

POLICY='rsi_gpt2_prefix_training_data_v1'
def config(kind,steps=4,seed=600422,lr=.00125):
 old=plain(original_config('incumbent',steps=steps,seed=seed,lr=lr));backbone=old.pop('model');old['backbone_template']=backbone;old['prefix_model']=model_config(kind);old['data']['tokenizer']=POLICY
 # No fixture may accidentally write into a closed historical run.
 old['training'].update(checkpoint_dir=str(ROOT/'research_runs/UNLAUNCHED-gpt2-prefix-fixture/checkpoints'),metrics_file=str(ROOT/'research_runs/UNLAUNCHED-gpt2-prefix-fixture/metrics.jsonl'),summary_file=str(ROOT/'research_runs/UNLAUNCHED-gpt2-prefix-fixture/summary.json'))
 return OmegaConf.create(old)
def architecture(cfg):
 reg=plain(cfg.prefix_model);require(canonical(cfg.backbone_template)==canonical(reg['backbone_template']),'Genuine native initialization template required');require('model' not in cfg,'Explicit prefix policy must not masquerade as an old model config');return signature(reg)
class View(SimpleNamespace):
 def get_batch(self,split='train'):return _get_batch(self,split)
def load(cfg,device='cpu'):
 sig=architecture(cfg);require(cfg.data.tokenizer==POLICY and cfg.data.train_fraction==.8 and cfg.data.val_fraction==.1 and cfg.training.batch_size==8 and cfg.training.batch_layout=='paired_blocks512','Exact source/split/sampler policy required')
 require(device in ('cpu','mps'),'Declared local device required');path=Path(cfg.data.path);raw=path.read_bytes();h=hashlib.sha256(raw).hexdigest();require(h=='86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed','Original corpus required');text=raw.decode('utf8');a=int(.8*len(text));b=int((.8+.1)*len(text));codec=Codec.from_descriptor(plain(cfg.prefix_model.tokenizer))
 values={'train':codec.encode(text[:a]),'val':codec.encode(text[a:b])};common=target_plan(codec,text[a:b]);validate(codec,text[a:b],common)
 # Actual count is bound by the earlier complete census, not a guessed constant.
 census=json.loads((ROOT/'research/tokenizer-evidence/tokenizer-capacity-boundary-001/result.json').read_text());row=next(r for r in census['reports'] if r['vocab_size']==sig['vocab_size']);require((len(values['train']),len(values['val']))==(row['train_tokens'],row['validation_tokens']),'Actual census IDs differ')
 if sig['vocab_size']==50257:require((len(values['train']),len(values['val']))==(267688,34279),'Native data bridge differs')
 tensors={n:torch.tensor(v,dtype=torch.int64) for n,v in values.items()};require(all(bool((t>=0).all()) and bool((t<sig['vocab_size']).all()) for t in tensors.values()),'ID outside effective vocabulary')
 require(hashlib.sha256(path.read_bytes()).hexdigest()==h,'Corpus changed')
 identity=dict(policy=POLICY,corpus_sha256=h,raw_characters=len(text),raw_train_end=a,raw_val_end=b,tokenizer=codec.descriptor(),train_tokens=len(values['train']),validation_tokens=len(values['val']),train_ids_sha256=digest(values['train']),validation_ids_sha256=digest(values['val']),common_target_sha256=common['target_sha256'],common_target_bytes=common['target_bytes'],common_target_tokens=common['target_tokens'],common_target_ids_sha256=digest(common['target_ids']),split_policy='Original raw80/10/10,each permitted split encoded separately;holdout not encoded.',exposure_scope='Same scalar sampler seeds/1024targets per update,not identical raw examples/bytes or equal compute across codecs.')
 return View(train_data=tensors['train'],val_data=tensors['val'],splits=tensors,generators={n:torch.Generator().manual_seed(s) for n,s in [('train',cfg.training.seed),('val',cfg.training.eval_seed)]},device=device,batch_size=8,block_size=128,batch_layout='paired_blocks512',vocab_size=sig['vocab_size'],data_identity=identity,common_byte_plan=common)
def restore_samplers(view,states):
 require(type(states) is dict and set(states)=={'train','val'},'Exact private sampler set required');probes={}
 for name,t in states.items():
  template=view.generators[name].get_state();require(type(t) is torch.Tensor and t.dtype==torch.uint8 and t.device.type=='cpu' and t.is_contiguous() and t.shape==template.shape,'Exact CPU sampler RNG layout required');probe=torch.Generator();probe.set_state(t.clone());probes[name]=probe.get_state()
 for name,t in probes.items():view.generators[name].set_state(t)
