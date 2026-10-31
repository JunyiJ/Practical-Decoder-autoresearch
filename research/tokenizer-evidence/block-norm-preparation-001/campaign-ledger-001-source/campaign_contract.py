"""Read-only preparation contracts. No launcher, writes or readiness override."""
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import stat
import sys
from pathlib import Path
import tiktoken
import torch
import torch.optim._muon as muon
from . import campaign_rules as rules
from .spec import Spec, encode
from .runtime.models.block_norm import GROUPED_POLICY, contract as model_contract
from .runtime.models.grouped_embedding import EMBEDDING_SPEC
from .runtime.models.output_bias import digest as prior_digest

ROOT=Path(__file__).resolve().parents[2]
BASELINE='research/tokenizer-evidence/mla-001/adopted-width256-mla-recipe.json'
BASELINE_SHA256='14936864a1ef15aaa1e97c763755f0b090c0e31babdb5c27edc657417aa4f3d3'
RECOVERY_PLAN='research_runs/tokenizer-activation-inference-recovery-002/evidence/meta/recovery-plan.json'
RECOVERY_PLAN_SHA256='f28fe8194baf1fe4b05a1c480b4868e41a1b7b66fe7410e163e52f883edbb1b3'
ORIGINAL_SOURCE_SHA256='7a4c02946a31820a7b1ff36b8cc4be42659a4fc4a62424cbc4ea5f32d6cbc487'
SOURCE_ROOTS=('src','research/block_norm_v1','research/activation_inference_recovery_v1','research/activation_inference_recovery_v2')
LIMITS=dict(reserve_bytes=8*1024**3,rss_limit_bytes=4*1024**3,driver_limit_bytes=6*1024**3,
            pilot_seconds=600,quality_seconds=7200,inference_seconds=600,cpu_threads=4,device='mps',cpu_fallback=False,fast_math=False)
require=rules.require


def canonical(value):
    def check(v):
        require(type(v) in (dict,list,str,int,float,bool,type(None)), 'Native JSON types required')
        if type(v) is dict:
            require(all(type(k) is str for k in v), 'String JSON keys required')
            for item in v.values():check(item)
        elif type(v) is list:
            for item in v:check(item)
    check(value)
    try:return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)
    except (TypeError,ValueError) as exc:raise ValueError('Portable finite JSON required') from exc


def fingerprint(value):return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path,*,root=None,max_bytes=8*1024**2):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts,'Absolute normalized file path required')
    if root is not None:require(path.is_relative_to(root),'File outside bound root')
    for part in [path,*path.parents]:
        require(not part.is_symlink(),'Symlink in bound path')
        if root is not None and part==Path(root):break
    before=path.stat();require(stat.S_ISREG(before.st_mode) and before.st_nlink==1 and before.st_size<=max_bytes,'Regular unaliased bounded file required')
    out=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024**2),b''):out.update(chunk)
    after=path.stat()
    fields=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    require(fields(before)==fields(after),'File changed during hashing')
    return out.hexdigest()


def baseline():
    path=ROOT/BASELINE;require(file_sha(path,root=ROOT)==BASELINE_SHA256,'Adopted baseline changed')
    return json.loads(path.read_text())['config']


def _output(output,repo_root=ROOT):
    path=Path(output);require(path.is_absolute() and '..' not in path.parts,'Absolute normalized output required')
    require(path.is_relative_to(repo_root),'Output outside repository')
    parts=path.relative_to(repo_root).parts
    require(len(parts)==5 and parts[0]=='research_runs'
            and re.fullmatch(r'tokenizer-block-norm-v2-[a-z0-9][a-z0-9-]*',parts[1]) is not None
            and parts[2]=='evidence' and parts[3] in ('pilots','quality')
            and re.fullmatch(r'[0-9]{2}',parts[4]) is not None,'Dedicated new campaign slot required')
    for p in [path,*path.parents]:
        require(not p.is_symlink(),'Symlink in output path')
        if p==repo_root:break
    return path,parts[3],int(parts[4])


def trial_config(output,spec,seeds):
    rules.validate_seeds(seeds);path,stage,slot=_output(output)
    if stage=='pilots':
        candidates=rules.pilot_specs(seeds);require(slot<len(candidates),'Pilot slot outside finite plan')
        candidates=[candidates[slot]]
    elif slot<6:candidates=[rules.screen_specs(seeds)[slot]]
    else:
        require(slot<12,'Quality slot outside finite plan')
        candidates=[rules.confirmation_specs(seeds,dict(zip(rules.ARMS,(a,b))))[slot-6] for a in rules.RATES for b in rules.RATES]
    require(any(canonical(spec)==canonical(s) for s in candidates),'Spec differs from finite slot')
    cfg=copy.deepcopy(baseline());arm=spec['arm']
    cfg['model'].update(tokenizer_architecture_policy=GROUPED_POLICY,
        block_normalization=encode(Spec(arm)),embedding_backward=dict(EMBEDDING_SPEC),
        norm_type='layernorm' if arm=='layer_matched_epsilon' else 'rmsnorm')
    model_contract(cfg['model'])
    t=cfg['training'];pilot=spec['stage']=='pilot';start=13 if pilot else 800
    t.update(seed=spec['seed'],eval_seed=seeds['evaluation'],learning_rate=spec['lr'],max_iters=spec['steps'],
        device='mps',strict_device=True,resume_from=None,max_train_seconds=None,
        checkpoint_dir=str(path/'checkpoints'),metrics_file=str(path/'metrics.jsonl'),summary_file=str(path/'training-summary.json'),
        checkpoint_every=0,keep_best_checkpoint=False,eval_every=20 if pilot else 300)
    t['lr_schedule'].update(total_steps=spec['steps'],start_step=start)
    t['weight_averaging'].update(total_steps=spec['steps'],start_step=start,interval=1 if pilot else 10,half_lives=[200])
    return cfg


def validate_trial_config(cfg,output,spec,seeds):
    require(canonical(cfg)==canonical(trial_config(output,spec,seeds)),'Exact bound trial configuration required')


def bind_data():
    """Hash corpus bytes; tokenize train/validation only, never the test suffix."""
    cfg=baseline();path=Path(cfg['data']['path']);prior=cfg['model']['output_bias_prior']
    corpus_sha=file_sha(path,root=ROOT,max_bytes=4*1024**2)
    require(corpus_sha==prior['corpus_sha256'],'Corpus changed')
    text=path.read_text(encoding='utf-8');train_end=int(.8*len(text));val_end=int((.8+.1)*len(text))
    enc=tiktoken.get_encoding('gpt2')
    ranks=[(token.hex(),rank) for token,rank in enc._mergeable_ranks.items()]
    tokenizer=dict(name='gpt2',implementation='tiktoken',version=importlib.metadata.version('tiktoken'),
        vocabulary_sha256=hashlib.sha256(json.dumps(sorted(ranks)).encode()).hexdigest(),pattern=enc._pat_str,
        special_tokens=dict(enc._special_tokens),special_token_policy='ordinary_text',split_unit='raw_text',vocab_size=enc.n_vocab)
    require(canonical(tokenizer)==canonical(prior['tokenizer']),'Tokenizer implementation/vocabulary differs')
    train=enc.encode(text[:train_end],disallowed_special=())
    validation=enc.encode(text[train_end:val_end],disallowed_special=())
    body=dict(version=2,tokenizer=tokenizer,corpus_sha256=corpus_sha,train_fraction=.8,val_fraction=.1,train_ids_sha256=prior_digest(train))
    require(canonical(dict(body,prior_sha256=prior_digest(body)))==canonical(prior),'Training token/prior binding differs')
    count=len(validation)-1
    require(count>=8192 and (count+2047)//2048<=rules.MAX_BASE_BLOCKS,'Validation target panel outside bounded statistics')
    def counts(block):return [min(block,count-start) for start in range(0,count,block)]
    value=dict(path=str(path),corpus_sha256=corpus_sha,raw_characters=len(text),raw_train_end=train_end,raw_val_end=val_end,
        train_fraction=.8,val_fraction=.1,tokenizer=tokenizer,output_bias_prior=prior,
        train_tokens=len(train),train_ids_sha256=prior_digest(train),validation_tokens=len(validation),
        validation_ids_sha256=prior_digest(validation),validation_target_ids_sha256=prior_digest(validation[1:]),
        validation_targets=count,validation_target_bytes=sum(len(enc.decode_single_token_bytes(x)) for x in validation[1:]),
        validation_window_target_counts=counts(128),statistical_block_target_counts=counts(2048),
        evaluation='all validation targets except first, disjoint context128 windows, no padding; tail retained',
        holdout_tokenized=False,holdout_scored=False)
    require(file_sha(path,root=ROOT,max_bytes=4*1024**2)==corpus_sha,'Corpus changed during binding')
    return value


def source_inventory(repo_root=ROOT):
    root=Path(repo_root);require(root.is_absolute() and not root.is_symlink(),'Absolute unaliased source root required')
    paths=[root/'requirements.txt',root/BASELINE]
    for directory in SOURCE_ROOTS:
        base=root/directory;require(base.is_dir() and not base.is_symlink(),'Missing/unaliased source namespace required')
        for parent,dirs,files in os.walk(base,followlinks=False):
            require(not any((Path(parent)/n).is_symlink() for n in dirs+files),'Source symlink forbidden')
            dirs[:]=[n for n in dirs if n!='__pycache__']
            paths.extend(Path(parent)/n for n in files if n.endswith('.py'))
    return {str(p.relative_to(root)):file_sha(p,root=root) for p in sorted(paths)}


def verify_sources(expected,repo_root=ROOT):
    require(type(expected) is dict and canonical(expected)==canonical(source_inventory(repo_root)),'Source content or inventory changed')


def frozen_predecessors(sources):
    original={p:s for p,s in sources.items() if p.startswith('src/') or p=='requirements.txt'}
    historical=hashlib.sha256(json.dumps(original,sort_keys=True,allow_nan=False).encode()).hexdigest()
    require(historical==ORIGINAL_SOURCE_SHA256,'Frozen original source changed')
    path=ROOT/RECOVERY_PLAN;require(file_sha(path,root=ROOT)==RECOVERY_PLAN_SHA256,'Frozen recovery plan changed')
    external=json.loads(path.read_text())['external_source']
    require(len(external)==13 and all(sources.get(p)==s for p,s in external.items()),'Frozen recovery runtime changed')
    return dict(original_source_sha256=historical,recovery_plan_sha256=RECOVERY_PLAN_SHA256,recovery_modules=external)


def environment():
    variables={n:os.environ.get(n) for n in ['PYTORCH_ENABLE_MPS_FALLBACK','PYTORCH_MPS_FAST_MATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}
    require(variables==dict(PYTORCH_ENABLE_MPS_FALLBACK='0',PYTORCH_MPS_FAST_MATH='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4'),'Strict environment differs')
    require(torch.get_num_threads()==4,'Four torch CPU threads required')
    return dict(python=sys.version,executable=sys.executable,platform=platform.platform(),machine=platform.machine(),host=platform.node(),
        packages={n:importlib.metadata.version(n) for n in ['torch','numpy','omegaconf','hydra-core','tiktoken']},
        environment=variables,torch_threads=4,muon_source_path=muon.__file__,muon_source_sha256=file_sha(Path(muon.__file__)))


def preparation_manifest(seeds):
    rules.validate_seeds(seeds);sources=source_inventory()
    return dict(kind='block_norm_v2_preparation_contract_v1',launch_permitted=False,seeds=copy.deepcopy(seeds),
        seed_scope='unreserved preparation inputs; freshness census and launch freeze pending',baseline_sha256=BASELINE_SHA256,
        sources=sources,frozen_predecessors=frozen_predecessors(sources),environment=environment(),data=bind_data(),
        limits=dict(LIMITS),pilots=rules.pilot_specs(seeds),screens=rules.screen_specs(seeds),
        remaining_gates=['immutable_ledger','worker_parent_acceptance','full_validation_inference','retention','actual_corpus_pilots','full_panel_capacity','quality_plan_freeze'])


def verify_preparation_manifest(value):
    require(type(value) is dict and 'seeds' in value,'Preparation manifest required')
    require(canonical(value)==canonical(preparation_manifest(value['seeds'])),'Preparation binding changed')
