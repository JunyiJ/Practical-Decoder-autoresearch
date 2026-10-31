"""One guarded, read-only prefreeze audit; no training/checkpoint IO or holdout score."""
import json
import math
import os
from pathlib import Path
import shutil
from datetime import datetime,timezone

import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.models.tokenizer_architecture import validate_architecture_model
from src.train.optimizers import build_optimizer
from src.train.train import _make_loader
from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_config import config,BASELINE,BASELINE_SHA256,LIMITS
from src.research.tokenizer_activation_entry import (ALLOWANCE,HIGH_WATER,FULL_FREE,METADATA,RESERVE,
    PREREQUISITES,design,idle,session_lock,GLOBAL_LOCK)
from src.research.tokenizer_activation_session import reference
from src.research.tokenizer_activation_worker import input_identity,CORPUS_SHA256
from src.research.tokenizer_activation_rules import pilot_specs,RATES,FAMILIES,confirmation_specs
from src.research.tokenizer_wide_hybrid_metadata import budgets

BASE=ROOT/'research/tokenizer-evidence/activation-preparation-001'
NAMES=dict(model='contract-tests-v1-review.json',state='state-prerequisite-review.json',rules='rules-tests-v1-review.json',
 session='session-prerequisite-review.json',worker='worker-prerequisite-review.json',dispatch='dispatch-retention-prerequisite-review.json',
 retention='dispatch-retention-prerequisite-review.json',inference='inference-prerequisite-review.json',final='final-prerequisite-review.json')
ROOT_NEW=ROOT/'research_runs/tokenizer-activation-001'


def audit():
    assert not ROOT_NEW.exists(),'New campaign root already exists'
    assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')!='1' and torch.backends.mps.is_available()
    torch.set_num_threads(4);idle();source=source_fingerprint();runtime=runtime_identity()
    latest=json.loads((BASE/'final-tests-v1-review.json').read_text())
    assert latest['status']=='passed' and latest['source_sha256']==digest(source)
    assert runtime==latest['runtime'] and file_digest(BASELINE)==BASELINE_SHA256
    index=json.loads((ROOT/'research/TOKENIZER_RESUME_STATE.json').read_text())
    assert all(file_digest(ROOT/name)==sha for name,sha in index['evidence_sha256'].items())
    refs={k:reference(ROOT,str((BASE/v).relative_to(ROOT))) for k,v in NAMES.items()}
    assert set(refs)==set(PREREQUISITES)-{'capacity'}
    for name,ref in refs.items():
        doc=json.loads((ROOT/ref['path']).read_text());assert doc['status']==PREREQUISITES[name]
    # All final changes are research adapters; every model, optimizer, EMA,
    # checkpoint, data, cache and timing/transport primitive stays byte-identical
    # to the completed dispatch source, whose earlier applicability is retained.
    ancestor=json.loads((BASE/'dispatch-retention-tests-v2-review.json').read_text())['source']
    changed={k:dict(previous=ancestor.get(k),current=v) for k,v in source.items() if ancestor.get(k)!=v}
    assert set(changed)=={'src/research/tokenizer_activation_entry.py','src/research/tokenizer_activation_timing.py',
        'src/research/tokenizer_activation_inference.py','src/research/tokenizer_activation_inference_entry.py','src/research/tokenizer_activation_final.py'}
    assert all(k in source for k in ancestor)
    rows={}
    for n,spec in enumerate(pilot_specs()):
        cfg=OmegaConf.create(config(ROOT_NEW/f'evidence/pilots/{n:02d}',spec))
        assert cfg.training.checkpoint_every==0 and cfg.training.keep_best_checkpoint is False and cfg.training.resume_from is None
        with torch.device('meta'):
            model=GPT(cfg.model);validate_architecture_model(model);opt=build_optimizer(model,cfg.training)
            named=dict(model.named_parameters());count=sum(p.numel() for p in named.values())
            groups={type(child).__name__:dict(parameters=sum(p.numel() for g in child.param_groups for p in g['params']),
                        tensors=sum(len(g['params']) for g in child.param_groups)) for child in opt.optimizers}
            covered=[id(p) for child in opt.optimizers for g in child.param_groups for p in g['params']]
            assert len(covered)==len(set(covered))==len(named) and set(covered)=={id(p) for p in named.values()}
            buffers=sum(v.numel()*v.element_size() for k,v in model.named_buffers() if k in model.state_dict())
        assert set(groups)=={'Muon','AdamW'} and count==28492416 and buffers==65536
        optimizer=4*groups['Muon']['parameters']+8*groups['AdamW']['parameters']+4*groups['AdamW']['tensors']
        payload=12*count+optimizer+2*buffers
        allowance=math.ceil((24*count+2*buffers+2*1024**2)*1.15)
        assert payload==558969520 and optimizer==216929456 and allowance==ALLOWANCE
        assert payload+4*count<=24*count+2*buffers
        rows[spec['family']]=dict(parameters=count,optimizer_groups=groups,optimizer_tensor_bytes=optimizer,
            persistent_buffer_bytes=buffers,raw_h200_payload_bytes=payload,per_trajectory_allowance_bytes=allowance)
    assert rows['silu']==rows['gelu_exact']
    branches=[]
    for a in RATES:
        for b in RATES:
            selected=dict(zip(FAMILIES,(a,b)));fresh=confirmation_specs(selected)
            assert len(fresh)==6
            branches.append(dict(selection=selected,pilot_pairs=2,screen_pairs=6,
                retained_selected_pairs=2,retained_fresh_pairs=6,high_water_pairs=8,weight_high_water_bytes=8*ALLOWANCE))
    assert max(b['weight_high_water_bytes'] for b in branches)==HIGH_WATER
    assert sum(budgets().values())==METADATA and HIGH_WATER+METADATA+RESERVE==FULL_FREE
    cache=json.loads((ROOT/'research/tokenizer-evidence/wide-hybrid-preparation-001/campaign005-final-cache-environment-diagnosis-v2.json').read_text())['cache_sha256']
    assert all(file_digest(ROOT/name)==sha for name,sha in cache.items())
    cache_root=ROOT/'research_state/tiktoken-cache'
    assert Path(os.environ['TIKTOKEN_CACHE_DIR']).resolve()==cache_root.resolve()
    cfg=OmegaConf.create(config(ROOT_NEW/'evidence/pilots/00',pilot_specs()[0]))
    assert file_digest(Path(cfg.data.path))==CORPUS_SHA256
    loader=_make_loader(cfg,'cpu');inputs=input_identity(loader)
    assert inputs['validation_targets']==34278 and inputs['validation_target_bytes']==111536
    del loader
    assert source_fingerprint()==source and runtime_identity()==runtime
    assert all(reference(ROOT,ref['path'])==ref for ref in refs.values())
    free=shutil.disk_usage(ROOT).free;assert free>=FULL_FREE
    return dict(status='passed',source_sha256=digest(source),source=source,runtime=runtime,prerequisites=refs,
        storage=dict(trajectory_allowance_bytes=ALLOWANCE,maximum_weight_bytes=HIGH_WATER,metadata_bytes=METADATA,
            reserve_bytes=RESERVE,full_required_free_bytes=FULL_FREE),inputs=inputs,complete_executable_review=True,
        exact_implemented_shapes=rows,all_nine_selection_branches=branches,
        metadata_accounting=dict(total_reserved_bytes=METADATA,inference_panel_cap_bytes=16*1024**2,
            inference_bucket_capacity_bytes=sum(v for k,v in budgets().items() if k.startswith('evidence/inference/')),
            note='Inference consumes existing metadata buckets, no extra weight files. Final-only raw/H200 output; single serialized producer; checkpoint atomic temporaries count at full size inside per-trajectory allowance.'),
        changed_since_dispatch=changed,unchanged_runtime_and_model_training_primitives=True,tokenizer_cache_sha256=cache,
        corpus_sha256=CORPUS_SHA256,free_bytes=free,margin_bytes=free-FULL_FREE,root=str(ROOT_NEW),
        limits=LIMITS,design_sha256=digest(design()),observed_at=datetime.now(timezone.utc).isoformat(),
        procedure_sha256=file_digest(Path(__file__)),indexed_evidence_verified=len(index['evidence_sha256']),
        existing_protected_checkpoint_deletion_credit_bytes=0,checkpoint_read_write_or_delete=False,
        optimizer_step=False,model_forward=False,holdout_scored=False,research_training_launched=False,
        scope='Implemented meta shapes and complete optimizer groups; payload/overlap plus15percent allowances, all future retained pairs and256MiB metadata. Deterministic CPU tokenizer/input hashes only; no holdout score. Actual MPS memory/time/cache fit remains pilot-gated.')


def main():
    guard=BASE/'freeze-preflight-v1-guard.json'
    with session_lock(GLOBAL_LOCK.parent):
        with CpuPrerequisiteGuard(guard,ROOT,timeout_seconds=180):result=audit()
    g=json.loads(guard.read_text());assert g['status']=='finished' and g['reason'] is None
    result['guard']=reference(ROOT,str(guard.relative_to(ROOT)))
    with (BASE/'capacity-applicability-v1.json').open('x') as out:json.dump(result,out,indent=2);out.write('\n')
    print(json.dumps({k:result[k] for k in ('status','source_sha256','free_bytes','margin_bytes','indexed_evidence_verified')}))


if __name__=='__main__':main()
