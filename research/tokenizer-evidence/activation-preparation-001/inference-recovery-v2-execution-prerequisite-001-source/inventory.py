"""Finite inference quota observations and immutable per-slot read proofs.

This component does not authorize launches or accept timing semantics. Callers
must independently bind a frozen plan, source import, completed process/guards
and serial receipts. It may only be called outside a writer serialization block.
"""
from pathlib import Path
import shutil
import time

from src.research.tokenizer_activation_session import reference,require
from src.research.tokenizer_activation_timing import CONTRACT,FRESH_SEEDS,slot
from src.research.tokenizer_wide_hybrid_metadata import snapshot
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_retention import METADATA

SECONDS=2.0
META={'recovery-plan.json','original-import.json','carried-pair-import.json','recovery-inference-summary.json',
      'recovery-final-attempt.json','recovery-final-decision.json','recovery-final-complete.json'}
SLOT_FILES={'request.json','attempt.json','models.json','result.json','worker-outputs.json',
            'review.json','accepted.json','process/launch.json','process/worker.log','process/process.json'}
CONTROLS={'control/.metadata.lock'}|{f'control/process-inference-{slot(seed)}.json' for seed in FRESH_SEEDS}|{
    f'control/recovery-{phase}-inference-{slot(seed)}.json' for seed in FRESH_SEEDS
    for phase in ('prepare','entry','worker','review','final')}


def _allowed(name):
    if name=='.lock' or name in CONTROLS:return True
    if name.startswith('evidence/meta/'):return name.removeprefix('evidence/meta/') in META
    return any(name.startswith(f'evidence/inference/{slot(seed)}/')
        and name.removeprefix(f'evidence/inference/{slot(seed)}/') in SLOT_FILES for seed in FRESH_SEEDS)


def _validate(writer,seen,capacity):
    # snapshot already counts all distinct temporary bytes/names and enforces
    # original group caps, regular files, symlinks and hardlink alias constraints.
    require('STOP' not in seen['files'],'STOP blocks inference')
    require(not any(name.endswith('.tmp') or 'failure' in Path(name).name for name in seen['files']),
            'Failed or partial inference cannot retry or advance')
    require(seen['checkpoint_logical_bytes']==0,'Inference cannot allocate checkpoint files')
    require(all(_allowed(name) for name in seen['files']),'Unknown inference output')
    size=sum(entry['size'] for name,entry in seen['files'].items() if name.startswith('evidence/inference/'))
    require(size+16384<=CONTRACT['panel_limit_bytes'],'Inference panel cap exceeded')
    required=CONTRACT['reserve_bytes']+(METADATA-seen['metadata_bytes'] if capacity else 0)
    require(shutil.disk_usage(writer.root).free>=required,'Inference reserve or remaining allocation insufficient')


def _observe(writer,seed=None,capacity=False):
    require(isinstance(writer,MetadataWriter),'Original bounded writer required')
    if seed is not None:slot(seed)  # exact integer and finite scientific seed
    started=time.monotonic()
    with writer._serialized():
        before=snapshot(writer.root);_validate(writer,before,capacity)
        refs=None
        if seed is not None:
            stem=f'evidence/inference/{slot(seed)}/';suffix=f'inference-{slot(seed)}.json'
            names=sorted(name for name in before['files'] if name.startswith(stem)
                         or name.startswith('control/') and name.endswith(suffix))
            require(any(name.startswith(stem) for name in names),'No slot evidence to verify')
            refs={name:reference(writer.root,name) for name in names}
        after=snapshot(writer.root);_validate(writer,after,capacity)
        require(before==after,'Inventory changed inside metadata lock; no retry')
        require(time.monotonic()-started<SECONDS,'Inference inventory time limit')
    return after,refs


def global_observation(writer,*,capacity=False):
    """Check the entire root at one bounded instant; never demand time stability."""
    return _observe(writer,capacity=capacity)[0]


def slot_inventory(writer,seed):
    """Hash a slot's files and associated controls while checking global limits.

    An immutable completed slot must retain this exact map across a longer audit.
    Other slots' live control/log bytes may change, but every observation still
    checks their names, sizes, caps, failures and STOP. No prefix is exempt from
    the global census. A partial current-slot proof is not completion evidence.
    """
    slot(seed)
    return _observe(writer,seed=seed)[1]


def require_slot_unchanged(writer,seed,expected):
    observed=slot_inventory(writer,seed)
    require(type(expected) is dict and observed==expected,'Immutable slot evidence changed')
    return observed
