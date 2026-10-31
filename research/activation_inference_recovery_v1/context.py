"""Prospective separate-root freeze/guard contract. No freezer or launch CLI."""
from contextlib import contextmanager
import copy
from pathlib import Path
import shutil

from src.research.protocol import ROOT,digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_activation_session import Session,reference,ref_schema,require
from src.research.tokenizer_activation_entry import session_lock
from src.research.tokenizer_activation_timing import CONTRACT,slot
from src.research.tokenizer_activation_final import CONTRACT as FINAL
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_cleanup import _locks
from src.research.tokenizer_wide_hybrid_retention import METADATA
from src.research.tokenizer_wide_hybrid_pilot_dispatch_v2 import _Guard
from src.research.cpu_prerequisite_guard import violation
from research.activation_inference_recovery_v1.census import synchronized_snapshot
from research.activation_inference_recovery_v1.source_binding import (
    ORIGINAL_ROOT,SOURCE_SHA256,PLAN_SHA256,verify_source_import)

KIND='activation_inference_recovery_v1'
RECOVERY_ROOT=ROOT/'research_runs/tokenizer-activation-inference-recovery-001'
PLAN_NAME='evidence/meta/recovery-plan.json'
IMPORT_NAME='evidence/meta/original-import.json'
IMPORT_SHA256='feb9fc8403459aa1d16f50052a26ddba243985c32146257582873c96f8644e87'
CODE_ROOT=ROOT/'research/activation_inference_recovery_v1'
# Missing controller/final components or prerequisites deliberately block any
# prospective bind until they exist, pass independent checks, and are frozen.
SOURCE_FILES=('census.py','source_binding.py','quality.py','context.py','worker.py','controller.py','final.py')
PREREQUISITES={name:'research/tokenizer-evidence/activation-preparation-001/inference-recovery-'+name+'-prerequisite-001-review.json'
               for name in ('census','binding','quality','execution','final')}
for _stage in ('execution','final'):
    PREREQUISITES[_stage]=PREREQUISITES[_stage].replace('-001-review.json','-002-review.json')
STORAGE=dict(metadata_bytes=METADATA,checkpoint_allocation_bytes=0,reserve_bytes=CONTRACT['reserve_bytes'],
             full_required_free_bytes=CONTRACT['reserve_bytes']+METADATA)


def external_source():
    return {str((CODE_ROOT/name).relative_to(ROOT)):file_digest(CODE_ROOT/name) for name in SOURCE_FILES}


class Recovery:
    """Separate artifact reader, deliberately not the original training Session."""
    verify=Session.verify
    document=Session.document
    def __init__(self,writer,plan_ref):
        require(isinstance(writer,MetadataWriter),'Original bounded writer required')
        self.writer,self.root=writer,writer.root
        self.plan=copy.deepcopy(ref_schema(self.root,plan_ref))
        require(self.plan['path']==PLAN_NAME,'Dedicated recovery plan path required')


def budget(ctx, *, capacity=False):
    observed=synchronized_snapshot(ctx.writer)
    require(not any(name.endswith('.tmp') or 'failure' in Path(name).name for name in observed['files']),
            'Failed or partial recovery cannot retry or advance')
    require(observed['checkpoint_logical_bytes']==0,'Recovery cannot write checkpoints')
    require(all(name in ('.lock','STOP') or name.startswith(('control/','emergency/','evidence/meta/','evidence/inference/'))
                for name in observed['files']),'Recovery cannot contain training or retention output')
    size=sum(v['size'] for name,v in observed['files'].items() if name.startswith('evidence/inference/'))
    require(size+16384<=CONTRACT['panel_limit_bytes'],'Inference panel output cap exceeded')
    required=CONTRACT['reserve_bytes']+(METADATA-observed['metadata_bytes'] if capacity else 0)
    require(shutil.disk_usage(ctx.root).free>=required,'Recovery disk reserve or finite allocation insufficient')
    require(not (ctx.root/'STOP').exists() and not (ctx.root/'STOP').is_symlink(),'STOP blocks recovery')
    return observed


def bind(ctx, *, capacity=False):
    require(ctx.root==RECOVERY_ROOT and ctx.root!=ORIGINAL_ROOT,'Exact separate recovery root required')
    _locks(ctx.root/'evidence',ctx.writer.lock_fds)
    budget(ctx,capacity=capacity)
    plan=ctx.document(ctx.plan)
    require(set(plan)=={'kind','status','root','external_source','original_source_sha256','runtime','source_import',
                       'timing','final','storage','prerequisites'} and plan['kind']==KIND and plan['status']=='frozen'
            and plan['root']==str(ctx.root),'Complete frozen recovery plan required')
    require(plan['external_source']==external_source() and plan['original_source_sha256']==SOURCE_SHA256
            and digest(source_fingerprint())==SOURCE_SHA256,'Frozen original or external source changed')
    require(plan['runtime']==runtime_identity() and plan['timing']==CONTRACT and plan['final']==FINAL
            and plan['storage']==STORAGE,'Runtime, finite design or storage contract changed')
    require(set(plan['prerequisites'])==set(PREREQUISITES),'All recovery prerequisites required')
    for name,path in PREREQUISITES.items():
        ref=plan['prerequisites'][name]
        require(ref['path']==path and reference(ROOT,path)==ref and ref['size']<=2*1024**2,
                'Exact recovery prerequisite reference required')
        import json
        report=json.loads((ROOT/path).read_text())
        require(report.get('status')=='passed','Recovery prerequisite not passed: '+name)
        # Review identities bind tested code, not just a status word. Historical
        # census and binder reports may name a subset of the eventual sources.
        for name_or_path,sha in report['code_sha256'].items():
            path=Path(name_or_path)
            path=path if path.is_absolute() else ROOT/path
            require(path.parent==CODE_ROOT and file_digest(path)==sha,'Reviewed prerequisite source changed')
    ref=plan['source_import']
    require(ref['path']==IMPORT_NAME and ref['sha256']==IMPORT_SHA256,'Exact preserved original import required')
    imported=ctx.document(ref)
    require(imported['original_root']==str(ORIGINAL_ROOT) and imported['source_sha256']==SOURCE_SHA256
            and imported['plan']['sha256']==PLAN_SHA256 and imported['runtime']==plan['runtime']
            and imported['timing_contract']==CONTRACT and imported['final_contract']==FINAL
            and imported['training_complete'] is True and imported['original_inference_failed'] is True
            and imported['original_final_completed'] is False and imported['dispatch_authorized'] is False,
            'Original source import identity differs')
    return imported


@contextmanager
def original_session(ctx):
    """Borrow inherited global ownership; acquire original root independently.

    Parent must close this context BEFORE spawning a child. The child acquires
    its own original-root lock and retains it through model loading/measurement.
    New-root parent monitoring never opens the original root while child runs.
    """
    with session_lock(ORIGINAL_ROOT) as original_fd:
        writer=MetadataWriter(ORIGINAL_ROOT,(ctx.writer.lock_fds[0],original_fd))
        source=Session(writer,reference(ORIGINAL_ROOT,'evidence/meta/activation-plan.json'))
        require(source.plan['sha256']==PLAN_SHA256,'Original plan identity changed')
        yield source


def verify_original(ctx,source,imported):
    require(source.root==ORIGINAL_ROOT and source.root!=ctx.root,'Separate original evidence reader required')
    return verify_source_import(source,imported)


class PhaseGuard(_Guard):
    KIND=KIND
    def __init__(self,writer,seed,phase):
        super().__init__(writer,slot(seed),phase,CONTRACT['job_seconds'])
        self.name=f'control/recovery-{phase}-inference-{slot(seed)}.json'


def guard_ref(ctx,seed,phase):
    return reference(ctx.root,f'control/recovery-{phase}-inference-{slot(seed)}.json')


def verify_guard(ctx,ref,pid):
    value=ctx.document(ref)
    limits=dict(rss_limit_bytes=CONTRACT['rss_limit_bytes'],reserve_bytes=CONTRACT['reserve_bytes'],timeout_seconds=600)
    require(value.get('kind')==KIND and value.get('status')=='finished' and value.get('reason') is None
            and value.get('pid')==pid and value.get('limits')==limits,'Complete recovery phase guard required')
    sample=value.get('observation',{});elapsed=value.get('elapsed_seconds')
    require(set(sample)=={'peak_rss_bytes','free_bytes'} and all(type(v) is int and v>0 for v in sample.values())
            and type(elapsed) in (float,int) and 0<=elapsed<600
            and violation(sample,elapsed,**limits) is None,'Recovery guard resource observation differs')
