"""Separate-output inference body and read-back review, no launch entry point.

Reuse only original load_pair/expected_models/callbacks, which accept an explicit
original reader and budget callback. Never call original produce, budget,
quality_binding, inference entry or final campaign wrapper. Original kernels,
loader/cache checks and timing arithmetic remain under their unchanged source pin.
"""
import gc
import os

import torch
from src.research.protocol import digest
from src.research.tokenizer_activation_session import reference,require
from src.research.tokenizer_activation_timing import slot,measure_pair,review_result
from src.research.tokenizer_activation_inference import strict_mps,load_pair,callbacks,expected_models
from src.research.memory_guard import start_memory_guard
from research.activation_inference_recovery_v2.context import (
    KIND,bind,budget,verify_original,PhaseGuard,guard_ref,verify_guard,carried_import,REMAINING_SEEDS)
from research.activation_inference_recovery_v2.inventory import slot_inventory,require_slot_unchanged

OUTPUT_KIND=KIND+'_outputs'
REVIEW_KIND=KIND+'_review'


def produce(ctx,source,request_ref,binding):
    """Invoked only after the future child-entry transport gate has completed."""
    seed=binding['seed'];require(seed in REMAINING_SEEDS,'Carried first pair cannot repeat');stem=f'evidence/inference/{slot(seed)}';loaded=None
    require((ctx.root/stem).is_dir() and {p.name for p in (ctx.root/stem).iterdir()}<={'request.json','process'},
            'Existing recovery worker attempt cannot retry')
    def recheck():
        imported=bind(ctx)
        require(imported['pair_bindings'][str(seed)]==binding,'Fresh imported binding changed')
        verify_original(ctx,source,imported)
        require(reference(ctx.root,request_ref['path'])==request_ref,'Recovery request changed')
        return imported
    try:
        with PhaseGuard(ctx.writer,seed,'worker'):
            recheck();strict_mps();start_memory_guard();torch.set_num_threads(4)
            ctx.writer.publish_json(stem+'/attempt.json',dict(request=request_ref,pid=os.getpid(),parent_pid=os.getppid(),seed=seed))
            # The original ledger resolves original model paths/lineage. Only
            # the output-side synchronized budget is used in kernel callbacks.
            loaded=load_pair(source,binding,lambda:budget(ctx))
            recheck()
            check,measure=callbacks(loaded,binding['inputs']['training96_sha256'],lambda:budget(ctx))
            result=measure_pair(seed,binding['selection'],loaded['token_ids'],binding['inputs']['training96_sha256'],check=check,measure=measure)
            review_result(result,seed,binding['selection'],binding['inputs']['training96_sha256'])
            recheck()
            ctx.writer.publish_json(stem+'/models.json',loaded['metadata'])
            ctx.writer.publish_json(stem+'/result.json',result)
            ctx.writer.publish_json(stem+'/worker-outputs.json',dict(kind=OUTPUT_KIND,request=request_ref,
                seed=seed,pid=os.getpid(),parent_pid=os.getppid(),status='complete_pending_parent_review',
                models=reference(ctx.root,stem+'/models.json'),result=reference(ctx.root,stem+'/result.json')))
            budget(ctx)
        verify_guard(ctx,guard_ref(ctx,seed,'worker'),os.getpid())
        return reference(ctx.root,stem+'/worker-outputs.json')
    except BaseException as exc:
        try:ctx.writer.publish_json(stem+'/worker-failure.json',dict(error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Recovery worker failure publication: '+repr(error)[:1000])
        raise
    finally:
        del loaded;gc.collect()
        if torch.backends.mps.is_available():torch.mps.empty_cache()


def review_outputs(ctx,source,request_ref,binding,child_pid,parent_pid):
    seed=binding['seed'];require(seed in REMAINING_SEEDS,'Carried first pair cannot repeat');stem=f'evidence/inference/{slot(seed)}'
    before=budget(ctx)
    proof=slot_inventory(ctx.writer,seed)
    require(not any(name.startswith('evidence/inference/') and (name.endswith('.tmp') or 'failure' in name)
                    for name in before['files']),'Failed or partial recovery output')
    imported=bind(ctx)
    require(imported['pair_bindings'][str(seed)]==binding,'Original selected pair binding differs')
    verify_original(ctx,source,imported)
    refs={name:reference(ctx.root,stem+'/'+name+'.json') for name in ('request','attempt','models','result','worker-outputs')}
    require(refs['request']==request_ref,'Recovery request changed')
    require(ctx.document(refs['attempt'])==dict(request=request_ref,pid=child_pid,parent_pid=parent_pid,seed=seed),
            'Recovery worker identity differs')
    expected=dict(kind=OUTPUT_KIND,request=request_ref,seed=seed,pid=child_pid,parent_pid=parent_pid,
        status='complete_pending_parent_review',models=refs['models'],result=refs['result'])
    require(ctx.document(refs['worker-outputs'])==expected,'Recovery worker completion differs')
    require(ctx.document(refs['models'])==expected_models(source,binding),'Original model provenance differs')
    reviewed=review_result(ctx.document(refs['result']),seed,binding['selection'],binding['inputs']['training96_sha256'])
    verify_guard(ctx,guard_ref(ctx,seed,'worker'),child_pid)
    require_slot_unchanged(ctx.writer,seed,proof)
    require(all(reference(ctx.root,r['path'])==r for r in refs.values()),
            'Recovery output changed during review')
    return dict(kind=REVIEW_KIND,status='reviewed_pending_transport',request=request_ref,
        original_import_sha256=digest(imported),carried_import_sha256=digest(carried_import(ctx)),binding_sha256=digest(binding),review=reviewed,files=list(refs.values()))
