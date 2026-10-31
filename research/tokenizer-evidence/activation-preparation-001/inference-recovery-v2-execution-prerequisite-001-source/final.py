"""Separate finite recovery closure; original failed campaign is never closed anew."""
import argparse
from pathlib import Path
import os

from src.research.protocol import digest
from src.research.tokenizer_activation_session import reference,require
from src.research.tokenizer_activation_entry import idle,GLOBAL_LOCK,session_lock,MetadataWriter
from src.research.tokenizer_activation_timing import FRESH_SEEDS,slot,summarize
from research.activation_inference_recovery_v2.context import (
    KIND,PLAN_NAME,Recovery,bind,original_session,PhaseGuard,guard_ref,verify_guard,carried_import)
from research.activation_inference_recovery_v2.controller import accepted_prefix,SUMMARY,timing_results
from research.activation_inference_recovery_v1.quality import analyze_imported_quality

BASE='evidence/meta/recovery-final'


def build_report(ctx):
    imported=bind(ctx)
    with original_session(ctx) as source:
        prior=accepted_prefix(ctx,source,imported)
        require(len(prior)==3,'All three independently accepted recovery inference pairs required')
        quality=analyze_imported_quality(source,imported)
    results=timing_results(ctx)
    timing=summarize(results,imported['selection'],imported['inputs']['training96_sha256'])
    summary=reference(ctx.root,SUMMARY)
    require(ctx.document(summary)==dict(report=timing,accepted=prior,original_import_sha256=digest(imported),carried_import_sha256=digest(carried_import(ctx)),adopted=False),
            'Saved recovery inference summary differs')
    require(quality['source_import_sha256']==digest(imported) and quality['formal_final_completed'] is False,
            'Independent quality import identity differs')
    passed=quality['analysis']['strict_quality'];representative=None
    if passed:
        seed=imported['final_contract']['representative_export_seed']
        selected=imported['pair_bindings'][str(seed)]['bindings']['gelu_exact']
        representative=dict(family='gelu_exact',seed=seed,original_root=imported['original_root'],
            request=selected['request'],weights=selected['weights'],selection_rule='First predeclared fresh seed; never best observed seed.')
    with original_session(ctx) as source:
        require(bind(ctx)==imported and accepted_prefix(ctx,source,imported)==prior,'Evidence changed during final review')
    require(reference(ctx.root,SUMMARY)==summary,'Recovery summary changed during final review')
    return dict(kind=KIND+'_final',status='closed',plan=ctx.plan,original_import_sha256=digest(imported),
        carried_import_sha256=digest(carried_import(ctx)),carried_pair_origin=carried_import(ctx)['original_root'],
        carried_first_pair_repeated=False,new_inference_seeds=[3612,3613],
        original_root=imported['original_root'],original_failure_audit=imported['failure_audit'],
        original_campaign_status='training_complete_inference_failed_no_retry',original_campaign_retried=False,
        analysis=quality['analysis'],fresh_provenance=quality['fresh_provenance'],inference=timing,
        inference_acceptances=prior,inference_summary=summary,adopted=passed,adopted_family='gelu_exact' if passed else None,
        representative_recipe=representative,retained_fresh_pairs=6,retained_selected_screen_pairs=2,holdout_scored=False,
        reason='strict_paired_quality_passed' if passed else 'strict_paired_quality_failed')


def complete(ctx):
    bind(ctx);idle()
    require(not any((ctx.root/(BASE+suffix)).exists() for suffix in ('-attempt.json','-decision.json','-complete.json','-failure.json'))
        and not (ctx.root/f'control/recovery-final-inference-{slot(FRESH_SEEDS[0])}.json').exists(),
        'Existing final recovery attempt cannot retry')
    ctx.writer.publish_json(BASE+'-attempt.json',dict(plan=ctx.plan,status='started',retry_allowed=False))
    try:
        with PhaseGuard(ctx.writer,FRESH_SEEDS[0],'final'):
            decision=build_report(ctx);ctx.writer.publish_json(BASE+'-decision.json',decision)
        guard=guard_ref(ctx,FRESH_SEEDS[0],'final');verify_guard(ctx,guard,os.getpid())
        ctx.writer.publish_json(BASE+'-complete.json',dict(kind=KIND,status='closed',plan=ctx.plan,
            decision=reference(ctx.root,BASE+'-decision.json'),guard=guard,adopted=decision['adopted']))
        return reference(ctx.root,BASE+'-complete.json')
    except BaseException as exc:
        try:ctx.writer.publish_json(BASE+'-failure.json',dict(error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Recovery final failure publication: '+repr(error)[:1000])
        raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args(argv);root=Path(args.root).absolute()
    require(root==root.resolve() and root.is_dir(),'Canonical existing recovery root required')
    with session_lock(GLOBAL_LOCK.parent) as g,session_lock(root) as c:
        writer=MetadataWriter(root,(g,c));plan=reference(root,PLAN_NAME)
        require(plan['sha256']==args.plan_sha256,'Frozen recovery final CLI plan differs')
        return complete(Recovery(writer,plan))


if __name__=='__main__':main()
