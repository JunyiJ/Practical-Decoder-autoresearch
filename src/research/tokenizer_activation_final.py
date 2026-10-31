"""Independent finite activation campaign decision from immutable evidence.

No training, inference rerun, checkpoint deletion, extra seed or holdout access.
One terminal publication; incomplete attempts remain evidence and cannot retry.
"""
import argparse
import math
import os
from pathlib import Path
from statistics import mean

from .protocol import digest
from .tokenizer_activation_session import require,reference,Session
from .tokenizer_activation_entry import bind,idle,PhaseGuard,guard_ref,verify_guard,session_lock,GLOBAL_LOCK,MetadataWriter
from .tokenizer_activation_rules import (FAMILIES,FRESH_SEEDS,BLOCKS,BOOTSTRAP_SEED,DRAWS,GATE_QUANTILE,
    confirmation_specs,paired_statistics,quality_gate)
from .tokenizer_activation_inference import quality_binding
from .tokenizer_activation_inference_entry import accepted_prefix
from .tokenizer_activation_timing import slot,summarize
from .tokenizer_activation_audit import PARAMETERS,_score

CONTRACT=dict(kind='activation_final_decision_v1',contrast='gelu_exact_minus_silu',
    mean_improvement_strictly_greater_than=.02,every_seed_strictly_better=True,
    all_conditional_upper_bounds_strictly_negative=True,blocks=list(BLOCKS),draws=DRAWS,
    bootstrap_seed=BOOTSTRAP_SEED,gate_quantile=GATE_QUANTILE,quantile_method='linear',
    cost_adoption_route=False,representative_export_seed=3611,holdout_scored=False)
SCOPE=('Width256 four-layer quarter-latent RoPE MLA, equally tuned SiLU/exact-GELU gates, '
       '1200 updates and H200 on GPT2-tokenized Tiny Shakespeare. Adaptive validation; '
       'text-block intervals condition on these trained seeds and selection, not population '
       'seed uncertainty or independent holdout. No cost-only adoption route or transfer claim.')


def analyze(rows,selection):
    """Arithmetic/coverage decision only; executable wrapper supplies provenance."""
    specs=confirmation_specs(selection)
    require(type(rows) is list and len(rows)==6,'Exactly six fresh paired endpoints required')
    indexed={};costs=[];counts=None
    for row,spec in zip(rows,specs):
        require(set(row)=={'spec','raw','h200','training_seconds','worker_seconds','peak_rss_bytes','peak_mps_driver_bytes'}
                and digest(row['spec'])==digest(spec),'Ordered fresh endpoint identity differs')
        for role in ('raw','h200'):_score(row[role],PARAMETERS)
        observed=row['h200']['block_target_counts']
        require(counts is None or counts==observed,'Paired validation counts differ');counts=observed
        require(row['raw']['block_target_counts']==counts,'Raw/H200 coverage differs')
        for field in ('training_seconds','worker_seconds'):
            require(type(row[field]) in (int,float) and math.isfinite(row[field]) and row[field]>0,'Finite positive elapsed costs required')
        require(row['training_seconds']<=row['worker_seconds']<7200,'Training/worker duration differs')
        for field,limit in [('peak_rss_bytes',4*1024**3),('peak_mps_driver_bytes',6*1024**3)]:
            require(type(row[field]) is int and 0<row[field]<limit,'Observed resource limit differs')
        indexed[spec['seed'],spec['family']]=row
        costs.append(dict(seed=spec['seed'],family=spec['family'],lr=spec['lr'],
            raw_ce=row['raw']['mean_ce'],h200_ce=row['h200']['mean_ce'],
            raw_bits_per_byte=row['raw']['bits_per_byte'],h200_bits_per_byte=row['h200']['bits_per_byte'],
            **{k:row[k] for k in ('training_seconds','worker_seconds','peak_rss_bytes','peak_mps_driver_bytes')}))
    losses=[[indexed[s,f]['h200']['block_loss_sums'] for f in FAMILIES] for s in FRESH_SEEDS]
    stats=paired_statistics(losses,counts);gate=quality_gate(stats)
    means={f:mean(indexed[s,f]['h200']['mean_ce'] for s in FRESH_SEEDS) for f in FAMILIES}
    require(math.isclose(means['gelu_exact']-means['silu'],stats['mean_delta'],abs_tol=1e-12),'Paired score arithmetic differs')
    return dict(status='arithmetic_reviewed_pending_provenance',contract=CONTRACT,selection=selection,
        fresh_endpoints=costs,h200_means=means,statistics=stats,strict_quality=gate['strict_quality'],
        eligible_family='gelu_exact' if gate['strict_quality'] else None,scope=SCOPE,adopted=False)


def build_report(ledger):
    bind(ledger);state=ledger.inspect()
    require(state['next']['kind']=='review_final','Complete quality panel required')
    bindings={s:quality_binding(ledger,s) for s in FRESH_SEEDS}
    inference=accepted_prefix(ledger);require(len(inference)==3,'Complete independently accepted inference panel required')
    results=[ledger.document(reference(ledger.root,f'evidence/inference/{slot(s)}/result.json')) for s in FRESH_SEEDS]
    selection=state['selection'];inputs=bindings[FRESH_SEEDS[0]]['inputs']
    timing=summarize(results,selection,inputs['training96_sha256'])
    summary_ref=reference(ledger.root,'evidence/meta/activation-inference-summary.json')
    require(ledger.document(summary_ref)==dict(report=timing,accepted=inference,adopted=False),'Saved inference summary differs')
    rows=[];fresh_refs=[]
    for spec in confirmation_specs(selection):
        b=bindings[spec['seed']]['bindings'][spec['family']];rq=ledger.document(b['request'])
        require(rq['spec']==spec,'Selected fresh request differs');stem=str(Path(b['request']['path']).parent)
        refs={n:reference(ledger.root,stem+'/'+n+'.json') for n in ('measurements','training-summary','resource-audit')}
        measured=ledger.document(refs['measurements']);training=ledger.document(refs['training-summary']);resources=ledger.document(refs['resource-audit'])
        for role in ('raw','h200'):
            weight=b['weights'][role]
            require(measured[role]['checkpoint']==str(ledger.root/weight['path']) and measured[role]['checkpoint_sha256']==weight['sha256'],
                    'Final endpoint checkpoint binding differs')
        row=dict(spec=spec,raw=measured['raw'],h200=measured['h200'],training_seconds=training['training_time_sec'],
            worker_seconds=resources['samples'][-1]['elapsed_seconds'],
            peak_rss_bytes=max(s['peak_rss_bytes'] for s in resources['samples']),
            peak_mps_driver_bytes=max(s['mps_driver_bytes'] for s in resources['samples']))
        rows.append(row);fresh_refs.append(dict(spec=spec,accepted=b['acceptance'],request=b['request'],weights=b['weights'],measurements=refs))
    analysis=analyze(rows,selection);passed=analysis['strict_quality'];representative=None
    if passed:
        selected=bindings[CONTRACT['representative_export_seed']]['bindings']['gelu_exact']
        representative=dict(family='gelu_exact',seed=CONTRACT['representative_export_seed'],
            config=ledger.document(selected['request'])['config'],request=selected['request'],weights=selected['weights'],
            selection_rule='First predeclared fresh seed, never best observed seed; all fresh pairs remain protected.')
    # Exact pinned bytes must still match after all calculations. This repeats
    # provenance checks, never numerical training, inference or deleted-file audits.
    require(all(reference(ledger.root,r['path'])==r for item in fresh_refs for r in item['measurements'].values()),
            'Fresh endpoint metadata changed during final review')
    require(all(quality_binding(ledger,s)==b for s,b in bindings.items()) and accepted_prefix(ledger)==inference,
            'Evidence changed during final review')
    return dict(kind=CONTRACT['kind'],status='closed',plan=ledger.plan,analysis=analysis,inference=timing,
        inference_acceptances=inference,inference_summary=summary_ref,fresh_provenance=fresh_refs,
        adopted=passed,adopted_family='gelu_exact' if passed else None,representative_recipe=representative,
        reason='strict_paired_quality_passed' if passed else 'strict_paired_quality_failed',
        retained_fresh_pairs=6,retained_selected_screen_pairs=2,holdout_scored=False,scope=SCOPE)


def complete(ledger):
    bind(ledger);idle();base='evidence/meta/activation-final'
    require(not any((ledger.root/(base+suffix)).exists() for suffix in ('-attempt.json','-decision.json','-complete.json','-failure.json'))
            and not (ledger.root/'control/activation-final-review-00.json').exists(),'Existing final attempt cannot retry')
    ledger.writer.publish_json(base+'-attempt.json',dict(plan=ledger.plan,status='started',retry_allowed=False))
    try:
        with PhaseGuard(ledger.writer,'review',0,'final'):
            decision=build_report(ledger)
            ledger.writer.publish_json(base+'-decision.json',decision)
        g=guard_ref(ledger,'review',0,'final');verify_guard(ledger,g,os.getpid())
        ledger.writer.publish_json(base+'-complete.json',dict(status='closed',plan=ledger.plan,
            decision=reference(ledger.root,base+'-decision.json'),guard=g,adopted=decision['adopted']))
        return reference(ledger.root,base+'-complete.json')
    except BaseException as exc:
        try:ledger.writer.publish_json(base+'-failure.json',dict(error=repr(exc)[:1000],retry_allowed=False))
        except BaseException as error:exc.add_note('Final failure publication: '+repr(error)[:1000])
        raise


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True);parser.add_argument('--plan-sha256',required=True)
    args=parser.parse_args(argv)
    with session_lock(GLOBAL_LOCK.parent) as g,session_lock(Path(args.root)) as c:
        writer=MetadataWriter(args.root,(g,c));ref=reference(writer.root,'evidence/meta/activation-plan.json')
        require(ref['sha256']==args.plan_sha256,'Final CLI plan hash differs')
        return complete(Session(writer,ref))


if __name__=='__main__':main()
