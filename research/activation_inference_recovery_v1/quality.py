"""Imported fresh-endpoint arithmetic only; no terminal campaign publication."""
from pathlib import Path

from src.research.protocol import digest
from src.research.tokenizer_activation_session import require
from src.research.tokenizer_activation_rules import confirmation_specs
from src.research.tokenizer_activation_final import analyze
from research.activation_inference_recovery_v1.source_binding import verify_source_import


def _fresh_rows(ledger, imported):
    rows, provenance = [], []
    for index, spec in enumerate(confirmation_specs(imported['selection']), 6):
        pair = imported['pair_bindings'][str(spec['seed'])]['bindings'][spec['family']]
        request = ledger.document(pair['request'])
        stem = f'evidence/quality/{index:02d}'
        require(request['spec'] == spec and pair['request']['path'] == stem+'/request.json',
                'Exact original fresh request order required')
        refs = {name:imported['files'][stem+'/'+name+'.json']
                for name in ('measurements','training-summary','resource-audit')}
        require(all(ref['path'] == stem+'/'+name+'.json' for name,ref in refs.items()),
                'Imported endpoint path differs')
        measured = ledger.document(refs['measurements'])
        training = ledger.document(refs['training-summary'])
        resources = ledger.document(refs['resource-audit'])
        for role in ('raw','h200'):
            weight = pair['weights'][role]
            require(measured[role]['checkpoint'] == str(ledger.root/weight['path'])
                    and measured[role]['checkpoint_sha256'] == weight['sha256'],
                    'Imported score checkpoint lineage differs')
        rows.append(dict(spec=spec,raw=measured['raw'],h200=measured['h200'],
            training_seconds=training['training_time_sec'],worker_seconds=resources['samples'][-1]['elapsed_seconds'],
            peak_rss_bytes=max(s['peak_rss_bytes'] for s in resources['samples']),
            peak_mps_driver_bytes=max(s['mps_driver_bytes'] for s in resources['samples'])))
        provenance.append(dict(spec=spec,accepted=pair['acceptance'],request=pair['request'],
                               weights=pair['weights'],measurements=refs))
    return rows, provenance


def analyze_imported_quality(ledger, expected):
    imported = verify_source_import(ledger, expected)
    rows, provenance = _fresh_rows(ledger, imported)
    analysis = analyze(rows, imported['selection'])
    require(analysis['contract'] == imported['final_contract'], 'Frozen statistical contract differs')
    verify_source_import(ledger, imported)
    return dict(kind='activation001_imported_quality_review_v1',status='quality_reviewed_inference_pending',
        source_import_sha256=digest(imported),plan=imported['plan'],analysis=analysis,fresh_provenance=provenance,
        accepted_inference_pairs=0,formal_final_completed=False,adopted=False,holdout_scored=False,
        limitations='Original predeclared conditional text-block arithmetic on the same accepted six fresh endpoints. '
                    'No new scoring or seed-population inference. The separate complete inference panel and '
                    'independent terminal receipt are still required; this quality review cannot close the campaign.')
