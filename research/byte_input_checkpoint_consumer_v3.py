"""Load the qualified raw-byte model through the batched-finite implementation.

The trusted implementation receipt admits the additive sources and their exact
differential evidence. Plan-derived identity and artifact admission reuse v1.
The caller owns the shared research lock and CPU/MPS resource guards.
"""
from pathlib import Path
from research import byte_input_checkpoint_consumer_v1 as original


def load(implementation_record, plan_record, acceptance_record, backend):
    implementation = original.read_bound(implementation_record)
    assert implementation['kind'] == 'qualified_byte_input_batched_finite_implementation_v1'
    assert backend in ('cpu', 'mps')
    for record in implementation['sources']:
        original.verify(record)
    mapping = original.read_bound(implementation['source_map'])
    for pair in mapping['pairs']:
        for record in (pair['original'], pair['candidate']):
            bound = dict(record)
            if not Path(bound['path']).is_absolute():
                bound['path'] = str(Path(__file__).resolve().parents[1]/bound['path'])
            original.verify(bound)
    evidence = original.read_bound(implementation['runtime_evidence'])
    guard = original.read_bound(implementation['runtime_guard'])
    assert evidence['accepted'] and evidence['numerical_integration_qualified']
    assert evidence['plan_sha256'] == implementation['runtime_plan']['sha256']
    original.verify(implementation['runtime_plan'])
    assert evidence['guard_sha256'] == implementation['runtime_guard']['sha256']
    assert guard['status'] == 'finished' and guard['reason'] is None
    assert len(evidence['cases']) == 2 and {case['backend'] for case in evidence['cases']} == {'cpu', 'mps'}
    assert all(case['exact'] and case['parameter_gradients'] == 105 and case['state_boundaries'] == 3
               for case in evidence['cases'])
    prepared = original.prepare(plan_record, acceptance_record)
    assert prepared['identity']['model'][1] == 'raw_bytes'
    from research import byte_input_inference_v3 as inference
    model = inference.load(prepared['h200'], prepared['identity'], prepared['parent'], prepared['codebook'], backend)
    return model, prepared['identity'], prepared['h200']
