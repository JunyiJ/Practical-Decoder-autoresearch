"""Load the qualified raw-byte model through complete-buffer hashing.

Caller supplies trusted receipts and owns the shared worker lock/resource guard.
Historical typed identity and selected checkpoint admission reuse consumer-v1.
"""
from pathlib import Path
from research import byte_input_checkpoint_consumer_v1 as original


def load(implementation_record, plan_record, acceptance_record, backend):
    implementation = original.read_bound(implementation_record)
    assert implementation['kind'] == 'qualified_byte_input_bulk_hash_implementation_v1'
    assert backend in ('cpu', 'mps')
    for record in implementation['sources']: original.verify(record)
    mapping = original.read_bound(implementation['source_map'])
    for pair in mapping['pairs']:
        for record in (pair['original'], pair['candidate']):
            bound = dict(record)
            if not Path(bound['path']).is_absolute():
                bound['path'] = str(Path(__file__).resolve().parents[1]/bound['path'])
            original.verify(bound)
    evidence = original.read_bound(implementation['runtime_evidence'])
    guard = original.read_bound(implementation['runtime_guard'])
    original.verify(implementation['runtime_plan'])
    assert evidence['plan_sha256'] == implementation['runtime_plan']['sha256']
    assert evidence['guard_sha256'] == implementation['runtime_guard']['sha256']
    assert guard['status'] == 'finished' and guard['reason'] is None
    assert evidence['completed'] and evidence['exact_history_and_endpoint']
    assert evidence['exact_cached_logits_and_reset']
    assert evidence['cpu']['exact_boundaries'] == [1, 2]
    assert evidence['cpu']['exact_resume_step1_to_step2'] and evidence['cpu']['gradient_tensors'] == 105
    assert evidence['useful_training_improvement'] and all(evidence['gates'].values())
    prepared = original.prepare(plan_record, acceptance_record)
    assert prepared['identity']['model'][1] == 'raw_bytes'
    from research import byte_input_inference_v5 as inference
    model = inference.load(prepared['h200'], prepared['identity'], prepared['parent'], prepared['codebook'], backend)
    return model, prepared['identity'], prepared['h200']
