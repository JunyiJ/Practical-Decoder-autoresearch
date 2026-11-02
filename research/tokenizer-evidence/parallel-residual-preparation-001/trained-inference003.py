"""Declared alternating inference on the two accepted generated-text worker outputs."""
import io,json,shutil
from pathlib import Path
import torch
from research.parallel_residual_worker_v2 import evidence_contract as c,inference as inf
from research.local_mix_v1.output import PerFileStream
from research.ffn_norm_quality_v1.support import idle
from research.block_norm_v1.fixture_dispatch import reference,read
from research.block_norm_v1.pilot_execution import exclusive
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle();accepted=read(reference(B/'trainer-worker003.json'))
    assert accepted['status']=='real_bounded_worker_parent_acceptance_passed'
    c.verify_sources(accepted['sources']);bindings={r['arm']:r['binding'] for r in accepted['arms']}
    plan=dict(contract=inf.CONTRACT,bindings=bindings,pair_index=0,runner=reference(__file__),sources=accepted['sources'],source_acceptance=reference(B/'trainer-worker003.json'),scientific=False,generated_fixture_only=True)
    exclusive(B/'trained-inference003-plan.json',plan)
    with StrictMpsGuard(B/'trained-inference003-guard.json',c.ROOT):
        # Negative boundary probe of the newly added per-file limiter: failure
        # must happen before the underlying stream receives the excess byte.
        buffer=io.BytesIO();limited=PerFileStream(buffer,7);limited.write(memoryview(b'1234567'))
        try:limited.write(b'8')
        except ValueError:pass
        else:raise AssertionError('Per-file cap accepted overflow')
        assert buffer.getvalue()==b'1234567' and limited.tell()==7
        measured=inf.measure_pair(bindings,0);review=inf.review(measured,bindings,0)
        exclusive(B/'trained-inference003-measurements.json',measured)
        c.verify_sources(accepted['sources'])
        result=dict(status='generated_trained_parallel_residual_pair_inference_passed',plan=reference(B/'trained-inference003-plan.json'),measurements=reference(B/'trained-inference003-measurements.json'),review=review,
            individual_writer_exact_boundary_rejection=True,
            limitations='Four-update generated-text serial/parallel fixture; validates measurement/acceptance mechanics, not scientific quality or production-performance advantage.',
            scientific_training=False,holdout_read=False,free_bytes=shutil.disk_usage(c.ROOT).free)
    result['guard']=reference(B/'trained-inference003-guard.json');assert read(result['guard'])['status']=='finished'
    exclusive(B/'trained-inference003.json',result);print(json.dumps(dict(status=result['status'],cells=review['cells'])),flush=True)
