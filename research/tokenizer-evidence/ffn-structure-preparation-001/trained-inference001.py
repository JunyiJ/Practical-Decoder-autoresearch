"""Declared alternating inference measurement on already accepted generated fixtures."""
import hashlib,json,shutil
from pathlib import Path
import torch
from research.ema_horizon_v1 import contract as c,execution as e
from research.block_norm_v1.fixture_dispatch import reference,read
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.ffn_structure_v1 import inference as inf
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=Path(__file__).resolve().parent
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    e.idle();accepted=read(reference(B/'trainer-worker003.json'))
    assert accepted['status']=='actual_serial_trainer_and_parent_acceptance_passed'
    bindings={r['arm']:r['binding'] for r in accepted['arms']}
    frozen=c.source_inventory();assert frozen==accepted['frozen_sources']
    sources={str(p.relative_to(c.ROOT)):sha(p) for p in (c.ROOT/'research/ffn_structure_v1').rglob('*.py')}
    for n,h in accepted['sources'].items():assert sources[n]==h
    plan=dict(contract=inf.CONTRACT,bindings=bindings,pair_index=0,runner=reference(__file__),sources=sources,frozen_sources=frozen,
              source_acceptance=reference(B/'trainer-worker003.json'),scientific=False,generated_fixture_only=True)
    e.exclusive(B/'trained-inference001-plan.json',plan)
    with StrictMpsGuard(B/'trained-inference001-guard.json',c.ROOT):
        measured=inf.measure_pair(bindings,0);review=inf.review(measured,bindings,0)
        e.exclusive(B/'trained-inference001-measurements.json',measured)
        assert frozen==c.source_inventory()
        for n,h in sources.items():assert sha(c.ROOT/n)==h
        result=dict(status='generated_trained_ffn_pair_inference_passed',plan=reference(B/'trained-inference001-plan.json'),
            measurements=reference(B/'trained-inference001-measurements.json'),review=review,
            limitations='Two-update generated-text fixture only; validates measurement/acceptance mechanics. Not scientific quality or production-performance evidence.',
            scientific_training_launched=False,holdout_read=False,free_bytes=shutil.disk_usage(c.ROOT).free)
    result['guard']=reference(B/'trained-inference001-guard.json');e.exclusive(B/'trained-inference001.json',result)
    print(json.dumps(dict(status=result['status'],cells=review['cells'])),flush=True)
