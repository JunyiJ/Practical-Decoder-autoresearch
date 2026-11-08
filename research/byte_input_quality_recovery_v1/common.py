"""Evidence routing for additive recovery; original failed records stay closed."""
import json
from pathlib import Path
from research.byte_input_quality_v1.controller import sha, acceptance as original_acceptance

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001/quality001'
BASE = ORIGINAL.parent/'quality-recovery001'
OLD_WEIGHTS = ROOT/'research_runs/byte-input-quality-v1-001'
WEIGHTS = ROOT/'research_runs/byte-input-quality-recovery-v1-001'


def read(path):
    return json.loads(path.read_text())


def put(path, value):
    import os
    with path.open('x') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def slot_dir(index):
    return (ORIGINAL if index < 14 else BASE)/'slots'/f'{index:02}'


def acceptance(index):
    if index < 14:
        return original_acceptance(index)
    out = slot_dir(index); record = read(out/'acceptance.json')
    assert record['accepted'] and record['slot'] == index
    for name, digest in record['evidence'].items():
        assert sha(out/name) == digest
    guard = read(out/'guard.json')
    assert guard['status'] == 'finished' and guard['reason'] is None
    assert read(out/'STOP')['status'] == 'complete_recovery_scientific_slot'
    row = read(out/'review.json')
    assert row['accepted'] and row['slot'] == index
    for artifact in (row['parent'], row['h200']):
        path = Path(artifact['path'])
        assert not path.is_symlink() and path.stat().st_nlink == 1 and path.stat().st_size == artifact['size']
    return row


def accept(index):
    out = slot_dir(index); row = read(out/'review.json')
    assert read(out/'guard.json')['status'] == 'finished'
    assert read(out/'STOP')['status'] == 'complete_recovery_scientific_slot'
    for artifact in (row['parent'], row['h200']):
        assert sha(Path(artifact['path'])) == artifact['sha256']
    names = ('attempt.json', 'result.json', 'artifacts.json', 'validation.json',
             'inference.json', 'review.json', 'guard.json', 'STOP', 'lineage.json')
    put(out/'acceptance.json', dict(accepted=True, slot=index, no_adoption=True,
        evidence={name: sha(out/name) for name in names}, original_failed_slot_not_reclassified=index == 14))
    return acceptance(index)
