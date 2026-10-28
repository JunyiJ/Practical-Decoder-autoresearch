"""One-shot, user-authorized cleanup of enumerated closed pilot/losing-screen weights.

plan hashes all remaining tokenizer weights and all non-checkpoint evidence in
those run directories. apply requires the unchanged manifest, rechecks every
file before deleting, holds the shared worker lock, and journals each deletion.
No model loading, corpus read, controller run, or recursive directory deletion.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
MANIFEST = OUT/'manifest.json'
PILOTS = ('pilots', 'gqa-pilots', 'horizon-pilot', 'mla-pilot', 'hybrid-pilot', 'table-pilot', 'wide-pilot')
QUALITY = ('table', 'wide-attention')
START = time.monotonic()


def sha(p):
    assert time.monotonic()-START < 600, 'Cleanup time bound reached'
    h = hashlib.sha256()
    with p.open('rb') as s:
        for block in iter(lambda: s.read(4*1024**2), b''): h.update(block)
    return h.hexdigest()


def read(p): return json.loads(p.read_text())
def rel(p): return str(p.relative_to(ROOT))
def now(): return datetime.now(timezone.utc).isoformat()

def item(p):
    assert not p.is_symlink() and p.is_file()
    assert p.resolve().is_relative_to(ROOT/'research_runs')
    return dict(path=rel(p), bytes=p.stat().st_size, sha256=sha(p))


def dump(p, d):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(d, indent=2)+'\n')
    tmp.replace(p)


def checkpoints(d):
    if 'checkpoints' in d:
        return [(x['kind'], Path(x['checkpoint']), x['sha256']) for x in d['checkpoints']]
    if 'raw_checkpoint' in d:
        return [(k, Path(d[k+'_checkpoint']), d[k+'_sha256']) for k in ('raw','h200')]
    return [(k, Path(d[k]['checkpoint']), d[k]['checkpoint_sha256']) for k in ('raw','h200')]


def plan():
    assert not MANIFEST.exists(), 'Existing cleanup evidence must not be overwritten'
    candidates, bindings = {}, {}
    for name in PILOTS+QUALITY:
        root = ROOT/'research_runs'/('tokenizer-'+name+'-001')
        state = read(root/'state.json')
        assert state['status'] == 'completed' and state.get('active') is None
        bindings[rel(root/'state.json')] = sha(root/'state.json')
        ev = ROOT/'research/tokenizer-evidence'/(name+'-001')
        review_path = ev/('review.json' if name in PILOTS else 'final-review.json')
        review = read(review_path)
        assert review['status'] == 'passed' if name in PILOTS else review['status'].startswith('completed_')
        bindings[rel(review_path)] = sha(review_path)
        selected = None
        if name in QUALITY:
            selected = read(root/'selection.json')
            assert selected == review['selection']
            bindings[rel(root/'selection.json')] = sha(root/'selection.json')
            tensor_path = ev/'tensor-audit.json'
            assert sha(tensor_path) == review['tensor_audit_sha256']
            bindings[rel(tensor_path)] = sha(tensor_path)
            audits = {r['id']:r for r in read(tensor_path)['records']}
        for p in sorted(root.rglob('result.json')):
            d = read(p)
            # Only actual trajectory results, never timing or other result files.
            if d.get('status') != 'completed' or d.get('holdout_consumed') is not False: continue
            if name in QUALITY:
                if d.get('stage') != 'screen' or d['lr'] == selected[d['family']]: continue
                a = audits[d['id']]
                assert a['all_saved_float_tensors_finite'] and a['export_exactly_matches_saved_ema_and_raw_buffers']
                reason = 'Closed completed screen rejected by saved per-family LR selection; fresh confirmation and selected screen weights retained.'
            else:
                reason = 'Successful closed resource pilot, superseded by fully audited quality round; not quality-selection or failure evidence.'
            for kind, path, expected in checkpoints(d):
                assert path.parent == p.parent/'checkpoints'
                rec = item(path)
                assert rec['sha256'] == expected
                if name in QUALITY: assert expected == a[kind+'_sha256']
                rec.update(kind=kind, result_path=rel(p), result_sha256=sha(p), reason=reason)
                assert rec['path'] not in candidates
                candidates[rec['path']] = rec
    roots = sorted(p for p in (ROOT/'research_runs').glob('tokenizer*') if p.is_dir())
    protected, metadata = [], []
    for root in roots:
        for p in sorted(root.rglob('*')):
            if not p.is_file() or '__pycache__' in p.parts or p.name == '.lock': continue
            if p.suffix == '.pt':
                if rel(p) not in candidates: protected.append(item(p))
            else: metadata.append(item(p))
    assert len(candidates) == 42, (len(candidates), 'Unexpected allowlist size')
    d = dict(status='planned', created_at=now(), authorization='Human, October 28 2026 Los Angeles: feel free to clean up storage with non-critical check point moving forward! instead of silently blocked by it',
             scope='Seven successful closed tokenizer pilot rounds plus seven unselected table/wide LR screens. Preserve all selected screens and every fresh confirmation raw/H200 pair, all failures and all non-checkpoint evidence.',
             limitation='Deleted weights cannot be loaded or tensor-audited again. Historical audit receipts, hashes and metrics remain; never regenerate weights or rerun closed controllers.',
             free_bytes_before=shutil.disk_usage(ROOT).free, reserve_bytes=8*1024**3,
             delete_bytes=sum(r['bytes'] for r in candidates.values()), delete_files=list(candidates.values()),
             protected_checkpoints=protected, preserved_metadata=metadata, bindings=bindings, deleted_paths=[])
    dump(MANIFEST,d)
    print(json.dumps(dict(status=d['status'],delete_files=len(candidates),delete_bytes=d['delete_bytes'],protected_files=len(protected),metadata_files=len(metadata),manifest_sha256=sha(MANIFEST)),indent=2),flush=True)


def apply(expected):
    assert sha(MANIFEST) == expected
    d=read(MANIFEST)
    assert d['status'] == 'planned' and not d['deleted_paths']
    for p,h in d['bindings'].items(): assert sha(ROOT/p) == h
    for rec in d['delete_files']+d['protected_checkpoints']+d['preserved_metadata']:
        p=ROOT/rec['path']; assert p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256']
    d.update(status='deleting',started_at=now()); dump(MANIFEST,d)
    for rec in d['delete_files']:
        p=ROOT/rec['path']; assert not p.is_symlink() and p.suffix == '.pt' and p.parent.name == 'checkpoints'
        assert p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256']
        p.unlink(); d['deleted_paths'].append(rec['path']); dump(MANIFEST,d)
    for rec in d['protected_checkpoints']+d['preserved_metadata']:
        p=ROOT/rec['path']; assert p.stat().st_size == rec['bytes'] and sha(p) == rec['sha256']
    assert all(not (ROOT/r['path']).exists() for r in d['delete_files'])
    d.update(status='completed',completed_at=now(),free_bytes_after=shutil.disk_usage(ROOT).free,
             verification=dict(retained_checkpoint_hashes_unchanged=True,non_checkpoint_evidence_hashes_unchanged=True,all_allowlisted_weights_absent=True))
    dump(MANIFEST,d)
    print(json.dumps({k:d[k] for k in ['status','delete_bytes','free_bytes_before','free_bytes_after','verification']},indent=2),flush=True)


if __name__ == '__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['plan','apply']); ap.add_argument('--manifest-sha256');args=ap.parse_args()
    lock=ROOT/'research_state/tokenizer-worker-lock/.lock'
    with lock.open('a+b') as s:
        fcntl.flock(s.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.mode == 'plan': plan()
        else: apply(args.manifest_sha256)
