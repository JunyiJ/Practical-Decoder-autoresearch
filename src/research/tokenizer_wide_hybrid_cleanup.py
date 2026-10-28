"""Internal v2 boundary-cleanup executor; no CLI, freeze or training launcher.

The future controller must supply trusted authorize/revalidate callbacks after
its full frozen-plan and stage-audit gates. The immutable evidence tree is kept
separate from mutable controller state/locks. Tests use disposable byte files.
Existing v1 controllers do not call this module. It never retries an attempt.
"""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

from .protocol import digest, file_digest, ROOT
from .cpu_prerequisite_guard import CpuPrerequisiteGuard
from .tokenizer_wide_hybrid_retention import VERSION, RESERVE
from .tokenizer_wide_hybrid_retention_receipts import _path, _sha, planned_manifest
from .tokenizer_wide_hybrid_simulation import _publish

GLOBAL_LOCK = ROOT/'research_state/tokenizer-worker-lock/.lock'
TIMEOUT = 600
JOURNAL_CAP = 1024**2
DOCUMENT_CAP = 1024**2
ATTEMPT_CAP = 5*1024**2
MAX_FILES = 2048


def _save(path, value):
    size = len((json.dumps(value,sort_keys=True,allow_nan=False,indent=2)+"\n").encode())
    if size > DOCUMENT_CAP:
        raise ValueError("Cleanup document exceeds 1MiB cap")
    existing = sum(p.stat().st_size for p in path.parent.rglob('*') if p.is_file())
    if existing+size > ATTEMPT_CAP: raise RuntimeError('Cleanup metadata cap exceeded')
    _publish(path,value,exclusive=True)


def _locks(root, fds):
    if len(fds) != 2 or len(set(fds)) != 2 or any(type(fd) is not int or fd <= 2 for fd in fds):
        raise ValueError('Distinct global/campaign lock descriptors required')
    inodes = []
    for fd, path in zip(fds, (GLOBAL_LOCK, root.parent/'.lock')):
        if path.is_symlink(): raise ValueError('Symlink lock refused')
        actual, expected = os.fstat(fd), path.stat()
        inode = (actual.st_dev, actual.st_ino)
        if inode != (expected.st_dev, expected.st_ino): raise ValueError('Worker lock identity differs')
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inodes.append(inode)
    if len(set(inodes)) != 2: raise ValueError('Lock inode aliases refused')


def _scan(root, owned):
    files = set()
    for path in root.rglob('*'):
        if path.is_relative_to(owned): continue
        if path.is_symlink(): raise ValueError('Evidence inventory contains a symlink')
        if path.is_dir(): continue
        if not path.is_file(): raise ValueError('Evidence inventory contains a nonregular file')
        files.add(str(path.relative_to(root)))
        if len(files) > MAX_FILES: raise ValueError('Cleanup inventory file-count cap exceeded')
    return files


def _check(root, name, expected, deadline):
    if set(expected) != {'size','sha256'} or type(expected['size']) is not int or expected['size'] < 0:
        raise ValueError('Exact file size/hash inventory required')
    _sha(expected['sha256']); path = _path(root, name)
    before = path.stat()
    if not path.is_file() or before.st_size != expected['size']:
        raise ValueError('Inventoried file missing or changed size')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            if time.monotonic() >= deadline: raise TimeoutError('Cleanup hashing time limit')
            h.update(block)
    after = path.stat()
    if (h.hexdigest() != expected['sha256'] or (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)
            != (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)):
        raise ValueError('Inventoried file changed while hashing')
    return path, (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def execute_boundary(root, boundary, lock_fds, authorize, revalidate):
    """Apply one independently authorized boundary under inherited worker locks.

    authorize() must run the future full v2 gates and return exactly
    {'inputs': planned_manifest keyword arguments, 'inventory': complete immutable
    root relative-path -> {size,sha256}}. revalidate() must recheck frozen source,
    plan, persisted selections, STOP and idle-worker state without requiring
    already-deleted weights. Its returned identity must stay unchanged.
    Neither callback is a user-configurable bypass; they are controller code.
    No production implementation of those complete gates is wired here yet.
    """
    root = Path(root).resolve()
    if boundary not in ('pilot','lr','rank'): raise ValueError('No confirmation cleanup boundary')
    _locks(root, lock_fds)
    owned = _path(root, '.retention/'+boundary)
    if owned.exists(): raise FileExistsError('Existing cleanup attempt is closed; no retry')
    if not callable(authorize) or not callable(revalidate): raise ValueError('Independent controller gates required')
    started = time.monotonic(); deadline = started+TIMEOUT
    identity = digest(revalidate())
    authorized = authorize()
    if set(authorized) != {'inputs','inventory'}: raise ValueError('Incomplete cleanup authorization binding')
    inputs = authorized['inputs']
    if inputs.get('boundary') != boundary: raise ValueError('Cleanup boundary changed')
    manifest = planned_manifest(**inputs)
    inventory = authorized['inventory']
    if not isinstance(inventory,dict) or not 0 < len(inventory) <= MAX_FILES: raise ValueError('Complete protected inventory required')
    candidates = {e['path']: e for e in manifest['delete']}
    if len(candidates) != len(manifest['delete']): raise ValueError('Duplicate deletion candidate')
    for e in manifest['delete']+manifest['retain']:
        if inventory.get(e['path']) != dict(size=e['size'],sha256=e['sha256']):
            raise ValueError('Checkpoint inventory differs from original audited hash')
    for name, sha in manifest['evidence'].items():
        if name not in inventory or inventory[name]['sha256'] != sha:
            raise ValueError('Bound non-checkpoint evidence omitted from inventory')
    for name in candidates:
        p = _path(root,name)
        if p.suffix != '.pt' or p.parent.name != 'checkpoints':
            raise ValueError('Only explicit checkpoint weights may be removed')
    if _scan(root,owned) != set(inventory): raise ValueError('Immutable evidence inventory incomplete')
    # The new attempt directory is a permanent no-reentry marker even on failure.
    owned.mkdir(parents=True,exist_ok=False); _sync_directory(owned.parent)
    journal = owned/'deletions.jsonl'
    removed = []
    def validate():
        _locks(root,lock_fds)
        if digest(revalidate()) != identity: raise ValueError('Frozen cleanup context changed')
        if time.monotonic() >= deadline: raise TimeoutError('Cleanup wall-time limit')
        paths = list(owned.rglob('*'))
        if any(p.is_symlink() for p in paths): raise ValueError('Cleanup output symlink')
        if sum(p.stat().st_size for p in paths if p.is_file()) > ATTEMPT_CAP:
            raise RuntimeError('Cleanup metadata cap exceeded')
        if shutil.disk_usage(root).free < RESERVE: raise RuntimeError('Cleanup disk reserve crossed')
    def emit(stream, value):
        data = (json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
        if stream.tell()+len(data) > JOURNAL_CAP: raise RuntimeError('Deletion journal cap exceeded')
        stream.write(data); stream.flush(); os.fsync(stream.fileno())
    try:
        with CpuPrerequisiteGuard(owned/'guard.json',root,timeout_seconds=TIMEOUT):
            validate()
            for name, entry in inventory.items(): _check(root,name,entry,deadline)
            _save(owned/'inventory.json',inventory)
            _save(owned/'manifest.json',manifest)
            manifest_sha = file_digest(owned/'manifest.json')
            with journal.open('xb') as stream:
                for name, entry in candidates.items():
                    validate()
                    if file_digest(owned/'manifest.json') != manifest_sha: raise ValueError('Cleanup manifest changed')
                    path, stamp = _check(root,name,inventory[name],deadline)
                    emit(stream,dict(event='intent',entry=entry))
                    now = path.stat()
                    if (now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns) != stamp or path.is_symlink():
                        raise ValueError('Checkpoint changed before unlink')
                    path.unlink(); _sync_directory(path.parent)
                    emit(stream,dict(event='deleted',entry=entry))
                    removed.append(entry)
                validate()
                expected_remaining = set(inventory)-set(candidates)
                if _scan(root,owned) != expected_remaining: raise ValueError('Post-cleanup inventory changed')
                for name in expected_remaining: _check(root,name,inventory[name],deadline)
                emit(stream,dict(event='protected_verified',files=len(expected_remaining),
                    inventory_sha256=file_digest(owned/'inventory.json')))
        validate()
        guard = json.loads((owned/'guard.json').read_text())
        if guard['status'] != 'finished' or guard['reason'] is not None: raise ValueError('Cleanup guard incomplete')
        receipt = dict(version=VERSION,status='completed',manifest_sha256=manifest_sha,
                       deleted=removed,protected_evidence_sha256=digest(manifest['evidence']))
        _save(owned/'receipt.json',receipt)
        references = {name:file_digest(owned/name) for name in
                      ('manifest.json','inventory.json','deletions.jsonl','guard.json','receipt.json')}
        review = dict(version=VERSION,status='completed',boundary=boundary,
            files=references,context_sha256=identity,deleted_count=len(removed),
            deleted_logical_bytes=sum(e['size'] for e in removed),
            protected_files_verified=len(inventory)-len(removed),free_bytes_after=shutil.disk_usage(root).free,
            limitation='Deleted weights cannot be reloaded; original audit evidence is retained.')
        # Parent must bind this final marker, not accept an earlier receipt alone.
        validate()
        _save(owned/'execution-review.json',review)
        return review
    except BaseException as exc:
        if not (owned/'failure.json').exists():
            _save(owned/'failure.json',dict(status='failed',error=repr(exc)[:1000],
                confirmed_deleted=removed,retry_allowed=False))
        raise


def recheck_execution(root, expected_manifest, execution_ref, context_sha256):
    """Verify the final marker, full journal, guard and all protected inventory.

    A receipt alone cannot prove that the executor finished post-verification.
    Frozen-plan/context authorization still belongs to the future controller.
    This is boundary-local: after a later authorized rank cleanup retires an LR
    anchor, a cumulative receipt-chain verifier is required instead.
    """
    from .tokenizer_wide_hybrid_retention_receipts import _document, recheck_boundary
    root = Path(root).resolve(); _sha(context_sha256)
    boundary = expected_manifest['boundary']
    if boundary not in ('pilot','lr','rank'): raise ValueError('Invalid execution boundary')
    base = '.retention/'+boundary
    if execution_ref['path'] != base+'/execution-review.json': raise ValueError('Wrong execution marker location')
    review = _document(root,execution_ref)
    names = ('manifest.json','inventory.json','deletions.jsonl','guard.json','receipt.json')
    if (review.get('version') != VERSION or review.get('status') != 'completed'
            or review.get('boundary') != boundary or review.get('context_sha256') != context_sha256
            or set(review.get('files',{})) != set(names)):
        raise ValueError('Final cleanup marker incomplete or from different context')
    refs = {name:dict(path=base+'/'+name,sha256=review['files'][name]) for name in names}
    documents = {name:_document(root,refs[name]) for name in names if name != 'deletions.jsonl'}
    recheck_boundary(root,expected_manifest,refs['manifest.json'],refs['receipt.json'])
    inventory = documents['inventory.json']; candidates = {e['path']:e for e in expected_manifest['delete']}
    for e in expected_manifest['delete']+expected_manifest['retain']:
        if inventory.get(e['path']) != dict(size=e['size'],sha256=e['sha256']):
            raise ValueError('Original execution inventory differs from checkpoint bindings')
    for name, sha in expected_manifest['evidence'].items():
        if name not in inventory or inventory[name]['sha256'] != sha:
            raise ValueError('Original execution inventory omits bound evidence')
    journal = _path(root,refs['deletions.jsonl']['path'])
    if journal.stat().st_size > JOURNAL_CAP or file_digest(journal) != refs['deletions.jsonl']['sha256']:
        raise ValueError('Deletion journal changed or exceeds cap')
    events = [json.loads(line) for line in journal.read_text().splitlines()]
    expected = [dict(event=kind,entry=e) for e in expected_manifest['delete'] for kind in ('intent','deleted')]
    expected.append(dict(event='protected_verified',files=len(inventory)-len(candidates),
                         inventory_sha256=refs['inventory.json']['sha256']))
    if digest(events) != digest(expected): raise ValueError('Deletion journal incomplete')
    guard = documents['guard.json']
    expected_limits = dict(rss_limit_bytes=4*1024**3,reserve_bytes=RESERVE,timeout_seconds=TIMEOUT)
    from .cpu_prerequisite_guard import violation
    sample = guard.get('observation', {})
    elapsed = guard.get('elapsed_seconds')
    if (type(elapsed) not in (int,float) or not math.isfinite(elapsed) or elapsed < 0
            or set(sample) != {'peak_rss_bytes','free_bytes'}
            or any(type(v) is not int or v < 0 for v in sample.values())):
        raise ValueError('Cleanup resource observation invalid')
    if (guard.get('status') != 'finished' or guard.get('reason') is not None
            or digest(guard.get('limits')) != digest(expected_limits)
            or violation(guard['observation'],guard['elapsed_seconds'],**expected_limits) is not None):
        raise ValueError('Cleanup resource guard incomplete')
    if (review.get('deleted_count') != len(candidates)
            or review.get('deleted_logical_bytes') != sum(e['size'] for e in candidates.values())
            or review.get('protected_files_verified') != len(inventory)-len(candidates)):
        raise ValueError('Execution counts differ')
    deadline = time.monotonic()+TIMEOUT
    for name in set(inventory)-set(candidates): _check(root,name,inventory[name],deadline)
    # Boundary-local verification. A later authorized retirement of an LR anchor
    # needs cumulative receipt-chain validation, not an arbitrary missing-file exception.
    _document(root,execution_ref)
    return dict(status='verified_completed_cleanup',deleted_files=len(candidates),
                protected_files_verified=len(inventory)-len(candidates),dispatch_authorized=False)
