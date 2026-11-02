"""Exact fixed-LR successful-pilot retention boundary over the exercised bounded executor.

Only caller-audited complete pilots qualify. All six fresh endpoints are protected. This module never
accepts fresh confirmation indices or authorizes training. Real callers must
independently audit payloads, inference and saved selection before authorizing.
"""
import json,os,time,shutil
from pathlib import Path
from src.research.protocol import digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_retention import VERSION,RESERVE
from src.research.tokenizer_wide_hybrid_retention_receipts import _path,_sha,_relative
from src.research.tokenizer_wide_hybrid_cleanup import (_locks,_scan,_check,_save,
    _sync_directory,TIMEOUT,JOURNAL_CAP,ATTEMPT_CAP,MAX_FILES,recheck_execution)
from .evidence_contract import require
ARMS=('unit','fixed_half')

def planned_manifest(boundary,records,selection,pairs,evidence):
    require(boundary=='pilot','No fresh-confirmation cleanup')
    expected=[(a,.00125) for a in ARMS]
    require(type(records) is list and len(records)==len(expected),'Complete accepted stage required')
    for i,(row,(arm,lr)) in enumerate(zip(records,expected)):
        require(set(row)=={'slot','arm','lr','status','h200_ce'} and type(row['slot']) is int and row['slot']==i and row['arm']==arm and type(row['lr']) is float and row['lr']==lr and row['status']=='completed','Exact accepted ordered stage identity required')
        import math
        require(type(row['h200_ce']) in (int,float) and math.isfinite(row['h200_ce']) and row['h200_ce']>=0,'Finite validation CE required')
    require(selection is None,'No pilot-loss or LR selection');selected=[]
    require('meta/pilot-inference.json' in evidence,'Completed pilot inference binding required')
    require(type(pairs) is dict and set(pairs)==set(range(len(expected))) and all(type(k) is int for k in pairs),'Every stage pair and no fresh pair required')
    require(type(evidence) is dict and evidence,'Bound immutable evidence required')
    for n,h in evidence.items():_relative(n);_sha(h)
    entries=[];seen=set(evidence)
    for slot in range(len(expected)):
        require(set(pairs[slot])=={'raw','h200'},'Exact raw/H200 pair required')
        for role in ('raw','h200'):
            item=pairs[slot][role];require(set(item)=={'path','size','sha256'},'Exact file identity required')
            n=_relative(item['path']);_sha(item['sha256'])
            stage='pilots'
            require(Path(n).parent==Path(f'{stage}/{slot:02d}/checkpoints') and Path(n).suffix=='.pt','Exact stage checkpoint location required')
            require(n not in seen and type(item['size']) is int and item['size']>0,'No aliased or empty weights')
            seen.add(n);entries.append(dict(index=slot,role=role,**item))
    return dict(version=VERSION,status='planned',boundary=boundary,records_sha256=digest(records),selection_sha256=digest(selection),evidence=evidence,
        delete=[e for e in entries if e['index'] not in selected],retain=[e for e in entries if e['index'] in selected])

def execute_boundary(root, boundary, lock_fds, authorize, revalidate):
    """Apply one independently authorized boundary under inherited worker locks.

    authorize() must run the future full decay payload/inference gates and return exactly
    {'inputs': planned_manifest keyword arguments, 'inventory': complete immutable
    root relative-path -> {size,sha256}}. revalidate() must recheck frozen source,
    plan, persisted selections, STOP and idle-worker state without requiring
    already-deleted weights. Its returned identity must stay unchanged.
    Neither callback is a user-configurable bypass; they are controller code.
    No production implementation of those complete gates is wired here yet.
    """
    root = Path(root).resolve()
    if boundary!='pilot': raise ValueError('No confirmation cleanup boundary')
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

