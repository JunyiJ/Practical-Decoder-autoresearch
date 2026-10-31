"""Exact activation boundary cleanup under complete dispatch and stage reviews.

Only current-campaign audited successful pilots and losing screen pairs qualify.
Every attempt is permanent; partial removals halt continuation. No fresh, selected,
failed, unknown or external path is eligible, and no checkpoint is regenerated.
"""
import json
from pathlib import Path
import shutil
import time

from .protocol import digest
from .tokenizer_activation_session import reference, require
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_cleanup import _locks, _sync_directory, _check
from .tokenizer_wide_hybrid_retention import RESERVE
from .cpu_prerequisite_guard import CpuPrerequisiteGuard, violation

SECONDS=600


def _base(boundary):
    require(boundary in ('pilot','screen'),'No fresh confirmation cleanup')
    return 'evidence/.retention/'+('pilot' if boundary=='pilot' else 'lr')


def inventory(ledger, owned):
    return {name:reference(ledger.root,name) for name in snapshot(ledger.root)['files']
            if not name.startswith(owned+'/')}


def verify_execution(ledger, state, boundary):
    """Verify terminal journal/guard/inventory proof; Session verifies cumulative live pins."""
    base=_base(boundary);receipt_ref=reference(ledger.root,base+'/activation-complete.json')
    require(not any(p.name.endswith('.tmp') or 'failure' in p.name for p in (ledger.root/base).rglob('*')),
            'Failed/partial retention cannot authorize progress')
    receipt=ledger.document(receipt_ref)
    preserved={r['path']:r for r in receipt['preserved']}
    names=('manifest.json','inventory.json','deletions.jsonl','guard.json','execution.json')
    require(all(base+'/'+n in preserved for n in names),'Complete retention execution proof required')
    for n in names:ledger.verify(preserved[base+'/'+n])
    docs={n:ledger.document(preserved[base+'/'+n]) for n in names if n!='deletions.jsonl'}
    manifest=docs['manifest.json'];original=docs['inventory.json'];execution=docs['execution.json']
    require(manifest.get('kind')=='activation_retention_manifest_v1' and manifest.get('boundary')==boundary
            and manifest.get('plan')==ledger.plan and manifest.get('reviews')==receipt['reviews']
            and manifest.get('selection')==receipt['selection'], 'Retention stage binding differs')
    deleted={r['path']:r for r in receipt['deleted']}
    require(manifest['delete']==receipt['deleted'] and len(deleted)==(4 if boundary=='pilot' else 8)
            and manifest['inventory_sha256']==preserved[base+'/inventory.json']['sha256'], 'Exact allowlist differs')
    require(all(original.get(name)==ref for name,ref in deleted.items()),'Deletion not in original inventory')
    require(manifest.get('reason')==('successful_resource_pilot' if boundary=='pilot' else 'complete_screen_loser'),
            'Deletion reason differs')
    # The immutable ledger independently derives eligible names using the saved
    # complete stage and LR decision, including after later stages have run.
    events=[json.loads(line) for line in (ledger.root/base/'deletions.jsonl').read_text().splitlines()]
    expected=[dict(event=event,entry=ref) for ref in receipt['deleted'] for event in ('intent','deleted')]
    expected.append(dict(event='preserved_verified',files=len(original)-len(deleted),inventory_sha256=manifest['inventory_sha256']))
    require(events==expected,'Incomplete or altered deletion journal')
    guard=docs['guard.json'];limits=dict(rss_limit_bytes=4*1024**3,reserve_bytes=RESERVE,timeout_seconds=SECONDS)
    require(guard.get('status')=='finished' and guard.get('reason') is None and guard.get('limits')==limits
            and violation(guard['observation'],guard['elapsed_seconds'],**limits) is None,'Cleanup guard incomplete')
    require(execution==dict(kind='activation_retention_execution_v1',status='completed',boundary=boundary,
        manifest=preserved[base+'/manifest.json'],inventory=preserved[base+'/inventory.json'],
        journal=preserved[base+'/deletions.jsonl'],guard=preserved[base+'/guard.json'],
        deleted_files=len(deleted),deleted_logical_bytes=sum(r['size'] for r in deleted.values()),
        preserved_files=len(original)-len(deleted)), 'Terminal execution proof differs')
    for name,ref in original.items():
        if name not in deleted:
            require(preserved.get(name)==ref,'Original preserved file omitted or changed')
    return receipt_ref


def verify_boundaries(ledger,state):
    for boundary in ('pilot','screen'):
        base=_base(boundary)
        if state[boundary+'_boundary']:
            verify_execution(ledger,state,boundary)
        else:
            require(not (ledger.root/base).exists(),'Existing incomplete retention attempt cannot be retried')


def execute_boundary(ledger):
    from .tokenizer_activation_entry import bind,idle,prefix
    bind(ledger);idle();state=ledger.inspect();prefix(ledger,state);verify_boundaries(ledger,state)
    action=state['next']['kind']
    require(action in ('review_pilot_boundary','review_screen_boundary'),'Complete audited stage required')
    boundary='pilot' if action=='review_pilot_boundary' else 'screen';base=_base(boundary)
    candidates=ledger._eligible(state,boundary)
    require(len(candidates)==(4 if boundary=='pilot' else 8),'Exact complete boundary pair count required')
    # Acceptance must originate from the activation independent reviewer and
    # every earlier child must have completed the dispatch/parent guard chain.
    for review_ref in state['reviews']:
        review=ledger.document(review_ref)
        for ref in review['audits'].values():
            require(ledger.document(ref).get('reviewer')=='activation_independent_v1','Independent scientific audit missing')
    require(not (ledger.root/base).exists(),'Existing cleanup attempt cannot be retried')
    # Directory creation is a permanent no-reentry marker before any removal.
    owned=ledger.root/base;owned.mkdir(parents=True,exist_ok=False);_sync_directory(owned.parent)
    removed=[];deadline=time.monotonic()+SECONDS
    context=digest(dict(plan=ledger.plan,reviews=state['reviews'],selection=state['selection_ref'],delete=candidates))
    def recheck():
        _locks(ledger.root/'evidence',ledger.writer.lock_fds);bind(ledger);idle()
        require(time.monotonic()<deadline and shutil.disk_usage(ledger.root).free>=RESERVE,'Retention time/disk guard')
        require(digest(dict(plan=ledger.plan,reviews=state['reviews'],selection=state['selection_ref'],delete=candidates))==context,
                'Cleanup context changed')
    try:
        with CpuPrerequisiteGuard(owned/'guard.json',ledger.root,timeout_seconds=SECONDS):
            recheck();before=inventory(ledger,base)
            require(all(before.get(k)==v for k,v in candidates.items()),'Candidate differs from accepted original hash')
            for name in candidates:
                path=ledger.root/name
                require(path.suffix=='.pt' and path.parent.name=='checkpoints'
                        and name.startswith('evidence/'+('pilots/' if boundary=='pilot' else 'quality/')),
                        'Only exact current-stage checkpoints are eligible')
            ledger.writer.publish_json(base+'/inventory.json',before)
            manifest=dict(kind='activation_retention_manifest_v1',boundary=boundary,plan=ledger.plan,
                reviews=state['reviews'],selection=state['selection_ref'],delete=list(candidates.values()),
                reason='successful_resource_pilot' if boundary=='pilot' else 'complete_screen_loser',
                inventory_sha256=reference(ledger.root,base+'/inventory.json')['sha256'])
            ledger.writer.publish_json(base+'/manifest.json',manifest)
            manifest_ref=reference(ledger.root,base+'/manifest.json')
            with ledger.writer.open_append(base+'/deletions.jsonl') as journal:
                for name,ref in candidates.items():
                    recheck();ledger.verify(manifest_ref)
                    path,stamp=_check(ledger.root,name,{k:ref[k] for k in ('size','sha256')},deadline)
                    journal.write((json.dumps(dict(event='intent',entry=ref),sort_keys=True)+'\n').encode())
                    now=path.stat()
                    require((now.st_dev,now.st_ino,now.st_size,now.st_mtime_ns)==stamp and not path.is_symlink(),
                            'Checkpoint changed before unlink')
                    path.unlink();_sync_directory(path.parent)
                    journal.write((json.dumps(dict(event='deleted',entry=ref),sort_keys=True)+'\n').encode());removed.append(ref)
                recheck()
                after=inventory(ledger,base)
                require(after=={k:v for k,v in before.items() if k not in candidates},'Preserved evidence changed')
                journal.write((json.dumps(dict(event='preserved_verified',files=len(after),
                    inventory_sha256=manifest['inventory_sha256']),sort_keys=True)+'\n').encode())
        recheck()
        guard=ledger.document(reference(ledger.root,base+'/guard.json'))
        require(guard['status']=='finished' and guard['reason'] is None,'Cleanup guard did not finish')
        refs={n:reference(ledger.root,base+'/'+n) for n in ('manifest.json','inventory.json','deletions.jsonl','guard.json')}
        execution=dict(kind='activation_retention_execution_v1',status='completed',boundary=boundary,
            manifest=refs['manifest.json'],inventory=refs['inventory.json'],journal=refs['deletions.jsonl'],guard=refs['guard.json'],
            deleted_files=len(removed),deleted_logical_bytes=sum(r['size'] for r in removed),preserved_files=len(after))
        ledger.writer.publish_json(base+'/execution.json',execution)
        preserved=list(after.values())+list(refs.values())+[reference(ledger.root,base+'/execution.json')]
        receipt=dict(kind='activation_retention_v1',status='completed',boundary=boundary,reviews=state['reviews'],
                     selection=state['selection_ref'],deleted=removed,preserved=preserved)
        ledger.writer.publish_json(base+'/activation-complete.json',receipt)
        ref=verify_execution(ledger,state,boundary)
        ledger.record('boundary',ref)
        return ref
    except BaseException as exc:
        try:ledger.writer.publish_json(base+'/failure.json',dict(status='failed',error=repr(exc)[:1000],
            confirmed_deleted=removed,retry_allowed=False))
        except BaseException as error:exc.add_note('Retention failure publication: '+repr(error)[:1000])
        raise
