"""Read-only full remaining-campaign storage gate; no cleanup or dispatcher.

Call at an idle audited trial boundary under worker locks and an outer guard.
Occupied checkpoint capacity is not free space. Future phase accounting assumes
only the frozen, audited cleanup transitions; each subsequent boundary rechecks
actual free bytes and complete receipts before progressing. No deletion receives
credit merely from a planned manifest or a missing file.
"""
import shutil
import time
from pathlib import Path

from .protocol import digest
from .tokenizer_wide_hybrid_retention import allocations,storage_envelope,METADATA,RESERVE,boundary_candidates
from .tokenizer_wide_hybrid_retention_receipts import _path,_sha,planned_manifest
from .tokenizer_wide_hybrid_retention_chain import recheck_chain
from .tokenizer_wide_hybrid_cleanup import _locks,_scan,_check,TIMEOUT
from .tokenizer_wide_hybrid_metadata import snapshot
from .tokenizer_wide_hybrid_pilot_config import specs as pilot_specs
from . import tokenizer_wide_hybrid_rules as rules


def remaining_projection(projection,pilots,records,rates,selection,boundaries):
    """Validate ordered completion/selection and calculate conservative peaks.

    boundaries is a verified cleanup-prefix length in the caller, not a deletion
    claim accepted by this pure arithmetic helper. Current completed checkpoints
    still occupy disk and remain protected until an actual receipt is checked.
    """
    budgets=allocations(projection);envelope=storage_envelope(projection)
    if type(boundaries) is not int or not 0<=boundaries<=3:raise ValueError('Invalid cleanup prefix')
    if not isinstance(pilots,list) or not 0<=len(pilots)<=3 or not isinstance(records,list):
        raise ValueError('Ordered completed records required')
    for row,spec in zip(pilots,pilot_specs()):rules._checked([row],[spec])
    if boundaries==0:
        if records or rates is not None or selection is not None:raise ValueError('Quality cannot precede pilot cleanup')
        phase='pilot';outstanding=pilot_specs()[len(pilots):]
    else:
        if len(pilots)!=3:raise ValueError('All pilot audits must precede quality')
        limits={1:(0,9),2:(9,15),3:(15,24)}
        low,high=limits[boundaries]
        if not low<=len(records)<=high:raise ValueError('Record count crosses an unverified cleanup boundary')
        queue=rules.lr_specs()
        for row,spec in zip(records[:9],queue):rules._checked([row],[spec])
        if len(records)<9 and rates is not None:raise ValueError('Premature LR selection')
        if rates is not None and digest(rates)!=digest(rules.choose_rates(records[:9])):
            raise ValueError('Persisted LR choice differs')
        if boundaries>=2:
            if rates is None:raise ValueError('Persisted LR choice required')
            queue+=rules.rank_specs(rates)
            for row,spec in zip(records[9:15],queue[9:]):rules._checked([row],[spec])
        if boundaries<2 or len(records)<15:
            if selection is not None:raise ValueError('Premature rank selection')
        elif selection is not None and digest(selection)!=digest(rules.choose_ranks(records[:9],records[9:15])):
            raise ValueError('Persisted rank choice differs')
        if boundaries==3:
            if selection is None:raise ValueError('Persisted final selection required')
            queue+=rules.confirmation_specs(selection)
            for row,spec in zip(records[15:],queue[15:]):rules._checked([row],[spec])
        phase=('lr','rank','confirm')[boundaries-1]
        if boundaries==3 and len(records)==24:phase='inference'
        outstanding=queue[len(records):]
    phase_totals=dict(envelope['phase_checkpoint_allowance_bytes'])
    if selection is not None:
        phase_totals['confirm']=4*sum(budgets[f,selection[f]['divisor']] for f in rules.FAMILIES)
    order=['pilot','lr','rank','confirm','inference'];index=order.index(phase)
    future={p:phase_totals[p] for p in order[index:] if p in phase_totals}
    return dict(phase=phase,remaining_current_phase_allowance_bytes=sum(budgets[s['family'],s['divisor']] for s in outstanding),
        remaining_phase_peak_bytes=future,remaining_peak_bytes=max(future.values(),default=0),
        reserve_bytes=RESERVE,metadata_allowance_bytes=METADATA,
        initial_full_panel_required_free_bytes=envelope['required_free_bytes'],
        cleanup_prefix_length=boundaries,arithmetic_only=True,dispatch_authorized=False)


def required_free(projection,occupied_checkpoint_bytes):
    """Keep the whole metadata budget; never discount current metadata writes."""
    if type(occupied_checkpoint_bytes) is not int or occupied_checkpoint_bytes<0:
        raise ValueError('Verified nonnegative occupied capacity required')
    current=projection['remaining_current_phase_allowance_bytes']
    future=max(0,projection['remaining_peak_bytes']-occupied_checkpoint_bytes)
    return RESERVE+METADATA+max(current,future)


def bind_storage(campaign,projection,*,pilots,records,rates,selection,stages,
                 current_inventory,checkpoint_pairs,lock_fds):
    """Verify receipts, all current evidence and actual disk at an idle boundary.

    Inputs must come from independent frozen/audited controller provenance.
    checkpoint_pairs contains ALL completed original pairs, including deleted
    ones: keys pilot/00..02 and quality/00..23, values raw/h200 path,size,sha256
    relative to evidence/. It cannot excuse unreviewed or failed output.
    """
    root=Path(campaign).absolute()
    if root!=root.resolve() or not root.is_dir():raise ValueError('Canonical campaign required')
    evidence=_path(root,'evidence');_locks(evidence,lock_fds)
    if _path(root,'STOP').exists():raise ValueError('STOP prevents storage acceptance')
    if not isinstance(stages,list):raise ValueError('Complete cleanup prefix required')
    state=remaining_projection(projection,pilots,records,rates,selection,len(stages))
    if not isinstance(current_inventory,dict):raise ValueError('Complete current inventory required')
    def stamps():
        result={}
        for name in current_inventory:
            st=_path(evidence,name).stat()
            result[name]=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns,st.st_blocks)
        return result
    initial_stamps=stamps()
    expected_keys={f'pilot/{i:02d}' for i in range(len(pilots))}|{f'quality/{i:02d}' for i in range(len(records))}
    if set(checkpoint_pairs)!=expected_keys:raise ValueError('All completed original checkpoint pairs required')
    all_entries={};slots=set()
    for key,pair in checkpoint_pairs.items():
        if set(pair)!={'raw','h200'}:raise ValueError('Complete raw/H200 pair required')
        stage,slot=key.split('/');stem=f"{'pilots' if stage=='pilot' else 'quality'}/{slot}";slots.add(stem)
        for entry in pair.values():
            if set(entry)!={'path','size','sha256'}:raise ValueError('Exact original checkpoint binding required')
            if type(entry['size']) is not int or entry['size']<=0:raise ValueError('Positive original checkpoint size required')
            _sha(entry['sha256'])
            path=_path(evidence,entry['path'])
            if path.parent!=evidence/stem/'checkpoints' or path.suffix!='.pt' or entry['path'] in all_entries:
                raise ValueError('Checkpoint alias or wrong declared slot')
            all_entries[entry['path']]=entry
        rows=pilots if stage=='pilot' else records
        spec=rows[int(slot)]
        if sum(e['size'] for e in pair.values())>allocations(projection)[spec['family'],spec['divisor']]:
            raise ValueError('Completed checkpoint pair exceeds trajectory allowance')
    for index,stage in enumerate(stages):
        boundary=('pilot','lr','rank')[index]
        rows=pilots if index==0 else records[:9 if index==1 else 15]
        rr=None if index==0 else rates;ss=selection if index==2 else None
        indices=boundary_candidates(boundary,rows,rr,ss)['audit_required_indices']
        pairs={i:checkpoint_pairs[f"{'pilot' if index==0 else 'quality'}/{i:02d}"] for i in indices}
        manifest=stage['manifest']
        expected=planned_manifest(boundary,rows,rr,ss,pairs,manifest['evidence'])
        if digest(expected)!=digest(manifest):raise ValueError('Cleanup manifest differs from audited stage/selection')
    if stages:
        chain=recheck_chain(evidence,stages,current_inventory)
        deleted=chain['deleted_files']
    else:
        attempts=_path(evidence,'.retention')
        if attempts.exists() and any(attempts.iterdir()):raise ValueError('Unreviewed cleanup attempt blocks storage gate')
        if _scan(evidence,None)!=set(current_inventory):raise ValueError('Complete current evidence inventory required')
        deadline=time.monotonic()+TIMEOUT
        for name,binding in current_inventory.items():_check(evidence,name,binding,deadline)
        deleted={}
    live={name:entry for name,entry in all_entries.items() if name not in deleted}
    if any(name not in all_entries or digest(entry)!=digest(dict(all_entries[name],index=entry['index'],role=entry['role']))
           for name,entry in deleted.items()):
        raise ValueError('Deleted original pair bindings differ')
    census=snapshot(root)
    actual_weights={name.removeprefix('evidence/') for name,row in census['files'].items() if row['group'].endswith('/weights')}
    if actual_weights!=set(live):raise ValueError('Unexpected, missing, failed or unreviewed checkpoint output')
    for name in census['files']:
        parts=Path(name).parts
        if len(parts)>=3 and parts[0]=='evidence' and parts[1] in ('pilots','quality'):
            if '/'.join(parts[1:3]) not in slots:raise ValueError('Unreviewed trajectory attempt blocks storage acceptance')
    occupied=0;logical=0;seen=set()
    for name,entry in live.items():
        if current_inventory.get(name)!={'size':entry['size'],'sha256':entry['sha256']}:
            raise ValueError('Current checkpoint differs from original audited pair')
        st=_path(evidence,name).stat();inode=(st.st_dev,st.st_ino)
        if st.st_nlink!=1 or inode in seen:raise ValueError('Checkpoint inode alias refused')
        seen.add(inode);logical+=st.st_size
        # Sparse/compressed logical bytes are never credited as occupied capacity.
        # Physical release is still checked from actual free space after cleanup.
        occupied+=min(st.st_size,st.st_blocks*512)
    if logical!=census['checkpoint_logical_bytes']:raise ValueError('Checkpoint census changed')
    needed=required_free(state,occupied);free=shutil.disk_usage(root).free
    if type(free) is not int or free<0:raise ValueError('Exact free-byte observation required')
    if free<needed:raise ValueError(f'Complete remaining panel does not fit: free={free}, required={needed}')
    # No changing worker is allowed here: reject a changed census/STOP/lock before
    # returning. The controller must still rebind source, plan and stage audits.
    _locks(evidence,lock_fds)
    if (_path(root,'STOP').exists() or digest(snapshot(root))!=digest(census)
            or stamps()!=initial_stamps):
        raise ValueError('Campaign changed during idle storage check')
    free=shutil.disk_usage(root).free
    if type(free) is not int or free<needed:raise ValueError('Disk space changed before storage acceptance')
    return dict(status='remaining_campaign_storage_verified',projection=state,free_bytes=free,
        required_free_bytes=needed,occupied_checkpoint_capacity_bytes=occupied,
        checkpoint_logical_bytes=logical,metadata_bytes=census['metadata_bytes'],
        cleanup_boundaries_verified=len(stages),current_inventory_sha256=digest(current_inventory),
        storage_projection_fits=True,current_phase_space_verified=True,
        frozen_plan_verified=False,audit_semantics_verified=False,dispatch_authorized=False)
