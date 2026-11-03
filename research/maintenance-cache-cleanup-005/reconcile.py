"""Record final capacity and a narrow, concurrent-safe retention correction."""
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from retire import ROOT, HERE, bound, digest, dump, identity


def main():
    receipt = json.loads((HERE/'completed.json').read_text())
    assert receipt['status']=='completed'
    plan_path=ROOT/'research/tokenizer-evidence/adamw-epsilon-preparation-001/quality001-plan.json'
    plan=json.loads(plan_path.read_text())
    new_path=ROOT/'research/tokenizer-evidence/storage-recovery-028/raw-completed.json'
    new=json.loads(new_path.read_text())
    fresh=[]
    for row in new['files']:
        bound(row['acceptance'])
        events=[json.loads(line) for line in Path(row['acceptance']['path']).read_text().splitlines()]
        assert len(events)==1
        event=events[0]
        assert event['artifact_acceptance']['accepted']
        spec=event['request']['trial']['spec']
        assert spec in plan['fresh'] and spec['seed'] in plan['seeds']['fresh']
        assert spec['seed']==event['request']['config']['training']['seed']
        files=event['artifact_acceptance']['files']
        assert files['parent']['sha256']==row['original']['sha256']
        assert files['h200']['sha256']==row['h200']['sha256']
        fresh.append(dict(archive=row['archive'],acceptance=row['acceptance'],
                          spec=spec,classification='fresh_confirmation'))
    assert len(fresh)==6
    dump('fresh-epsilon-supplement.json',dict(rows=fresh,plan=dict(path=str(plan_path),sha256=digest(plan_path)),
        archive_receipt=dict(path=str(new_path),sha256=digest(new_path)),
        conclusion='All six are accepted fresh confirmation endpoints, including both adopted and control arms. Preserve all.'))

    # Exact filesystem census uses unique device/inode identities. No hardlink
    # or already-retained endpoint is counted as newly reclaimable capacity.
    census=[]
    for p in (ROOT/'research_archives').rglob('*'):
        if p.is_file():
            census.append(identity(p))
    unique={(r['device'],r['inode']):r for r in census}
    counts=collections.Counter(Path(r['path']).parent.name for r in census)
    assert len(census)==204 and sum(v for k,v in counts.items() if k.startswith('protected-raw-storage-'))==202
    dump('remaining-archive-census.json',dict(files=census,unique_files=len(unique),
         allocated_bytes=sum(r['allocated_bytes'] for r in unique.values()),groups=dict(counts),
         retained_raw_classifications=dict(selected_screen=30,selected_final_rank=6,fresh_confirmation=166),
         other_archives='research-pytest-temp-004.tar.zst mixes original diagnostic evidence with fixtures; checkpoint-compaction-002/successful-diagnostics.tar.zst is 44KB. Neither was discarded merely because archived; this pass retires only the separately verified obsolete control payloads.'))

    resume_path=ROOT/'research/TOKENIZER_RESUME_STATE.json'
    before=resume_path.read_bytes()
    state=json.loads(before)
    free=shutil.disk_usage(ROOT).free
    reserve=8*1024**3;extra=2*1024**3
    predecessor=ROOT/'research/tokenizer-evidence/adamw-epsilon-preparation-001/quality001-capacity.json'
    capacity=json.loads(predecessor.read_text())
    scenario=6*(capacity['raw_cap_bytes']+capacity['h200_cap_bytes'])+capacity['round_metadata_cap_bytes']+reserve
    assert scenario==capacity['required_free_bytes']
    snapshot=dict(observed_unix=time.time(),actual_free_bytes=free,reserve_bytes=reserve,
        extra_headroom_bytes=extra,reserve_plus_headroom_target=reserve+extra,
        reserve_plus_headroom_deficit=max(0,reserve+extra-free),
        allocated_bytes_removed=receipt['allocated_bytes_removed'],
        observed_deletion_free_delta=receipt['observed_free_delta'],
        research_status=state['status'],active_run=state.get('active_run'),
        next_panel_measured_requirement=None,
        next_panel_status='Final-LayerNorm-scale full-model integration is still being prepared; no measured/frozen numerical panel. Preserve concurrent source work. Actual complete next-panel forecast remains required before numerical panel launch.',
        reference_full_panel=dict(required_free_bytes=scenario,with_extra_headroom=scenario+extra,
            deficit_with_extra_headroom=max(0,scenario+extra-free),
            source=dict(path=str(predecessor),sha256=digest(predecessor)),
            formula='6*(469762048 RAW + 117440512 H200) + 268435456 metadata including temporary overlap + 8589934592 reserve. Raw/H200 temporary publication is inside aggregate allowance; same-inode duplicate names add no allocation. Existing closed confirmations are not outputs of a new panel and give no new-panel credit.',
            caveat='Planning reference only; new integration/pilot/quality sizes must be measured, no partial panel permitted.'),
        memory_pressure=subprocess.check_output(['/usr/bin/memory_pressure','-Q'],text=True).strip(),
        swap=subprocess.check_output(['/usr/sbin/sysctl','vm.swapusage'],text=True).strip(),
        exhausted_scope='197 historical RAW archive identities audited using acceptance stages/final selections: one obsolete control removed;196 genuinely selected/fresh retained. Six new epsilon fresh archives retained. External unused package/cache classes already exhausted in cleanup003/004; no unchanged rescan.',
        wake_conditions=['New acceptance/dependency closure makes a specific retained payload disposable',
                         'New unused cache/installer material appears',
                         'Measured next panel establishes its exact full retained-output high-water budget',
                         'Approved storage expansion/offload changes available capacity'],
        escalation='Supply the recorded deficit in additional local free space, or choose an external volume for a separately verified offload of retained archives; selected/fresh bytes cannot be deleted under current preservation requirements.')
    dump('capacity-after.json',snapshot)

    old='All197 protected historical RAW archives/H100/H200 originals/storage004 intact.'
    replacement=('Historical archive re-audit maintenance-cache-cleanup-005 retired only the obsolete width64 gated-MLA losing-control RAW/H200 payloads after final KDA selection and closed dependencies; all196 genuinely selected/fresh historical RAW archives and associated retained models remain. Six additional fresh epsilon RAWs are losslessly archived by storage028:202 retained RAW archives total. Original evidence/storage004 intact; do not restore the two intentionally retired payloads.')
    assert old in state['next_action'], 'Concurrent next-action revision; inspect before reconciliation.'
    state['next_action']=state['next_action'].replace(old,replacement)
    state['actual_free_bytes']=free
    state['protected_lossless_raw_count']=202
    state['latest_archive_retirement']='research/maintenance-cache-cleanup-005/completed.json'
    state['latest_archive_retention_audit']='research/maintenance-cache-cleanup-005/archive-audit.json'
    state['latest_capacity_observation']='research/maintenance-cache-cleanup-005/capacity-after.json'
    after=(json.dumps(state,indent=2,sort_keys=True)+'\n').encode()
    temp=resume_path.with_name('TOKENIZER_RESUME_STATE.cleanup005.tmp')
    with temp.open('xb') as f:
        f.write(after);f.flush();os.fsync(f.fileno())
    assert resume_path.read_bytes()==before,'Concurrent resume update; no replacement performed.'
    os.replace(temp,resume_path)
    fd=os.open(str(resume_path.parent),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    dump('resume-reconciliation.json',dict(before_sha256=hashlib.sha256(before).hexdigest(),after_sha256=hashlib.sha256(after).hexdigest(),
        fields_changed=['next_action','actual_free_bytes','protected_lossless_raw_count','latest_archive_retirement','latest_archive_retention_audit','latest_capacity_observation'],
        concurrent_check='Shared worker lock held, original resume bytes revalidated immediately before atomic replace; all other fields preserved.',
        capacity_blocker_changed=False,reason='No current capacity blocker exists; numerical next-panel readiness remains unmeasured. Replaced the specific inaccurate all197-protected sentence and updated latest observation only.'))
    print(json.dumps(dict(free_bytes=free,headroom_deficit=snapshot['reserve_plus_headroom_deficit'],
                         reference_panel_deficit=snapshot['reference_full_panel']['deficit_with_extra_headroom'],retained_raw_archives=202)))


if __name__=='__main__':
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('r+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()
