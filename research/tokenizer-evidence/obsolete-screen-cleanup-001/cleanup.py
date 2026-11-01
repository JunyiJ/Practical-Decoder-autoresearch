"""User-authorized retirement of two obsolete screen archives, never fresh weights."""
import argparse, hashlib, json, os, shutil
from pathlib import Path
from research.ema_horizon_v1 import contract as c, execution as e
from research.block_norm_v1.fixture_dispatch import read, reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=Path(__file__).resolve().parent
OLD=c.ROOT/'research/tokenizer-evidence/ffn-structure-preparation-001/norm-pilot-retention-v2/manifest.json'
ARCH=c.ROOT/'research/tokenizer-evidence/checkpoint-compaction-001'
POLICY=c.ROOT/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
GROUPS=('campaign002-completed','campaign003-completed')
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(row):
    p=Path(row['path']);assert not p.is_symlink() and p.is_file(),p
    s=p.stat();assert s.st_size==row['size'] and digest(p)==row['sha256'],p
    assert s==p.stat(),p

def inventory():
    m=read(reference(OLD))
    return [r for refs in m['inventory_shards'].values() for ref in refs for r in read(ref)]
def actual_paths():
    out=set()
    for root in ('research_runs','research_archives'):
        for p in (c.ROOT/root).rglob('*'):
            if '__pycache__' in p.parts:continue
            assert not p.is_symlink(),p
            if p.is_file():out.add(str(p))
    return out
p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
with session_lock(GLOBAL_LOCK.parent):
    e.idle()
    with CpuPrerequisiteGuard(B/('apply-guard.json' if args.apply else 'audit-guard.json'),c.ROOT,timeout_seconds=120,poll_seconds=.25):
        rows=inventory();by_path={r['path']:r for r in rows}
        if not args.apply:
            assert actual_paths()==set(by_path)
            delete=[];receipts=[]
            for group in GROUPS:
                ref=reference(ARCH/(group+'-archive.json')); receipt=read(ref);receipts.append(ref)
                target=c.ROOT/receipt['archive'];row=by_path[str(target)]
                assert target.parent==c.ROOT/'research_archives/checkpoint-compaction-001'
                assert target.name==group+'.tar.zst'
                assert row['size']==receipt['archive_size'] and row['sha256']==receipt['archive_sha256']
                assert receipt['verified_before_removal'] is True
                campaign='002' if group==GROUPS[0] else '003'
                expected_slots=set(range(5 if campaign=='002' else 8));slots=set()
                for member,identity in receipt['members'].items():
                    path=c.ROOT/member;assert not path.exists()
                    rel=path.relative_to(c.ROOT/f'research_runs/tokenizer-wide-hybrid-{campaign}/evidence/quality')
                    assert len(rel.parts)==3 and rel.parts[1]=='checkpoints'
                    slots.add(int(rel.parts[0]));result=json.loads((path.parent.parent/'result.json').read_text())
                    endpoint=result['h200' if path.name=='h200.pt' else 'raw']
                    assert endpoint['checkpoint']==str(path) and endpoint['checkpoint_sha256']==identity['sha256']
                    assert result['holdout_consumed'] is False
                assert slots==expected_slots and len(receipt['members'])==2*len(slots)
                delete.append(row)
            for row in rows:verify(row)
            manifest=dict(status='audited',authorization='Human: find cleanup possibility - ok to remove old checkpoints. Narrow exception to archive retain-all; obsolete completed screening endpoints only.',
                delete_files=delete,archive_receipts=receipts,baseline_inventory=reference(OLD),source=reference(__file__),policy=reference(POLICY),
                deleted_bytes=sum(r['size'] for r in delete),free_bytes_before=shutil.disk_usage(c.ROOT).free,
                limitation='Permanent deletion of 26 archived completed screen endpoints. Metrics/configs/hashes remain; these weights can no longer be restored or re-evaluated. No selected, fresh, failed or numerical fixture weights removed.')
            e.exclusive(B/'manifest.json',manifest)
            print(json.dumps({k:manifest[k] for k in ('status','deleted_bytes','free_bytes_before')}),flush=True)
        else:
            m=read(reference(B/'manifest.json'))
            assert read(reference(B/'audit-guard.json'))['status']=='finished'
            assert m['source']==reference(__file__) and m['policy']==reference(POLICY)
            assert m['baseline_inventory']==reference(OLD)
            for ref in m['archive_receipts']:read(ref)
            remove={r['path'] for r in m['delete_files']}
            assert remove=={str(c.ROOT/'research_archives/checkpoint-compaction-001'/(g+'.tar.zst')) for g in GROUPS}
            assert actual_paths()==set(by_path)
            for row in rows:
                s=Path(row['path']).stat()
                assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['size'],row['mtime_ns'])
            for row in m['delete_files']:verify(row)
            with (B/'journal.jsonl').open('x') as log:
                for row in m['delete_files']:
                    log.write(json.dumps(dict(event='intent',file=row))+'\n');log.flush();os.fsync(log.fileno())
                    Path(row['path']).unlink()
                    log.write(json.dumps(dict(event='removed',path=row['path']))+'\n');log.flush();os.fsync(log.fileno())
            survivors=[r for r in rows if r['path'] not in remove]
            assert actual_paths()=={r['path'] for r in survivors}
            for row in survivors:verify(row)
            result=dict(status='completed',manifest=reference(B/'manifest.json'),deleted_bytes=m['deleted_bytes'],deleted_archives=sorted(remove),
                preserved_files_verified=len(survivors),preserved_checkpoints_and_archives=232,metadata_files_preserved=25974,
                free_bytes_after=shutil.disk_usage(c.ROOT).free,required_next_panel_free_bytes=15169995168,
                selected_fresh_failed_weights_unchanged=True,training_launched=False,holdout_read=False)
    if args.apply:
        result['guard']=reference(B/'apply-guard.json');e.exclusive(B/'completed.json',result);print(json.dumps(result),flush=True)
