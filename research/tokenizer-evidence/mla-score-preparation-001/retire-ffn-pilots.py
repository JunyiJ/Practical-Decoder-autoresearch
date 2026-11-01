"""Exact policy audit/removal of four completed non-selected pilot weights only."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
from research.ema_horizon_v1 import contract as c,execution as e
from research.block_norm_v1.fixture_dispatch import read,reference
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

B=Path(__file__).resolve().parent/'ffn-pilot-retention'
E=c.ROOT/'research/tokenizer-evidence/ffn-structure-preparation-001'
P=c.ROOT/'research_runs/tokenizer-ffn-structure-pilot-001'
def digest(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def record(path):
    assert not path.is_symlink() and path.is_file()
    before=path.stat();h=digest(path);after=path.stat()
    assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
    return dict(path=str(path),size=after.st_size,sha256=h,device=after.st_dev,inode=after.st_ino,mtime_ns=after.st_mtime_ns)
def paths():
    out=[]
    for name in ('research_runs','research_archives'):
        root=c.ROOT/name
        for p in sorted(root.rglob('*')):
            if '__pycache__' in p.parts:continue
            assert not p.is_symlink(),str(p)
            if p.is_file():out.append(p)
    return out

p=argparse.ArgumentParser();p.add_argument('--apply',action='store_true');args=p.parse_args()
B.mkdir(exist_ok=True)
with session_lock(GLOBAL_LOCK.parent):
    from research.ffn_quality_v1.support import idle
    idle()
    with CpuPrerequisiteGuard(B/('apply-guard.json' if args.apply else 'audit-guard.json'),c.ROOT,timeout_seconds=600,poll_seconds=.25):
        if not args.apply:
            assert not (B/'manifest.json').exists()
            result=read(reference(E/'pilot001-review.json'))
            closure=read(reference(E/'quality001-closure.json'))
            final=read(reference(E/'quality001-final-review.json'))
            assert result['status']=='both_resource_pilots_and_inference_accepted'
            assert closure['status']==final['status']=='closed_not_adopted'
            assert closure['accepted_count']==12 and closure['fresh_pairs_retained']==3
            assert (c.ROOT/'research_runs/tokenizer-ffn-quality-v1-001/STOP').is_file()
            allow=[]
            for slot in (0,1):
                old=read(reference(E/f'quality001-protected-pilot-{slot:02d}.json'))
                assert Path(old['root'])==P/f'{slot:02d}'
                for rel,binding in old['inventory'].items():
                    path=Path(old['root'])/rel;now=record(path)
                    assert now['size']==binding['size'] and now['sha256']==binding['sha256']
                    if path.suffix=='.pt':allow.append(now)
            assert len(allow)==4 and sum(x['size'] for x in allow)==1118322228
            protected=[];metadata=[];remove={x['path'] for x in allow}
            for path in paths():
                if str(path) in remove:continue
                row=record(path)
                (protected if path.suffix=='.pt' or path.is_relative_to(c.ROOT/'research_archives') else metadata).append(row)
            inventories={}
            for name,rows in [('protected',protected),('preserved_metadata',metadata)]:
                inventories[name]=[]
                for offset in range(0,len(rows),500):
                    part=B/f'{name}-{offset//500:03d}.json';e.exclusive(part,rows[offset:offset+500]);inventories[name].append(reference(part))
            manifest=dict(status='audited_exact_successful_pilot_allowlist',delete_files=allow,
                inventory_shards=inventories,protected_count=len(protected),metadata_count=len(metadata),free_bytes_before=shutil.disk_usage(c.ROOT).free,
                policy=reference(c.ROOT/'research/TOKENIZER_CHECKPOINT_RETENTION.md'),source=reference(__file__),
                eligibility=[reference(E/n) for n in ('pilot001-review.json','quality001-closure.json','quality001-final-review.json','quality001-protected-pilot-00.json','quality001-protected-pilot-01.json')],
                reason='Two successful FFN resource-pilot pairs, dependent twelve-endpoint quality and all inference now closed. Current user-authorized retention policy supersedes the completed campaign snapshot retain-pilots condition only after closure. Selected/fresh/failed and numerical fixtures excluded.',
                limitation='These four pilot weights cannot be reloaded after deletion. Original manifests and historical closure remain immutable and describe their review time; this receipt explains later absence. Never rerun closed controllers.')
            e.exclusive(B/'manifest.json',manifest)
            print(json.dumps(dict(status='audited',delete_bytes=1118322228,protected=len(protected),metadata=len(metadata))),flush=True)
        else:
            manifest=read(reference(B/'manifest.json'))
            for group,refs in manifest['inventory_shards'].items():
                manifest[group]=[row for ref in refs for row in read(ref)]
            assert reference(__file__)==manifest['source'] and reference(c.ROOT/'research/TOKENIZER_CHECKPOINT_RETENTION.md')==manifest['policy']
            assert read(reference(B/'audit-guard.json'))['status']=='finished'
            for ref in manifest['eligibility']:read(ref)
            assert len(manifest['delete_files'])==4 and sum(x['size'] for x in manifest['delete_files'])==1118322228
            original={r['path']:r for group in ('delete_files','protected','preserved_metadata') for r in manifest[group]}
            assert {str(p) for p in paths()}==set(original)
            # Verify full identities before mutation; the preceding audit hashed all files.
            for name,row in original.items():
                p=Path(name);s=p.stat();assert not p.is_symlink()
                assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['size'],row['mtime_ns'])
            deleted=[]
            with (B/'journal.jsonl').open('x') as log:
                for row in manifest['delete_files']:
                    p=Path(row['path']);assert p.parent in {P/'00/checkpoints',P/'01/checkpoints'} and record(p)==row
                    log.write(json.dumps(dict(event='intent',file=row))+'\n');log.flush();os.fsync(log.fileno())
                    p.unlink();deleted.append(row['path'])
                    log.write(json.dumps(dict(event='removed',path=str(p)))+'\n');log.flush();os.fsync(log.fileno())
            survivors=[*manifest['protected'],*manifest['preserved_metadata']]
            assert {str(p) for p in paths()}=={r['path'] for r in survivors}
            for row in survivors:assert record(Path(row['path']))==row
            result=dict(status='completed',manifest=reference(B/'manifest.json'),journal=reference(B/'journal.jsonl'),
                deleted_files=deleted,deleted_bytes=1118322228,protected_verified=len(manifest['protected']),
                metadata_verified=len(manifest['preserved_metadata']),free_bytes_after=shutil.disk_usage(c.ROOT).free,
                failed_and_all_selected_fresh_weights_unchanged=True,closed_campaigns_reopened=False)
    if args.apply:
        result['guard']=reference(B/'apply-guard.json');e.exclusive(B/'completed.json',result);print(json.dumps(result),flush=True)
