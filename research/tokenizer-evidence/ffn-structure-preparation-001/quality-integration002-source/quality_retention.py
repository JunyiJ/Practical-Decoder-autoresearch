"""Exact losing-screen deletion under a caller's verified stage audit and locks."""
import json,os,shutil
from pathlib import Path
from . import campaign_contract as c,campaign_rules as rules
from research.block_norm_v1.fixture_dispatch import reference,read
from .support import exclusive
from src.research.tokenizer_wide_hybrid_cleanup import _locks,_sync_directory
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
require=c.require


def inventory(root):
    out={}
    for path in sorted(root.rglob('*')):
        require(not path.is_symlink(),'No symlinks in retention inventory')
        if path.is_file():out[str(path)]=dict(size=path.stat().st_size,sha256=c.file_sha(path,max_bytes=512*1024**2))
    require(0<len(out)<2048,'Bounded complete evidence inventory required')
    return out


def manifest(root,seeds,records,selection,requests):
    decision=rules.screening_retention(seeds,records,selection);require(len(requests)==6,'Complete six-screen requests required')
    before=inventory(root/'evidence');delete={};retain={}
    for slot,req in enumerate(requests):
        expected=root/f'evidence/quality/{slot:02d}'
        c.validate_trial_config(req['config'],expected,rules.screen_specs(seeds)[slot],seeds)
        for role in ('parent','h200'):
            ref=req['files'][role];path=Path(ref['path'])
            require(path.parent==expected/'checkpoints' and path.suffix=='.pt','Exact screen checkpoint path required')
            require(before[str(path)]['sha256']==ref['sha256'],'Audit checkpoint hash differs')
            target=delete if slot in decision['losing_indices'] else retain
            require(str(path) not in delete and str(path) not in retain,'Aliased checkpoint')
            target[str(path)]=before[str(path)]
    require(len(delete)==8 and len(retain)==4,'Four losing and two selected pairs required')
    return dict(kind='ffn_screen_retention_v1',selection=selection,records=records,inventory=before,delete=delete,retain=retain)


def check_entry(path,row):
    p=Path(path);require(p.stat().st_size==row['size'] and c.file_sha(p,max_bytes=512*1024**2)==row['sha256'],'Retained file changed')


def verify(ref,expected_context):
    receipt=read(ref);require(receipt['status']=='completed' and receipt['context']==expected_context,'Completed exact cleanup context required')
    plan=read(receipt['manifest']);guard=read(receipt['guard']);require(guard['status']=='finished' and guard['reason'] is None,'Cleanup guard incomplete')
    require(receipt['deleted']==plan['delete'] and receipt['protected']=={k:v for k,v in plan['inventory'].items() if k not in plan['delete']},'Incomplete retention receipt')
    for name in plan['delete']:require(not Path(name).exists(),'Deleted losing weight reappeared')
    for name,row in receipt['protected'].items():check_entry(name,row)
    read(receipt['journal'])
    return plan['delete']


def execute(writer,destination,seeds,records,selection,requests,context,revalidate):
    # This function has no CLI: the quality controller independently re-audits all
    # six real checkpoints before calling it. Disposables exercise file handling.
    _locks(writer.root/'evidence',writer.lock_fds);require(not destination.exists(),'Retention attempts cannot retry')
    plan=manifest(writer.root,seeds,records,selection,requests);require(revalidate()==context,'Cleanup context differs')
    destination.mkdir();_sync_directory(destination.parent);exclusive(destination/'manifest.json',plan)
    done=[];error=None
    try:
        with CpuPrerequisiteGuard(destination/'guard.json',writer.root,timeout_seconds=600,poll_seconds=.25):
            with (destination/'journal.json').open('x') as stream:
                # Persist intent/deletion records as a bounded JSON list, updated
                # with fsync after each action; the final hash is bound below.
                def event(value):
                    done.append(value);raw=c.canonical(done)+'\n';require(len(raw)<1024**2,'Bounded deletion journal required')
                    stream.seek(0);stream.write(raw);stream.truncate();stream.flush();os.fsync(stream.fileno())
                for name,row in plan['delete'].items():
                    _locks(writer.root/'evidence',writer.lock_fds);require(revalidate()==context,'Cleanup context changed')
                    require(shutil.disk_usage(writer.root).free>=8*1024**3,'Disk reserve crossed')
                    require(read(reference(destination/'manifest.json'))==plan,'Cleanup manifest changed')
                    check_entry(name,row);path=Path(name);before=path.stat();event(dict(event='intent',path=name,**row))
                    after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns) and not path.is_symlink(),'Deletion target changed')
                    path.unlink();_sync_directory(path.parent);event(dict(event='deleted',path=name,**row))
                require(inventory(writer.root/'evidence')=={k:v for k,v in plan['inventory'].items() if k not in plan['delete']},'Protected evidence changed')
                event(dict(event='protected_verified',files=len(plan['inventory'])-len(plan['delete'])))
        require(revalidate()==context,'Final cleanup context changed')
        receipt=dict(status='completed',context=context,manifest=reference(destination/'manifest.json'),guard=reference(destination/'guard.json'),journal=reference(destination/'journal.json'),deleted=plan['delete'],protected={k:v for k,v in plan['inventory'].items() if k not in plan['delete']},deleted_logical_bytes=sum(v['size'] for v in plan['delete'].values()),free_bytes=shutil.disk_usage(writer.root).free)
        exclusive(destination/'receipt.json',receipt);verify(reference(destination/'receipt.json'),context)
        return reference(destination/'receipt.json')
    except BaseException as exc:
        exclusive(destination/'failure.json',dict(status='failed',error=repr(exc),no_retry=True));raise
