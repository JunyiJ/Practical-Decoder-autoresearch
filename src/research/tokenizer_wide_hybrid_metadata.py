"""Prospective v2 campaign layout and complete metadata budget, no launcher.

Existing v1 layouts/producers are unchanged. A future v2 controller must use
this layout, enforce writes and monitor active children; idle scans alone cannot
bound a producer's transient output. No checkpoint bytes are cleanup credit.
"""
from pathlib import Path
import stat

from .tokenizer_wide_hybrid_retention import METADATA, VERSION
from .tokenizer_wide_hybrid_retention_receipts import _path, _relative

MiB = 1024**2
MAX_FILES = 2048


def budgets():
    """All training, serving, cleanup, controller and atomic-write allowances."""
    limits = {'evidence/meta':2*MiB, 'control':3*MiB, 'emergency':MiB//2}
    for stage, count in (('pilots',3),('quality',24)):
        for index in range(count):
            base=f'evidence/{stage}/{index:02d}'
            limits.update({base+'/log':4*MiB,base+'/events':2*MiB,
                           base+'/resources':2*MiB,base+'/other':MiB//4})
    for seed in (3511,3512,3513):
        base=f'evidence/inference/{seed}'
        limits.update({base+'/log':4*MiB,base+'/other':MiB//4})
    for boundary in ('pilot','lr','rank'):
        limits['evidence/.retention/'+boundary]=5*MiB
    if sum(limits.values()) != METADATA: raise ValueError('Complete metadata allowance does not reconcile')
    return limits


def classify(name):
    """Return a bounded metadata group or an explicit trajectory weight group."""
    parts=Path(_relative(name)).parts
    if len(parts)==1 and parts[0] in ('.lock','STOP'):return 'control'
    if len(parts)>=2 and parts[0] in ('control','emergency'):return parts[0]
    if len(parts)>=3 and parts[:2]==('evidence','meta'):return 'evidence/meta'
    if len(parts)>=4 and parts[:2]==('evidence','.retention') and parts[2] in ('pilot','lr','rank'):
        return '/'.join(parts[:3])
    if len(parts)<4 or parts[0]!='evidence':raise ValueError('Undeclared v2 campaign output path')
    stage,slot=parts[1:3];base='/'.join(parts[:3]);tail='/'.join(parts[3:])
    if stage in ('pilots','quality'):
        if slot not in {f'{i:02d}' for i in range(3 if stage=='pilots' else 24)}:
            raise ValueError('Undeclared trajectory slot')
        if parts[3]=='checkpoints':
            if len(parts)!=5 or not parts[4].removesuffix('.tmp').endswith('.pt'):
                raise ValueError('Unrecognized checkpoint output')
            return base+'/weights'
        if '.pt' in parts[-1]:raise ValueError('Checkpoint outside declared weight directory')
        canonical=tail.removesuffix('.tmp')
        return base+('/log' if canonical=='process/worker.log' else
                     '/events' if canonical=='resource-events.jsonl' else
                     '/resources' if canonical=='resource-audit.json' else '/other')
    if stage=='inference' and slot in ('3511','3512','3513'):
        if '.pt' in parts[-1] or 'checkpoints' in parts[3:]:
            raise ValueError('Inference must not write model checkpoints')
        return base+('/log' if tail.removesuffix('.tmp')=='process/worker.log' else '/other')
    raise ValueError('Undeclared v2 campaign output path')


def snapshot(campaign):
    """Read-only census, counting distinct atomic temporary files at full size.

    The exclusive JSON publisher briefly hard-links a .tmp to its final path;
    that one inode is counted once, only for that exact same-directory pair.
    Other aliases, external hard links, symlinks and nonregular files fail closed.
    Hash/semantic provenance is separately enforced by the audit/receipt gates.
    """
    if Path(campaign).is_symlink() or not Path(campaign).is_dir():
        raise ValueError('Existing nonsymlink campaign directory required')
    root=Path(campaign).resolve();limits=budgets();groups=dict.fromkeys(limits,0)
    files={};inodes={};weight_bytes=0
    for path in root.rglob('*'):
        if path.is_symlink():raise ValueError('Metadata inventory symlink refused')
        st=path.stat()
        if stat.S_ISDIR(st.st_mode):continue
        if not stat.S_ISREG(st.st_mode):raise ValueError('Nonregular campaign output')
        name=str(path.relative_to(root));group=classify(name)
        files[name]=dict(size=st.st_size,group=group)
        if len(files)>MAX_FILES:raise ValueError('Campaign file-count cap exceeded')
        inodes.setdefault((st.st_dev,st.st_ino),[]).append((name,group,st.st_size,st.st_nlink))
    for entries in inodes.values():
        first=entries[0];names=sorted(e[0] for e in entries)
        if any(e[3]!=len(entries) for e in entries):raise ValueError('External hard link refused')
        if len(entries)>1 and (len(entries)!=2 or names[1]!=names[0]+'.tmp'
                or len({e[1] for e in entries})!=1 or first[1].endswith('/weights')):
            raise ValueError('Unexpected output inode alias')
        if first[1].endswith('/weights'):weight_bytes+=first[2]
        else:groups[first[1]]+=first[2]
    for group,used in groups.items():
        if used>limits[group]:raise ValueError('Metadata group allowance exceeded: '+group)
    return dict(version=VERSION,groups=groups,metadata_bytes=sum(groups.values()),
                remaining_metadata_bytes=METADATA-sum(groups.values()),
                checkpoint_logical_bytes=weight_bytes,files=files,
                transient_producer_enforcement_verified=False,dispatch_authorized=False)


def require_write_capacity(campaign, name, new_bytes):
    """Check space for a new temporary payload without discounting old contents.

    Caller must hold locks and use an enforced bounded writer, then recheck.
    A producer can still exceed a cap unless the future runner monitors it.
    """
    if type(new_bytes) is not int or new_bytes<0:raise ValueError('Exact nonnegative payload bytes required')
    observed=snapshot(campaign)
    root=Path(campaign).resolve();_path(root,name);group=classify(name)
    if group.endswith('/weights'):raise ValueError('Checkpoint space needs the full trajectory gate')
    if observed['groups'][group]+new_bytes>budgets()[group]:
        raise ValueError('Metadata atomic-write overlap exceeds group allowance')
    return dict(group=group,additional_bytes=new_bytes,dispatch_authorized=False)
