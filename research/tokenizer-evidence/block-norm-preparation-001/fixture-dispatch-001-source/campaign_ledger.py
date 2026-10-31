"""Append-only preparation ledger. No worker, deletion or launch authorization.

Parent/retention receipts are explicitly synthetic in this version. A caller must
retain the returned head outside this directory: a hash chain alone cannot detect
removal of a valid suffix. Future real-artifact acceptance needs separate tests.
"""
import contextlib
import fcntl
import json
import os
import re
from pathlib import Path
from . import campaign_contract as c, campaign_rules as r

SCOPE='synthetic_ledger_only'
MAX_EVENTS=64
MAX_OBJECTS=128
require=r.require


def keys(value,expected):
    require(type(value) is dict and set(value)==set(expected),'Exact ledger schema required')


def digest(value):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None,'SHA256 reference required')
    return value


def directory(path):
    path=Path(path)
    require(path.is_absolute() and '..' not in path.parts,'Absolute ledger directory required')
    for p in [path,*path.parents]:require(not p.is_symlink(),'Ledger symlink forbidden')
    require(path.is_dir(),'Ledger directory missing')
    return path


def sync_dir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)


def write_new(path,value):
    raw=c.canonical(value).encode();require(len(raw)<=2*1024**2,'Bounded ledger JSON required')
    with Path(path).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    sync_dir(Path(path).parent)


def read_json(path):
    before=c.file_sha(path,max_bytes=2*1024**2);raw=Path(path).read_text()
    value=json.loads(raw)
    require(c.canonical(value)==raw and c.file_sha(path,max_bytes=2*1024**2)==before,'Noncanonical or changing ledger JSON')
    return value


@contextlib.contextmanager
def locked(root):
    root=directory(root);lock=root/'lock'
    c.file_sha(lock,root=root,max_bytes=0)
    fd=os.open(lock,os.O_RDWR|os.O_NOFOLLOW)
    try:
        st=os.fstat(fd);require(st.st_nlink==1 and st.st_ino==lock.stat().st_ino,'Changed lock identity')
        try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise ValueError('Ledger already locked') from exc
        yield
    finally:os.close(fd)


def put(root,value):
    objects=directory(Path(root)/'objects');ref=c.fingerprint(value);path=objects/(ref+'.json')
    if path.exists():require(c.canonical(read_json(path))==c.canonical(value),'Object collision or mutation')
    else:
        require(len(list(objects.iterdir()))<MAX_OBJECTS,'Ledger object bound exceeded');write_new(path,value)
    return ref


def get(root,ref):
    path=directory(Path(root)/'objects')/(digest(ref)+'.json')
    value=read_json(path);require(c.fingerprint(value)==ref,'Object hash differs')
    return value


def action(state):
    if state['terminal'] is not None:return dict(kind='terminal',reason=state['terminal'])
    if state['pending'] is not None:return dict(kind='pending',request=state['pending']['event'])
    return r.next_action(state['manifest']['seeds'],state['pilots'],state['quality'],state['selection'],
                         pilots_released=state['pilot_boundary'],losers_released=state['screen_boundary'])


def request(state):
    a=action(state);require(a['kind'] in ('pilot','quality'),'No trial request allowed at this boundary')
    stage='pilots' if a['kind']=='pilot' else 'quality';slot=len(state[stage])
    output=Path(state['campaign'])/'evidence'/stage/f'{slot:02d}'
    cfg=c.trial_config(output,a['spec'],state['manifest']['seeds'])
    return dict(scope=SCOPE,manifest=state['manifest_ref'],stage=stage,slot=slot,spec=a['spec'],output=str(output),config_sha256=c.fingerprint(cfg)),cfg


def receipt(state,ce=5.):
    """Synthetic test receipt only; does not inspect checkpoints or scores."""
    p=state['pending'];require(p is not None,'Pending request required')
    row=dict(p['request']['spec'],status='completed')
    if p['request']['stage']=='quality':row['h200_ce']=ce
    return dict(scope=SCOPE,kind='synthetic_parent_review',request_event=p['event'],request_sha256=c.fingerprint(p['request']),
                config_sha256=p['request']['config_sha256'],record=row)


def boundary(state):
    a=action(state);require(a['kind'] in ('review_pilot_boundary','review_screen_boundary'),'No retention boundary requested')
    screen=a['kind']=='review_screen_boundary'
    retention=r.screening_retention(state['manifest']['seeds'],state['quality'],state['selection']) if screen else dict(pilot_indices=[0,1],deletion_authorized=False)
    return dict(scope=SCOPE,kind=a['kind'],accepted_events=list(state['accepted_events']),selection=state['selection'],retention=retention,deletion_performed=False)


def transition(root,state,event,event_ref):
    kind=event['kind'];payload=event['payload']
    if kind=='genesis':
        require(state is None and event['seq']==0,'One genesis required')
        keys(payload,('scope','manifest','campaign'));require(payload['scope']==SCOPE,'Only synthetic preparation supported')
        manifest=get(root,payload['manifest']);require(manifest['launch_permitted'] is False,'Preparation only; launch prohibited')
        r.validate_seeds(manifest['seeds'])
        campaign=Path(payload['campaign']);c._output(campaign/'evidence/pilots/00')
        return dict(manifest=manifest,manifest_ref=payload['manifest'],campaign=str(campaign),pilots=[],quality=[],selection=None,
                    pilot_boundary=False,screen_boundary=False,pending=None,terminal=None,accepted_events=[],launch_permitted=False)
    require(state is not None and state['terminal'] is None,'Terminal ledger cannot advance')
    if kind in ('stop','failure'):
        keys(payload,('reason','pending_request'));require(type(payload['reason']) is str and 0<len(payload['reason'])<=1024,'Bounded reason required')
        require(payload['pending_request']==(state['pending']['event'] if state['pending'] else None),'Terminal request binding differs')
        if kind=='failure':require(state['pending'] is not None,'Failure requires a pending request')
        state['terminal']=dict(kind=kind,**payload);return state
    if kind=='request':
        expected,cfg=request(state);keys(payload,('request','config'))
        require(c.canonical(get(root,payload['request']))==c.canonical(expected),'Request differs from next finite slot')
        require(payload['config']==expected['config_sha256'] and c.canonical(get(root,payload['config']))==c.canonical(cfg),'Exact request/config binding required')
        state['pending']=dict(event=event_ref,request=expected)
    elif kind=='accept':
        keys(payload,('receipt',));got=get(root,payload['receipt'])
        require(state['pending'] is not None,'Acceptance requires pending request')
        require(type(got) is dict and type(got.get('record')) is dict,'Parent record required')
        expected=receipt(state,got['record'].get('h200_ce',5.))
        require(c.canonical(got)==c.canonical(expected),'Parent receipt differs from pending request')
        p=state['pending']['request'];row=got['record'];r.checked_prefix([row],[p['spec']],quality=p['stage']=='quality',complete=True)
        state[p['stage']].append(row);state['accepted_events'].append(event_ref);state['pending']=None
    elif kind=='selection':
        keys(payload,('value',));a=action(state)
        require(a['kind']=='save_selection' and c.canonical(payload['value'])==c.canonical(a['value']),'Selection differs or is premature')
        state['selection']=payload['value']
    elif kind=='boundary':
        keys(payload,('receipt',));expected=boundary(state)
        require(c.canonical(get(root,payload['receipt']))==c.canonical(expected),'Audited boundary receipt differs')
        state['screen_boundary' if expected['kind']=='review_screen_boundary' else 'pilot_boundary']=True
    else:raise ValueError('Unknown ledger event')
    return state


def read(root,expected_head):
    root=directory(root);digest(expected_head)
    require({p.name for p in root.iterdir()}=={'lock','objects','events'},'Unexpected ledger entries')
    c.file_sha(root/'lock',root=root,max_bytes=0)
    objs=directory(root/'objects');obj_files=list(objs.iterdir());require(len(obj_files)<=MAX_OBJECTS,'Object count exceeds bound')
    for p in obj_files:
        require(re.fullmatch(r'[0-9a-f]{64}\.json',p.name) is not None,'Unexpected object filename')
        get(root,p.stem)  # Includes unreferenced crash-orphan objects; never delete them.
    files=sorted(directory(root/'events').iterdir());require(1<=len(files)<=MAX_EVENTS,'Bounded nonempty event chain required')
    state=None;prev=None
    for i,path in enumerate(files):
        require(path.name==f'{i:04d}.json','Missing, reordered or extra ledger event')
        value=read_json(path);keys(value,('seq','prev','kind','payload'))
        require(type(value['seq']) is int and value['seq']==i and value['prev']==prev,'Broken event sequence/hash chain')
        ref=c.fingerprint(value);state=transition(root,state,value,ref);prev=ref
    require(prev==expected_head,'Stale head, changed history or truncated suffix')
    return dict(state,head=prev,event_count=len(files))


def create(root,manifest,campaign):
    """Create a new synthetic-only ledger; no overwrite, launch or readiness."""
    root=Path(root);directory(root.parent);require(not root.exists() and not root.is_symlink(),'Ledger already exists')
    c.verify_preparation_manifest(manifest);c._output(Path(campaign)/'evidence/pilots/00')
    root.mkdir();(root/'objects').mkdir();(root/'events').mkdir()
    with (root/'lock').open('xb') as f:f.flush();os.fsync(f.fileno())
    sync_dir(root)
    with locked(root):
        ref=put(root,manifest);e=dict(seq=0,prev=None,kind='genesis',payload=dict(scope=SCOPE,manifest=ref,campaign=str(campaign)))
        head=c.fingerprint(e);transition(root,None,e,head);write_new(root/'events/0000.json',e)
    sync_dir(root.parent)
    return head


def append(root,expected_head,kind,payload,objects=()):
    """Compare-and-append under nonblocking lock; caller persists returned head.

    Payload object references are checked before event write. Valid orphan objects
    can remain after an invalid operation/crash; they cannot advance the state.
    """
    root=Path(root)
    with locked(root):
        state=read(root,expected_head);require(state['event_count']<MAX_EVENTS,'Event bound reached')
        if kind not in ('stop','failure'):
            c.verify_sources(state['manifest']['sources'])
            require(c.canonical(c.environment())==c.canonical(state['manifest']['environment']),'Environment drift')
            data=state['manifest']['data'];require(c.file_sha(Path(data['path']),root=c.ROOT,max_bytes=4*1024**2)==data['corpus_sha256'],'Corpus drift')
        require(state['terminal'] is None,'Terminal ledger cannot advance')
        for obj in objects:put(root,obj)
        e=dict(seq=state['event_count'],prev=expected_head,kind=kind,payload=payload);head=c.fingerprint(e)
        transition(root,state,e,head)
        write_new(root/'events'/f'{e["seq"]:04d}.json',e)
        return head
