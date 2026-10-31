"""Synthetic metadata-only ledger tests; never dispatch, score or delete weights."""
import copy
import json
from pathlib import Path
import pytest
from research.block_norm_v1 import campaign_contract as c, campaign_ledger as l, campaign_rules as r
from research.block_norm_v1.test_campaign_rules import SEEDS

CAMPAIGN=c.ROOT/'research_runs/tokenizer-block-norm-v2-ledger-fixture'


@pytest.fixture(scope='module')
def manifest():return c.preparation_manifest(SEEDS)


@pytest.fixture
def ledger(tmp_path,manifest):
    root=tmp_path/'ledger';head=l.create(root,manifest,CAMPAIGN)
    return root,head


def issue(root,head):
    state=l.read(root,head);req,cfg=l.request(state)
    return l.append(root,head,'request',dict(request=c.fingerprint(req),config=c.fingerprint(cfg)),[req,cfg])


def accept(root,head,ce=5.):
    rec=l.receipt(l.read(root,head),ce)
    return l.append(root,head,'accept',dict(receipt=c.fingerprint(rec)),[rec])


def release(root,head):
    rec=l.boundary(l.read(root,head))
    return l.append(root,head,'boundary',dict(receipt=c.fingerprint(rec)),[rec])


def terminal(root,head,kind='stop'):
    s=l.read(root,head)
    return l.append(root,head,kind,dict(reason='fixture terminal',pending_request=s['pending']['event'] if s['pending'] else None))


def test_complete_finite_replay_and_content_binding(ledger,record_property):
    root,head=ledger;history=[head];kinds=[]
    for _ in range(40):
        s=l.read(root,head);a=l.action(s);kinds.append(a['kind'])
        if a['kind'] in ('pilot','quality'):
            if a['kind']=='quality' and a['spec']['stage']=='confirm':
                req,cfg=l.request(s);req['spec']['lr']=.0025
                cfg=c.trial_config(req['output'],req['spec'],SEEDS);req['config_sha256']=c.fingerprint(cfg)
                with pytest.raises(ValueError):l.append(root,head,'request',dict(request=c.fingerprint(req),config=c.fingerprint(cfg)),[req,cfg])
            head=issue(root,head)
        elif a['kind']=='pending':head=accept(root,head,5.+len(s['quality'])/10)
        elif a['kind'] in ('review_pilot_boundary','review_screen_boundary'):
            if a['kind']=='review_screen_boundary':
                bad=l.boundary(s);bad['retention']['selected_indices']=[2,3]
                with pytest.raises(ValueError):l.append(root,head,'boundary',dict(receipt=c.fingerprint(bad)),[bad])
            head=release(root,head)
        elif a['kind']=='save_selection':
            with pytest.raises(ValueError):l.append(root,head,'selection',dict(value=dict.fromkeys(r.ARMS,.0025)))
            head=l.append(root,head,'selection',dict(value=a['value']))
        else:assert a==dict(kind='review_final',adopted=False);break
        history.append(head)
    s=l.read(root,head)
    assert len(history)==32 and len(s['pilots'])==2 and len(s['quality'])==12
    assert s['selection']==dict.fromkeys(r.ARMS,.000625)
    assert [row['seed'] for row in s['quality'][6:]]==[x for x in SEEDS['fresh'] for _ in range(2)]
    assert s['pending'] is None and s['launch_permitted'] is False
    assert s['pilot_boundary'] and s['screen_boundary'] and len(s['accepted_events'])==14
    assert len(set(history))==32 and not CAMPAIGN.exists()
    for h in history[:-1]:
        with pytest.raises(ValueError):l.read(root,h)
    # Final state cannot create another request, and never adopts a candidate.
    with pytest.raises(ValueError):issue(root,head)
    record_property('complete_replay',dict(events=len(history),accepted=14,objects=len(list((root/'objects').iterdir())),head=head,actions=kinds,launch_permitted=False))


def test_one_pending_stale_head_and_duplicate_accept_rejected(ledger):
    root,genesis=ledger;head=issue(root,genesis);s=l.read(root,head)
    assert l.action(s)['kind']=='pending'
    with pytest.raises(ValueError):issue(root,head)
    with pytest.raises(ValueError):l.append(root,genesis,'stop',dict(reason='stale',pending_request=None))
    old=l.receipt(s);head=accept(root,head)
    with pytest.raises(ValueError):l.append(root,head,'accept',dict(receipt=c.fingerprint(old)),[old])
    assert len(l.read(root,head)['pilots'])==1


@pytest.mark.parametrize('case',['request_config','request_seed','request_slot','request_extra','config_lr','config_path','config_policy'])
def test_request_mismatch_cannot_append(ledger,case):
    root,head=ledger;req,cfg=l.request(l.read(root,head))
    if case=='request_config':req['config_sha256']='0'*64
    elif case=='request_seed':req['spec']['seed']+=1
    elif case=='request_slot':req['slot']=1
    elif case=='request_extra':req['launch_permitted']=True
    elif case=='config_lr':cfg['training']['learning_rate']=.01
    elif case=='config_path':cfg['training']['checkpoint_dir']='/private/tmp/weights'
    else:cfg['model']['tokenizer_architecture_policy']='tokenizer_mla_block_norm_v1'
    with pytest.raises(ValueError):l.append(root,head,'request',dict(request=c.fingerprint(req),config=c.fingerprint(cfg)),[req,cfg])
    assert l.read(root,head)['pending'] is None and len(list((root/'events').iterdir()))==1


@pytest.mark.parametrize('case',['event','request','config','failed','wrong_arm','wrong_seed','extra','real_scope'])
def test_wrong_parent_receipt_rejected(ledger,case):
    root,head=ledger;head=issue(root,head);rec=l.receipt(l.read(root,head))
    if case in ('event','request','config'):rec[{'event':'request_event','request':'request_sha256','config':'config_sha256'}[case]]='0'*64
    elif case=='failed':rec['record']['status']='failed'
    elif case=='wrong_arm':rec['record']['arm']=r.ARMS[1]
    elif case=='wrong_seed':rec['record']['seed']+=1
    elif case=='extra':rec['checkpoints_validated']=True
    else:rec['scope']='real_worker'
    with pytest.raises(ValueError):l.append(root,head,'accept',dict(receipt=c.fingerprint(rec)),[rec])
    assert l.read(root,head)['pending'] is not None


@pytest.mark.parametrize('kind',['stop','failure'])
def test_terminal_is_durable_and_keeps_pending_request(ledger,kind):
    root,head=ledger;head=issue(root,head);old=head;head=terminal(root,head,kind)
    state=l.read(root,head);assert state['pending']['event']==old
    assert l.action(state)['kind']=='terminal' and state['terminal']['kind']==kind
    with pytest.raises(ValueError):accept(root,head)
    with pytest.raises(ValueError):terminal(root,head)
    assert l.read(root,head)==state


def test_stop_before_dispatch_and_failure_requires_pending(ledger):
    root,head=ledger
    with pytest.raises(ValueError):terminal(root,head,'failure')
    head=terminal(root,head)
    with pytest.raises(ValueError):issue(root,head)


def test_nonblocking_lock_serializes_writers(ledger):
    root,head=ledger
    with l.locked(root):
        with pytest.raises(ValueError,match='locked'):terminal(root,head)
    head=terminal(root,head);assert l.read(root,head)['terminal']['kind']=='stop'


@pytest.mark.parametrize('case',['accepted_events','kind','selection','deletion','indices','scope'])
def test_boundary_receipt_must_match_accepted_prefix(ledger,case):
    root,head=ledger
    for _ in range(2):head=accept(root,issue(root,head))
    rec=l.boundary(l.read(root,head))
    if case=='accepted_events':rec['accepted_events'].reverse()
    elif case=='kind':rec['kind']='review_screen_boundary'
    elif case=='selection':rec['selection']=dict.fromkeys(r.ARMS,.0025)
    elif case=='deletion':rec['deletion_performed']=True
    elif case=='indices':rec['retention']['pilot_indices']=[0]
    else:rec['scope']='cleanup_authorization'
    with pytest.raises(ValueError):l.append(root,head,'boundary',dict(receipt=c.fingerprint(rec)),[rec])
    with pytest.raises(ValueError):issue(root,head)
    assert l.action(l.read(root,head))['kind']=='review_pilot_boundary'


def test_premature_selection_and_unverified_boundary_rejected(ledger):
    root,head=ledger
    with pytest.raises(ValueError):l.append(root,head,'selection',dict(value=dict.fromkeys(r.ARMS,.000625)))
    with pytest.raises(FileNotFoundError):l.append(root,head,'accept',dict(receipt='0'*64))
    with pytest.raises(ValueError):l.append(root,head,'boundary',dict(receipt='0'*64))


@pytest.mark.parametrize('case',['missing_middle','truncated_suffix','event_edit','object_edit','symlink_event','hardlink_object','extra_file','extra_event','object_missing','noncanonical','event_bool'])
def test_disk_corruption_or_alias_fails_closed(ledger,case,tmp_path):
    root,head=ledger;head=accept(root,issue(root,head));event=root/'events/0001.json'
    obj=next((root/'objects').iterdir())
    if case=='missing_middle':event.unlink()
    elif case=='truncated_suffix':(root/'events/0002.json').unlink()
    elif case=='event_edit':event.write_text('{}')
    elif case=='object_edit':obj.write_text('{}')
    elif case=='symlink_event':saved=tmp_path/'saved';event.rename(saved);event.symlink_to(saved)
    elif case=='hardlink_object':
        import os
        os.link(obj,tmp_path/'alias')
    elif case=='extra_file':(root/'ignored').write_text('')
    elif case=='extra_event':(root/'events/0004.json').write_text('{}')
    elif case=='object_missing':obj.unlink()
    elif case=='noncanonical':event.write_text(event.read_text()+'\n')
    else:
        v=json.loads(event.read_text());v['seq']=True;event.write_text(c.canonical(v))
    with pytest.raises((ValueError,FileNotFoundError)):l.read(root,head)


def test_orphan_objects_preserved_but_do_not_advance(ledger):
    root,head=ledger;ref=l.put(root,dict(scope=l.SCOPE,orphan=True));before=(root/'objects'/(ref+'.json')).read_bytes()
    assert l.read(root,head)['pending'] is None
    head=issue(root,head)
    assert (root/'objects'/(ref+'.json')).read_bytes()==before
    assert l.read(root,head)['pending'] is not None


def test_source_drift_blocks_progress_but_stop_is_allowed(ledger,monkeypatch):
    root,head=ledger
    def fail(*args):raise ValueError('source drift fixture')
    monkeypatch.setattr(c,'verify_sources',fail)
    with pytest.raises(ValueError,match='source drift'):issue(root,head)
    head=terminal(root,head);assert l.read(root,head)['terminal']['kind']=='stop'


def test_existing_root_and_invalid_manifest_cannot_initialize(ledger,manifest,tmp_path):
    root,head=ledger
    with pytest.raises(ValueError):l.create(root,manifest,CAMPAIGN)
    changed=copy.deepcopy(manifest);changed['launch_permitted']=True
    with pytest.raises(ValueError):l.create(tmp_path/'other',changed,CAMPAIGN)
    assert not (tmp_path/'other').exists() and l.read(root,head)['event_count']==1
