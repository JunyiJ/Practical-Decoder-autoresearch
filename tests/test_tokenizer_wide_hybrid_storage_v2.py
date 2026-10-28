"""Synthetic trajectories, tiny cleanup receipts, sparse files; no real weights."""
import copy
import fcntl
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.research import tokenizer_wide_hybrid_storage_v2 as storage
from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research import tokenizer_wide_hybrid_rules as rules
from src.research.tokenizer_wide_hybrid_retention import boundary_candidates
from src.research.tokenizer_wide_hybrid_retention_receipts import planned_manifest
from src.research.protocol import digest,file_digest


@pytest.fixture
def projection():
    return json.loads((Path(__file__).resolve().parents[1]/'research/tokenizer-evidence/wide-hybrid-preparation-001/rank-storage-projection.json').read_text())


def rows(specs,ce=5.):return [dict(s,status='completed',ce=ce) for s in specs]


def inventory(root):
    return {str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in root.rglob('*') if p.is_file()}


@pytest.fixture
def fixture(tmp_path,monkeypatch,projection):
    root=tmp_path/'campaign';evidence=root/'evidence';(evidence/'meta').mkdir(parents=True)
    (evidence/'meta/fixture.json').write_text('{"synthetic":true}')
    global_lock=tmp_path/'global.lock';global_lock.touch();(root/'.lock').touch()
    monkeypatch.setattr(cleanup,'GLOBAL_LOCK',global_lock)
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=64*1024**3))
    with global_lock.open('rb') as g,(root/'.lock').open('rb') as c:
        fds=(g.fileno(),c.fileno())
        for fd in fds:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        args=dict(campaign=root,projection=projection,pilots=[],records=[],rates=None,selection=None,
            stages=[],current_inventory=inventory(evidence),checkpoint_pairs={},lock_fds=fds)
        def add(kind,spec):
            records=args['pilots' if kind=='pilot' else 'records'];index=len(records);records.append(dict(spec,status='completed',ce=5.))
            stem=f"{'pilots' if kind=='pilot' else 'quality'}/{index:02d}";pair={}
            for role in ('raw','h200'):
                path=evidence/stem/'checkpoints'/f'{role}.pt';path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(f'synthetic {stem} {role}'.encode())
                pair[role]=dict(path=str(path.relative_to(evidence)),size=path.stat().st_size,sha256=file_digest(path))
            (evidence/stem/'audit.json').write_text('{"synthetic":true}')
            args['checkpoint_pairs'][f'{kind}/{index:02d}']=pair
            args['current_inventory']=inventory(evidence)
        def boundary():
            n=len(args['stages']);name=('pilot','lr','rank')[n]
            record=args['pilots'] if n==0 else args['records']
            rr=None if n==0 else args['rates'];ss=args['selection'] if n==2 else None
            indices=boundary_candidates(name,record,rr,ss)['audit_required_indices']
            inputs=dict(boundary=name,records=copy.deepcopy(record),rates=copy.deepcopy(rr),selection=copy.deepcopy(ss),
                pairs={i:copy.deepcopy(args['checkpoint_pairs'][f"{'pilot' if n==0 else 'quality'}/{i:02d}"]) for i in indices},
                evidence={name:b['sha256'] for name,b in inventory(evidence).items() if not name.endswith('.pt')})
            bundle=dict(inputs=inputs,inventory=inventory(evidence));context={'synthetic':name}
            cleanup.execute_boundary(evidence,name,fds,lambda:copy.deepcopy(bundle),lambda:context)
            ref=f'.retention/{name}/execution-review.json'
            args['stages'].append(dict(manifest=planned_manifest(**inputs),context_sha256=digest(context),
                execution_ref=dict(path=ref,sha256=file_digest(evidence/ref))))
            args['current_inventory']=inventory(evidence)
        yield args,add,boundary


def test_full_trajectory_and_cleanup_sequence(fixture):
    args,add,boundary=fixture
    assert storage.bind_storage(**args)['required_free_bytes']==30_631_281_412
    for spec in storage.pilot_specs():add('pilot',spec);storage.bind_storage(**args)
    boundary()
    for spec in rules.lr_specs():add('quality',spec);storage.bind_storage(**args)
    args['rates']=rules.choose_rates(args['records']);storage.bind_storage(**args);boundary()
    for spec in rules.rank_specs(args['rates']):add('quality',spec);storage.bind_storage(**args)
    args['selection']=rules.choose_ranks(args['records'][:9],args['records'][9:]);storage.bind_storage(**args);boundary()
    for spec in rules.confirmation_specs(args['selection']):add('quality',spec);report=storage.bind_storage(**args)
    assert report['projection']['phase']=='inference'
    assert report['required_free_bytes']==storage.RESERVE+storage.METADATA
    assert report['cleanup_boundaries_verified']==3 and report['storage_projection_fits']
    assert not report['dispatch_authorized'] and not report['audit_semantics_verified']
    assert len(list((args['campaign']/'evidence').rglob('*.pt')))==24  # selected3 + fresh9 pairs


def test_free_space_exact_boundary_and_one_byte_short(fixture,monkeypatch):
    args,_,_=fixture
    for free,success in [(30_631_281_412,True),(30_631_281_411,False),(storage.RESERVE+2*1024**3,False)]:
        monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=free))
        if success:assert storage.bind_storage(**args)['free_bytes']==free
        else:
            with pytest.raises(ValueError,match='remaining panel'):storage.bind_storage(**args)


def test_current_phase_increment_prevents_spending_future_capacity_credit(projection):
    pilots=rows(storage.pilot_specs());r=rows(rules.lr_specs());rates=rules.choose_rates(r)
    p=storage.remaining_projection(projection,pilots,r,rates,None,2)
    assert storage.required_free(p,10**12)==storage.RESERVE+storage.METADATA+p['remaining_current_phase_allowance_bytes']


@pytest.mark.parametrize('divisors',list(itertools.product((8,4,2),repeat=3)))
def test_all_27_final_choices_reserve_selected_screens_and_all_fresh_pairs(projection,divisors):
    pilots=rows(storage.pilot_specs());lr=rows(rules.lr_specs());rates=rules.choose_rates(lr)
    desired=dict(zip(rules.FAMILIES,divisors));rank=rows(rules.rank_specs(rates),6.)
    for r in rank:r['ce']=4. if r['divisor']==desired[r['family']] else 6.
    chosen=rules.choose_ranks(lr,rank)
    assert {f:v['divisor'] for f,v in chosen.items()}==desired
    p=storage.remaining_projection(projection,pilots,lr+rank,rates,chosen,3)
    b=storage.allocations(projection)
    assert p['remaining_current_phase_allowance_bytes']==3*sum(b[f,desired[f]] for f in rules.FAMILIES)
    assert p['remaining_peak_bytes']==4*sum(b[f,desired[f]] for f in rules.FAMILIES)


@pytest.mark.parametrize('bad',['pilot_order','premature_quality','premature_rates','missing_pilots','rank_before_cleanup','fresh_before_cleanup','wrong_rates','wrong_selection','bool_boundary'])
def test_scientific_progress_cannot_skip_a_storage_boundary(projection,bad):
    p=rows(storage.pilot_specs());r=rows(rules.lr_specs());rates=rules.choose_rates(r);sel=None;n=1
    if bad=='pilot_order':p.reverse()
    elif bad=='premature_quality':n=0
    elif bad=='premature_rates':r=r[:2]
    elif bad=='missing_pilots':p.pop()
    elif bad=='rank_before_cleanup':r+=rows(rules.rank_specs(rates))
    elif bad=='fresh_before_cleanup':
        r+=rows(rules.rank_specs(rates));sel=rules.choose_ranks(r[:9],r[9:]);n=2;r+=rows(rules.confirmation_specs(sel))
    elif bad=='wrong_rates':rates=dict.fromkeys(rules.FAMILIES,.0025)
    elif bad=='wrong_selection':
        r+=rows(rules.rank_specs(rates));n=3;sel={f:dict(lr=rates[f],divisor=2) for f in rules.FAMILIES}
    else:n=True
    with pytest.raises(ValueError):storage.remaining_projection(projection,p,r,rates,sel,n)


@pytest.mark.parametrize('bad',['missing_pair','unknown_weight','unreviewed_request','changed_checkpoint','unbound_metadata','stop','partial_cleanup','planned_only','receipt_change','selection_change'])
def test_no_storage_credit_from_incomplete_or_changed_evidence(fixture,bad):
    args,add,boundary=fixture
    for spec in storage.pilot_specs():add('pilot',spec)
    root=args['campaign'];evidence=root/'evidence'
    if bad=='missing_pair':args['checkpoint_pairs'].pop('pilot/00')
    elif bad=='unknown_weight':(evidence/'pilots/00/checkpoints/unknown.pt').write_bytes(b'x')
    elif bad=='unreviewed_request':
        p=evidence/'quality/00/request.json';p.parent.mkdir(parents=True);p.write_text('{}')
    elif bad=='changed_checkpoint':(evidence/'pilots/00/checkpoints/raw.pt').write_bytes(b'changed')
    elif bad=='unbound_metadata':(evidence/'meta/unknown.json').write_text('{}')
    elif bad=='stop':(root/'STOP').touch()
    elif bad=='partial_cleanup':(evidence/'.retention/pilot').mkdir(parents=True)
    else:
        boundary()
        if bad=='planned_only':args['stages'][0]['execution_ref']['path']='.retention/pilot/manifest.json'
        elif bad=='receipt_change':(evidence/'.retention/pilot/receipt.json').write_text('{}')
        else:args['stages'][0]['manifest']['records_sha256']='0'*64
    with pytest.raises((ValueError,FileNotFoundError)):storage.bind_storage(**args)


def test_free_space_changed_during_final_recheck_blocks(fixture,monkeypatch):
    args,_,_=fixture;values=iter((64*1024**3,storage.RESERVE))
    monkeypatch.setattr(storage.shutil,'disk_usage',lambda p:SimpleNamespace(free=next(values)))
    with pytest.raises(ValueError,match='changed'):storage.bind_storage(**args)


def test_sparse_checkpoint_gets_only_allocated_capacity_credit(fixture):
    args,add,_=fixture;add('pilot',storage.pilot_specs()[0]);entry=args['checkpoint_pairs']['pilot/00']['raw']
    path=args['campaign']/'evidence'/entry['path']
    with path.open('wb') as f:f.truncate(1024**2)
    entry.update(size=path.stat().st_size,sha256=file_digest(path))
    args['current_inventory']=inventory(args['campaign']/'evidence')
    report=storage.bind_storage(**args)
    other=args['campaign']/'evidence'/args['checkpoint_pairs']['pilot/00']['h200']['path']
    expected=sum(min(p.stat().st_size,p.stat().st_blocks*512) for p in (path,other))
    assert report['occupied_checkpoint_capacity_bytes']==expected
    assert report['occupied_checkpoint_capacity_bytes']<report['checkpoint_logical_bytes']


def test_same_size_checkpoint_change_after_hashing_is_rejected(fixture,monkeypatch):
    args,add,_=fixture;add('pilot',storage.pilot_specs()[0])
    path=args['campaign']/'evidence'/args['checkpoint_pairs']['pilot/00']['raw']['path']
    real=storage.snapshot;changed=False
    def mutate(root):
        nonlocal changed
        observed=real(root)
        if not changed:
            changed=True;data=path.read_bytes();path.write_bytes(b'!'+data[1:])
        return observed
    monkeypatch.setattr(storage,'snapshot',mutate)
    with pytest.raises(ValueError,match='changed'):storage.bind_storage(**args)


def test_no_extra_disk_capacity_credit_from_planned_pilot_deletion(fixture):
    args,add,_=fixture
    for spec in storage.pilot_specs():add('pilot',spec)
    report=storage.bind_storage(**args)
    assert report['cleanup_boundaries_verified']==0
    assert report['required_free_bytes']==30_631_281_412-report['occupied_checkpoint_capacity_bytes']
    assert len(list((args['campaign']/'evidence').rglob('*.pt')))==6
    # Occupied capacity is used only in the full future peak equation; it is
    # never added to the actual free-byte observation or a deletion claim.
    assert report['free_bytes']==64*1024**3


def test_completed_pair_over_original_allowance_rejected(fixture):
    args,add,_=fixture;add('pilot',storage.pilot_specs()[0])
    entry=args['checkpoint_pairs']['pilot/00']['raw'];entry['size']=2*1024**3
    with pytest.raises(ValueError,match='trajectory allowance'):storage.bind_storage(**args)


def test_checkpoint_hardlink_alias_rejected_even_with_updated_inventory(fixture):
    import os
    args,add,_=fixture;add('pilot',storage.pilot_specs()[0]);root=args['campaign']/'evidence'
    pair=args['checkpoint_pairs']['pilot/00'];raw=root/pair['raw']['path'];export=root/pair['h200']['path']
    export.unlink();os.link(raw,export)
    pair['h200'].update(size=export.stat().st_size,sha256=file_digest(export))
    args['current_inventory']=inventory(root)
    with pytest.raises(ValueError):storage.bind_storage(**args)
