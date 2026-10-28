"""Real cleanup receipts over disposable tiny bytes; no model or corpus use."""
import copy
from contextlib import contextmanager
import fcntl
import json

import pytest

from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research import tokenizer_wide_hybrid_rules as rules
from src.research.protocol import digest, file_digest
from src.research.tokenizer_wide_hybrid_retention import _pilot_specs, boundary_candidates
from src.research.tokenizer_wide_hybrid_retention_chain import recheck_chain
from src.research.tokenizer_wide_hybrid_retention_receipts import planned_manifest


def inventory(root):
    return {str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=file_digest(p))
            for p in root.rglob('*') if p.is_file()}


def pair(root,folder):
    result={}
    for role in ('raw','h200'):
        path=root/folder/'checkpoints'/f'{role}.pt';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(f'synthetic {folder} {role}'.encode())
        result[role]=dict(path=str(path.relative_to(root)),size=path.stat().st_size,sha256=file_digest(path))
    return result


@contextmanager
def campaign(tmp_path,monkeypatch,stages=3,keep_anchors=False,rank_attack=None):
    root=tmp_path/'evidence';root.mkdir()
    (root/'audit.json').write_text('{"synthetic":true}')
    failure=pair(root,'failure')
    global_path=tmp_path/'global.lock';monkeypatch.setattr(cleanup,'GLOBAL_LOCK',global_path)
    chain=[];records=[];pairs={};rates=None;selection=None
    evidence={'audit.json':file_digest(root/'audit.json')}
    with global_path.open('a+b') as g,(tmp_path/'.lock').open('a+b') as c:
        fds=(g.fileno(),c.fileno())
        for fd in fds:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for boundary in ['pilot','lr','rank'][:stages]:
            if boundary=='pilot':
                records=[dict(s,status='completed',ce=5.) for s in _pilot_specs()]
                pairs={i:pair(root,f'pilot-{i}') for i in range(3)}
            elif boundary=='lr':
                records=[dict(s,status='completed',ce=5.) for s in rules.lr_specs()]
                pairs={i:pair(root,f'trial-{i}') for i in range(9)}
                rates=rules.choose_rates(records)
            else:
                ranks=[dict(s,status='completed',ce=6. if keep_anchors else 4.) for s in rules.rank_specs(rates)]
                records+=ranks
                pairs.update({i:pair(root,f'trial-{i}') for i in range(9,15)})
                selection=rules.choose_ranks(records[:9],records[9:])
                indices=boundary_candidates('rank',records,rates,selection)['audit_required_indices']
                pairs={i:pairs[i] for i in indices}
                if rank_attack=='protected_pair':pairs[0]=failure
                if rank_attack=='index_alias':pairs[0],pairs[1]=pairs[1],pairs[0]
                if rank_attack=='protected_probe':pairs[12]=failure
            inputs=dict(boundary=boundary,records=copy.deepcopy(records),rates=rates,selection=selection,
                        pairs=copy.deepcopy(pairs),evidence=evidence)
            bundle=dict(inputs=inputs,inventory=inventory(root))
            context=dict(campaign='synthetic',boundary=boundary,source='unchanged')
            cleanup.execute_boundary(root,boundary,fds,lambda:copy.deepcopy(bundle),lambda:context)
            name=f'.retention/{boundary}/execution-review.json'
            chain.append(dict(manifest=planned_manifest(**inputs),context_sha256=digest(context),
                              execution_ref=dict(path=name,sha256=file_digest(root/name))))
        yield root,chain


@pytest.mark.parametrize('count',[1,2,3])
def test_complete_prefix_and_new_fresh_pairs(tmp_path,monkeypatch,count):
    with campaign(tmp_path,monkeypatch,stages=count) as (root,chain):
        pair(root,'fresh')
        checked=recheck_chain(root,chain,inventory(root))
        assert checked['boundaries']==count and checked['dispatch_authorized'] is False
        assert len(checked['deleted_files'])==[6,18,30][count-1]
        assert checked['protected_files_verified']==len(inventory(root))
        if count==3:
            # Boundary-local validation still rejects missing LR winners. Only
            # the completed rank execution proves their legitimate retirement.
            lr=chain[1]
            with pytest.raises((ValueError,FileNotFoundError)):
                cleanup.recheck_execution(root,lr['manifest'],lr['execution_ref'],lr['context_sha256'])
            assert all(not (root/e['path']).exists() for e in lr['manifest']['retain'])


def test_lr_anchors_that_win_rank_remain_live(tmp_path,monkeypatch):
    with campaign(tmp_path,monkeypatch,keep_anchors=True) as (root,chain):
        checked=recheck_chain(root,chain,inventory(root))
        assert len(checked['deleted_files'])==30
        assert all((root/e['path']).exists() for e in chain[1]['manifest']['retain'])


@pytest.mark.parametrize('attack',['protected_pair','index_alias'])
def test_later_receipt_cannot_retire_different_prior_pair(tmp_path,monkeypatch,attack):
    with campaign(tmp_path,monkeypatch,rank_attack=attack) as (root,chain):
        with pytest.raises(ValueError,match='same complete LR anchor pairs'):
            recheck_chain(root,chain,inventory(root))


@pytest.mark.parametrize('mutation',[
    'empty','missing_pilot','missing_lr','reverse','duplicate','missing_rank_receipt',
    'context','manifest','resurrected','missing_selected','missing_failure','changed_log',
    'unknown_file','omitted_current','symlink','failed_attempt',
])
def test_invalid_chain_or_changed_evidence_is_rejected(tmp_path,monkeypatch,mutation):
    with campaign(tmp_path,monkeypatch) as (root,chain):
        current=inventory(root)
        if mutation=='empty':chain=[]
        elif mutation=='missing_pilot':chain=chain[1:]
        elif mutation=='missing_lr':chain=[chain[0],chain[2]]
        elif mutation=='reverse':chain=list(reversed(chain))
        elif mutation=='duplicate':chain=[chain[0],chain[1],chain[1]]
        elif mutation=='missing_rank_receipt':(root/'.retention/rank/receipt.json').unlink()
        elif mutation=='context':chain[2]['context_sha256']='0'*64
        elif mutation=='manifest':chain[2]['manifest']['rates_sha256']='0'*64
        elif mutation=='resurrected':(root/chain[0]['manifest']['delete'][0]['path']).write_bytes(b'recreated')
        elif mutation=='missing_selected':(root/chain[2]['manifest']['retain'][0]['path']).unlink()
        elif mutation=='missing_failure':(root/'failure/checkpoints/raw.pt').unlink()
        elif mutation=='changed_log':(root/'audit.json').write_text('{}')
        elif mutation=='unknown_file':(root/'unknown').write_bytes(b'unbound')
        elif mutation=='omitted_current':current.pop('failure/checkpoints/raw.pt')
        elif mutation=='symlink':(root/'link').symlink_to(root/'audit.json')
        elif mutation=='failed_attempt':
            (root/'.retention/rank/failure.json').write_text('{}');current=inventory(root)
        with pytest.raises((ValueError,FileNotFoundError)):
            recheck_chain(root,chain,current)


def test_current_inventory_cannot_rebind_old_protected_bytes(tmp_path,monkeypatch):
    with campaign(tmp_path,monkeypatch) as (root,chain):
        (root/'failure/checkpoints/raw.pt').write_bytes(b'changed')
        with pytest.raises(ValueError,match='changed protected evidence'):
            recheck_chain(root,chain,inventory(root))


def test_missing_rank_completion_never_excuses_retired_lr_anchor(tmp_path,monkeypatch):
    with campaign(tmp_path,monkeypatch) as (root,chain):
        with pytest.raises(ValueError,match='Every existing cleanup attempt'):
            recheck_chain(root,chain[:2],inventory(root))


def test_rank_probe_cannot_disguise_previously_protected_weights(tmp_path,monkeypatch):
    with campaign(tmp_path,monkeypatch,rank_attack='protected_probe') as (root,chain):
        with pytest.raises(ValueError,match='not an eligible LR anchor'):
            recheck_chain(root,chain,inventory(root))


def test_even_empty_interrupted_attempt_blocks_partial_chain(tmp_path,monkeypatch):
    with campaign(tmp_path,monkeypatch,stages=2) as (root,chain):
        (root/'.retention/rank').mkdir()
        with pytest.raises(ValueError,match='Every existing cleanup attempt'):
            recheck_chain(root,chain,inventory(root))


def test_cannot_ignore_rank_receipt_even_when_lr_anchors_survive(tmp_path,monkeypatch):
    with campaign(tmp_path,monkeypatch,keep_anchors=True) as (root,chain):
        with pytest.raises(ValueError,match='Every existing cleanup attempt'):
            recheck_chain(root,chain[:2],inventory(root))


def test_later_inventory_cannot_omit_original_failure_evidence(tmp_path,monkeypatch):
    with campaign(tmp_path,monkeypatch) as (root,chain):
        # Re-sign the synthetic rank execution after falsifying its original
        # inventory and matching journal/counts. Cross-stage continuity must
        # still reject this, independent of the local receipt hash checks.
        base=root/'.retention/rank'
        original=json.loads((base/'inventory.json').read_text())
        original.pop('failure/checkpoints/raw.pt')
        (base/'inventory.json').write_text(json.dumps(original))
        events=[json.loads(line) for line in (base/'deletions.jsonl').read_text().splitlines()]
        events[-1]['files']-=1
        events[-1]['inventory_sha256']=file_digest(base/'inventory.json')
        (base/'deletions.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        review=json.loads((base/'execution-review.json').read_text())
        review['protected_files_verified']-=1
        for name in ('inventory.json','deletions.jsonl'):review['files'][name]=file_digest(base/name)
        (base/'execution-review.json').write_text(json.dumps(review))
        chain[-1]['execution_ref']['sha256']=file_digest(base/'execution-review.json')
        with pytest.raises(ValueError,match='Later inventory omitted or changed'):
            recheck_chain(root,chain,inventory(root))
