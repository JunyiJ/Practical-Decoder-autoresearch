"""Disposable bytes and real temporary file locks only; no real weights removed."""
import copy
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path

import pytest

from src.research import tokenizer_wide_hybrid_cleanup as cleanup
from src.research import tokenizer_wide_hybrid_rules as rules
from src.research.protocol import digest, file_digest
from src.research.tokenizer_wide_hybrid_retention_receipts import planned_manifest


@contextmanager
def fixture(tmp_path,monkeypatch):
    root=tmp_path/'evidence';root.mkdir()
    records=[dict(s,status='completed',ce=5.) for s in rules.lr_specs()]
    rates=rules.choose_rates(records);pairs={}
    for i in range(9):
        folder=root/f'trial-{i}'/'checkpoints';folder.mkdir(parents=True)
        pairs[i]={}
        for role in ('raw','h200'):
            p=folder/f'{role}.pt';p.write_bytes(f'synthetic {i} {role}'.encode())
            pairs[i][role]=dict(path=str(p.relative_to(root)),size=p.stat().st_size,sha256=file_digest(p))
    # Extra protected fresh/failure evidence, outside the candidate stage.
    for name in ('fresh/checkpoints/raw.pt','fresh/checkpoints/h200.pt','failure/checkpoints/partial.pt','failed-log.txt','audit.json'):
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'protected synthetic data')
    evidence={'audit.json':file_digest(root/'audit.json'),'failed-log.txt':file_digest(root/'failed-log.txt')}
    inputs=dict(boundary='lr',records=records,rates=rates,selection=None,pairs=pairs,evidence=evidence)
    inventory={str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=file_digest(p)) for p in root.rglob('*') if p.is_file()}
    bundle=dict(inputs=inputs,inventory=inventory)
    global_path=tmp_path/'global.lock';monkeypatch.setattr(cleanup,'GLOBAL_LOCK',global_path)
    with global_path.open('a+b') as global_fd,(tmp_path/'.lock').open('a+b') as campaign_fd:
        for f in (global_fd,campaign_fd): fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        yield root,bundle,(global_fd.fileno(),campaign_fd.fileno())


def identity():return dict(synthetic_only=True,active=None,stop=False)


def run(root,bundle,fds):
    return cleanup.execute_boundary(root,'lr',fds,lambda:copy.deepcopy(bundle),identity)


def execution_ref(root):
    name='.retention/lr/execution-review.json'
    return dict(path=name,sha256=file_digest(root/name))


def test_cleanup_and_complete_independent_reverification(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        expected=planned_manifest(**bundle['inputs'])
        report=run(root,bundle,fds)
        assert report['deleted_count']==12
        assert report['deleted_logical_bytes']==sum(e['size'] for e in expected['delete'])
        for item in expected['delete']:assert not (root/item['path']).exists()
        for item in expected['retain']:assert file_digest(root/item['path'])==item['sha256']
        checked=cleanup.recheck_execution(root,expected,execution_ref(root),digest(identity()))
        assert checked['protected_files_verified']==len(bundle['inventory'])-12
        assert checked['dispatch_authorized'] is False
        before={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}
        with pytest.raises(FileExistsError):run(root,bundle,fds)
        assert before=={p:p.read_bytes() for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('mutation',['changed_candidate','changed_protected','missing_inventory','unknown_file','symlink','wrong_lock','confirmation'])
def test_preflight_failure_never_removes_checkpoint(tmp_path,monkeypatch,mutation):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        if mutation=='changed_candidate':(root/'trial-3/checkpoints/raw.pt').write_bytes(b'changed')
        elif mutation=='changed_protected':(root/'fresh/checkpoints/raw.pt').write_bytes(b'changed')
        elif mutation=='missing_inventory':bundle['inventory'].pop('failed-log.txt')
        elif mutation=='unknown_file':(root/'unknown.txt').write_bytes(b'unlisted')
        elif mutation=='symlink':(root/'link').symlink_to(root/'audit.json')
        elif mutation=='wrong_lock':fds=fds[::-1]
        elif mutation=='confirmation':bundle['inputs']['boundary']='confirm'
        before={p:p.read_bytes() for p in root.rglob('*.pt')}
        with pytest.raises((ValueError,FileNotFoundError)):run(root,bundle,fds)
        assert before=={p:p.read_bytes() for p in root.rglob('*.pt')}


def test_context_change_after_one_removal_records_failure_and_cannot_retry(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        calls=0
        def changing():
            nonlocal calls
            calls+=1
            # Initial, preflight and first deletion succeed; second is blocked.
            return dict(identity(),stop=calls>=4)
        with pytest.raises(ValueError,match='context changed'):
            cleanup.execute_boundary(root,'lr',fds,lambda:copy.deepcopy(bundle),changing)
        owned=root/'.retention/lr'
        failure=json.loads((owned/'failure.json').read_text())
        assert len(failure['confirmed_deleted'])==1 and failure['retry_allowed'] is False
        assert not (owned/'execution-review.json').exists()
        assert not (owned/'receipt.json').exists()
        events=[json.loads(line) for line in (owned/'deletions.jsonl').read_text().splitlines()]
        assert [e['event'] for e in events]==['intent','deleted']
        with pytest.raises(FileExistsError):run(root,bundle,fds)


def test_unlink_error_keeps_intent_and_blocks_completion(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        original=Path.unlink
        def fail_weight(self,*a,**k):
            if self.suffix=='.pt':raise OSError('synthetic unlink failure')
            return original(self,*a,**k)
        monkeypatch.setattr(Path,'unlink',fail_weight)
        with pytest.raises(OSError,match='synthetic unlink failure'):run(root,bundle,fds)
        owned=root/'.retention/lr'
        assert not (owned/'receipt.json').exists()
        assert len(list(root.rglob('*.pt')))==21
        events=[json.loads(line) for line in (owned/'deletions.jsonl').read_text().splitlines()]
        assert [e['event'] for e in events]==['intent']
        assert json.loads((owned/'failure.json').read_text())['confirmed_deleted']==[]


@pytest.mark.parametrize('mutation',['protected','journal','guard','missing_marker','context'])
def test_completed_receipt_alone_is_not_enough(tmp_path,monkeypatch,mutation):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        expected=planned_manifest(**bundle['inputs']);run(root,bundle,fds)
        owned=root/'.retention/lr';ref=execution_ref(root);context=digest(identity())
        if mutation=='protected':(root/'fresh/checkpoints/raw.pt').write_bytes(b'changed')
        elif mutation in ('journal','guard'):
            name='deletions.jsonl' if mutation=='journal' else 'guard.json'
            if mutation=='journal':(owned/name).write_text('')
            else:
                g=json.loads((owned/name).read_text());g['status']='failed';(owned/name).write_text(json.dumps(g))
            review=json.loads((owned/'execution-review.json').read_text())
            review['files'][name]=file_digest(owned/name)
            (owned/'execution-review.json').write_text(json.dumps(review));ref=execution_ref(root)
        elif mutation=='missing_marker':(owned/'execution-review.json').unlink()
        else:context='0'*64
        with pytest.raises(ValueError):cleanup.recheck_execution(root,expected,ref,context)


def test_hard_exit_after_unlink_leaves_durable_intent_and_closed_attempt(tmp_path,monkeypatch):
    import subprocess
    import sys
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        saved=tmp_path/'synthetic-input.json';saved.write_text(json.dumps(bundle))
        code='''
import json,os,sys
from pathlib import Path
from src.research import tokenizer_wide_hybrid_cleanup as c
root=Path(sys.argv[1]);bundle=json.loads(Path(sys.argv[2]).read_text())
bundle['inputs']['pairs']={int(k):v for k,v in bundle['inputs']['pairs'].items()}
c.GLOBAL_LOCK=Path(sys.argv[3]);original=Path.unlink
def interrupted(path,*a,**k):
    result=original(path,*a,**k)
    if path.suffix=='.pt':os._exit(77)
    return result
Path.unlink=interrupted
c.execute_boundary(root,'lr',tuple(map(int,sys.argv[4:])),lambda:bundle,
                   lambda:dict(synthetic_only=True,active=None,stop=False))
'''
        result=subprocess.run([sys.executable,'-c',code,str(root),str(saved),str(cleanup.GLOBAL_LOCK),*map(str,fds)],
            cwd=Path(__file__).resolve().parents[1],pass_fds=fds,timeout=15,capture_output=True)
        assert result.returncode==77, result.stderr
        owned=root/'.retention/lr'
        events=[json.loads(line) for line in (owned/'deletions.jsonl').read_text().splitlines()]
        assert [e['event'] for e in events]==['intent']
        assert not (root/events[0]['entry']['path']).exists()
        assert not (owned/'receipt.json').exists() and not (owned/'execution-review.json').exists()
        with pytest.raises(FileExistsError):run(root,bundle,fds)


def test_protected_postcheck_failure_never_publishes_completed_receipt(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        original=Path.unlink
        def corrupt_protected(path,*a,**k):
            result=original(path,*a,**k)
            if path.suffix=='.pt':(root/'fresh/checkpoints/raw.pt').write_bytes(b'externally changed synthetic protected file')
            return result
        monkeypatch.setattr(Path,'unlink',corrupt_protected)
        with pytest.raises(ValueError):run(root,bundle,fds)
        owned=root/'.retention/lr'
        assert not (owned/'receipt.json').exists() and not (owned/'execution-review.json').exists()
        failure=json.loads((owned/'failure.json').read_text())
        assert len(failure['confirmed_deleted'])==12 and failure['retry_allowed'] is False


def test_document_cap_fails_before_any_checkpoint_removal(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        monkeypatch.setattr(cleanup,'DOCUMENT_CAP',1024)
        before={p:p.read_bytes() for p in root.rglob('*.pt')}
        with pytest.raises(ValueError,match='document exceeds'):run(root,bundle,fds)
        assert before=={p:p.read_bytes() for p in root.rglob('*.pt')}
        assert not (root/'.retention/lr/receipt.json').exists()


def test_inventory_count_cap_is_fail_closed(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        monkeypatch.setattr(cleanup,'MAX_FILES',2)
        with pytest.raises(ValueError,match='inventory'):run(root,bundle,fds)
        assert len(list(root.rglob('*.pt')))==21


def test_aggregate_metadata_cap_fails_before_checkpoint_removal(tmp_path,monkeypatch):
    with fixture(tmp_path,monkeypatch) as (root,bundle,fds):
        monkeypatch.setattr(cleanup,'ATTEMPT_CAP',1024)
        before={p:p.read_bytes() for p in root.rglob('*.pt')}
        with pytest.raises(RuntimeError,match='metadata cap'):run(root,bundle,fds)
        assert before=={p:p.read_bytes() for p in root.rglob('*.pt')}
        assert not (root/'.retention/lr/execution-review.json').exists()
