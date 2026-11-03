"""One-shot, exact retirement of the closed hybrid study's losing control payloads."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CAMPAIGN = ROOT / 'research_runs/tokenizer-hybrid-001'
EVIDENCE = ROOT / 'research/tokenizer-evidence/hybrid-001'
NAME = 'screen-gated_mla-w64-s3001-lr0.0025'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    p = Path(path)
    assert p.is_absolute() and p.is_relative_to(ROOT)
    assert all(not q.is_symlink() for q in [p, *p.parents])
    s = p.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid()
    return dict(path=str(p), size=s.st_size, allocated_bytes=s.st_blocks*512,
                device=s.st_dev, inode=s.st_ino, nlink=s.st_nlink,
                uid=s.st_uid, mtime_ns=s.st_mtime_ns)


def bound(ref):
    p = Path(ref['path'])
    assert digest(p) == ref['sha256'], str(p)
    if 'size' in ref:
        assert p.stat().st_size == ref['size'], str(p)


def dump(name, data):
    with (HERE / name).open('x') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def consumers(paths):
    ps = subprocess.check_output(['/bin/ps', '-axo', 'pid=,comm=,args='], text=True)
    active = []
    for line in ps.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3 or int(parts[0]) == os.getpid():
            continue
        if ('python' in parts[1].lower() or 'pytest' in parts[1].lower()) and (
                str(ROOT) in parts[2] or 'src.research' in parts[2] or 'research/' in parts[2]):
            active.append(line)
    assert not active, active
    proc = subprocess.run(['/usr/sbin/lsof', '-nP', '--', *map(str, paths)],
                          text=True, capture_output=True)
    assert proc.returncode == 1 and not proc.stdout.strip() and not proc.stderr.strip(), (
        proc.returncode, proc.stdout, proc.stderr)
    return dict(observed_unix=time.time(), active_research_python=[],
                target_open_handles=[], lsof_returncode=proc.returncode)


def prove_closed(candidate):
    state = json.loads((CAMPAIGN / 'state.json').read_text())
    review = json.loads((EVIDENCE / 'final-review.json').read_text())
    assert state['status'] == 'completed' and state['active'] is None
    assert review['status'] == 'completed' and review['all_twelve_artifact_reviews_passed']
    assert review['source_runtime_protocol_recipe_corpus_resource_unchanged']
    records = state['records']
    assert len(records) == 12 and all(r['status'] == 'completed' for r in records)
    screens = [r for r in records if r['stage'] == 'screen']
    assert len(screens) == 6
    winners = {f: min((r for r in screens if r['family'] == f),
                     key=lambda r: (r['h200']['mean_ce'], r['lr']))
               for f in ['hybrid', 'gated_mla', 'kda']}
    control = min(['gated_mla', 'kda'], key=lambda f: (winners[f]['h200']['mean_ce'], f))
    expected = dict(control=control, rates={f:r['lr'] for f,r in winners.items()})
    assert state['selection'] == expected == json.loads((EVIDENCE/'selection.json').read_text())
    assert control == 'kda' and winners['gated_mla']['id'] == NAME
    fresh = [r for r in records if r['stage'] == 'confirm']
    assert sorted((r['family'],r['seed']) for r in fresh) == sorted(
        (f,s) for f in ['hybrid','kda'] for s in [3011,3012,3013])
    result = json.loads(Path(candidate['acceptance']['path']).read_text())
    assert result['raw']['checkpoint_sha256'] == candidate['original']['sha256']
    assert result['h200']['checkpoint_sha256'] == candidate['h200']['sha256']
    individual = json.loads((EVIDENCE/NAME/'review.json').read_text())
    assert individual['status'] == 'passed'
    assert individual['result_sha256'] == candidate['acceptance']['sha256']
    needles = [NAME, candidate['original']['sha256'], candidate['h200']['sha256']]
    scanned = []
    for folder in [ROOT/'src/research', ROOT/'research/final_ln_scale_model_v1',
                   ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001']:
        for p in folder.rglob('*.py'):
            assert not any(n in p.read_text() for n in needles), str(p)
            scanned.append(str(p))
    recipe = ROOT/'research/tokenizer-evidence/adamw-epsilon-preparation-001/adopted-width256-mla-epsilon-recipe.json'
    assert not any(n in recipe.read_text() for n in needles)
    return dict(selection_recomputed=expected, terminal_status='completed',
                fresh_pairs_closed=[3011,3012,3013],
                current_recipe=dict(path=str(recipe),sha256=digest(recipe)),
                source_dependency_files_checked=scanned,
                dependency_basis='Original protocol selects one control, final review closes all three hybrid/KDA pairs; no pending gated-MLA confirmation or inference. No exact target references in research Python/current preparation/current adopted recipe. Historical acceptance and archival references are evidence, not active consumers.')


def verify_member(candidate):
    proc = subprocess.Popen(['/opt/homebrew/bin/zstd','-d','--long=30','--stdout',
                             candidate['archive']['path']], stdout=subprocess.PIPE)
    members = []
    with tarfile.open(fileobj=proc.stdout, mode='r|') as tf:
        for member in tf:
            assert member.isfile() and not member.issym() and not member.islnk()
            assert member.name == str(Path(candidate['original']['path']).relative_to(ROOT))
            assert member.size == candidate['original']['size']
            h = hashlib.sha256()
            with tf.extractfile(member) as f:
                for block in iter(lambda:f.read(4*1024**2),b''):
                    h.update(block)
            assert h.hexdigest() == candidate['original']['sha256']
            members.append(dict(name=member.name,size=member.size,sha256=h.hexdigest()))
    while proc.stdout.read(4*1024**2):
        pass
    assert proc.wait() == 0 and len(members) == 1
    return members


def plan():
    audit = json.loads((HERE/'archive-audit.json').read_text())
    eligible = [r for r in audit['rows'] if r['classification']=='obsolete_losing_control_screen']
    assert len(eligible) == 1
    candidate = eligible[0]
    assert NAME in candidate['original']['path']
    for ref in audit['evidence']:
        bound(ref)
    targets = []
    for ref in [candidate['archive'],candidate['h200']]:
        bound(ref)
        info = identity(ref['path'])
        assert info['nlink'] == 1
        targets.append(dict(info,sha256=ref['sha256'],reason='Obsolete learning-rate winner for losing gated-MLA control; final control KDA, completed fresh hybrid/KDA dependencies closed.'))
    check = consumers([r['path'] for r in targets])
    proof = prove_closed(candidate)
    members = verify_member(candidate)
    paths = {Path(r['path']) for r in audit['evidence']}
    for folder in [CAMPAIGN,EVIDENCE]:
        paths.update(p for p in folder.rglob('*') if p.is_file() and
                     '__pycache__' not in p.parts and not p.name.endswith('.lock'))
    paths.add(ROOT/'src/research/tokenizer_hybrid_study.py')
    for row in audit['rows']:
        if str(CAMPAIGN) in row['original']['path'] and row != candidate:
            for ref in [row['archive'],row['h200']]:
                bound(ref)
                paths.add(Path(ref['path']))
    paths -= {Path(r['path']) for r in targets}
    retained = [dict(identity(p),sha256=digest(p)) for p in sorted(paths)]
    archives = [identity(Path(r['archive']['path'])) for r in audit['rows'] if r != candidate]
    # The six newly archived epsilon endpoints are outside the historical audit;
    # preserve every one and bind their independent archive receipt.
    new_receipt = ROOT/'research/tokenizer-evidence/storage-recovery-028/raw-completed.json'
    latest = json.loads(new_receipt.read_text())
    assert len(latest['files']) == 6 and latest['all_h200_unchanged'] and latest['all_raw_bytes_preserved']
    for r in latest['files']:
        s=identity(r['archive']['path'])
        assert s['size']==r['archive']['size']
        archives.append(s)
    retained.append(dict(identity(new_receipt),sha256=digest(new_receipt)))
    manifest = dict(created_unix=time.time(),script_sha256=digest(__file__),
        audit_sha256=digest(HERE/'archive-audit.json'),candidate=candidate,files=targets,
        allocated_bytes=sum(r['allocated_bytes'] for r in targets),
        archive_member_verified=members,dependency_proof=proof,consumer_check=check,
        retained_evidence=retained,retained_archive_identity=archives,
        policy_exception='Only the two exact losing-control payloads supersede storage010 selected_or_fresh labeling and prior all197 retention statements. Original failure/acceptance/config/metrics/source/archive receipts remain immutable. All actual selected models and fresh confirmations remain retained.')
    dump('manifest.json',manifest)
    print(json.dumps(dict(status='planned',targets=targets,retained_hashed=len(retained),retained_archive_count=len(archives))))


def apply():
    assert not (HERE/'deletion-journal.jsonl').exists(), 'One-shot only; inspect journal on partial failure.'
    m=json.loads((HERE/'manifest.json').read_text())
    assert digest(__file__)==m['script_sha256']
    assert digest(HERE/'archive-audit.json')==m['audit_sha256']
    prove_closed(m['candidate'])
    for r in m['retained_evidence']:
        bound(r)
    for r in m['retained_archive_identity']:
        assert identity(r['path'])==r
    for r in m['files']:
        bound(r)
        assert identity(r['path'])=={k:r[k] for k in identity(r['path'])}
    check=consumers([r['path'] for r in m['files']])
    before=shutil.disk_usage(ROOT).free
    with (HERE/'deletion-journal.jsonl').open('x') as journal:
        def event(data):
            journal.write(json.dumps(dict(observed_unix=time.time(),**data))+'\n')
            journal.flush();os.fsync(journal.fileno())
        event(dict(event='validated',manifest_sha256=digest(HERE/'manifest.json'),free_bytes=before,consumer_check=check))
        for r in m['files']:
            bound(r)
            assert identity(r['path'])=={k:r[k] for k in identity(r['path'])}
            consumers([r['path']])
            event(dict(event='unlink_intent',file=r))
            Path(r['path']).unlink()
            fd=os.open(str(Path(r['path']).parent),os.O_RDONLY)
            try:os.fsync(fd)
            finally:os.close(fd)
            event(dict(event='unlinked',path=r['path'],allocated_bytes=r['allocated_bytes']))
        for r in m['retained_evidence']:
            bound(r)
        for r in m['retained_archive_identity']:
            assert identity(r['path'])==r
        assert all(not Path(r['path']).exists() for r in m['files'])
        after=shutil.disk_usage(ROOT).free
        result=dict(status='completed',manifest_sha256=digest(HERE/'manifest.json'),
                    removed_files=len(m['files']),allocated_bytes_removed=m['allocated_bytes'],
                    free_before=before,free_after=after,observed_free_delta=after-before,
                    retained_evidence_hashes_verified=len(m['retained_evidence']),
                    retained_archive_identities_unchanged=len(m['retained_archive_identity']),
                    retained_selected_and_fresh_payloads=True,shared_worker_lock_held=True)
        event(dict(event='completed',result=result))
    dump('completed.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    assert sys.argv[1:] in [['plan'],['apply']]
    lock=ROOT/'research_state/tokenizer-worker-lock/.lock'
    assert lock.exists() and not lock.is_symlink()
    with lock.open('r+') as f:
        fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        (plan if sys.argv[1]=='plan' else apply)()
