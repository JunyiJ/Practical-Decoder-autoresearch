"""Exact retirement of four completed numerical-only general-data artifacts."""
import fcntl
import json
from pathlib import Path
import shutil
from research.general_rsi_training_v1 import sha
from research.general_rsi_worker_v1 import put, retire

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-capacity-001'


def main():
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pilot=ROOT/'research/tokenizer-evidence/general-rsi-pilot-001'
        qualifier=ROOT/'research/tokenizer-evidence/general-rsi-qualification-001'
        for base in (pilot,qualifier):
            guard=json.loads((base/'guard.json').read_text())
            assert guard['status']=='finished' and guard['reason'] is None
            assert not (base/'failure.json').exists()
        p=json.loads((pilot/'result.json').read_text())
        q=json.loads((qualifier/'result.json').read_text())
        assert p['paired_batches'] and not p['quality_selection']
        assert q['complete_restart_qualified'] and q['document_scoring_qualified'] and not q['scientific_selection']
        targets=[]
        for result in (p,q):
            for arm in result['arms']:
                receipt=arm['checkpoint'];path=Path(receipt['path'])
                assert path.stat().st_size==receipt['size'] and sha(path)==receipt['sha256']
                targets.append(dict(receipt=receipt,inode=path.stat().st_ino))
        # These four payloads were never trained as scientific screens/confirmations.
        # The qualified training worker reconstructs fresh models and uses only
        # their metric/qualification receipts, not these checkpoint payloads.
        protected={str(f):sha(f) for base in (pilot,qualifier) for f in base.iterdir() if f.is_file()}
        for name in ('general_rsi_training_v1.py','general_rsi_data_v1.py','general_rsi_pilot_v1.py','general_rsi_qualification_v1.py'):
            path=ROOT/'research'/name;protected[str(path)]=sha(path)
        recipe=json.loads((ROOT/'research/tokenizer-evidence/public-inference-correction-001/adoption/recipe.json').read_text())
        for name in ('parent','h200'):
            record=recipe['representative'][name]
            assert sha(record['path'])==record['sha256']
            protected[record['path']]=record['sha256']
        OUT.mkdir()
        manifest=dict(targets=targets,protected=protected,source_sha256=sha(__file__),
            free_before=shutil.disk_usage(ROOT).free,
            dependency_closure='Execution and exact restart/scoring prerequisites both passed. New scientific trajectories reconstruct from fresh roles and codebook, never these numerical-only weights. Full original evidence and selected Shakespeare weights retained.',
            authority='Current human AGENTS.md permits obsolete concluded numerical checkpoints after exact hash/consumer/dependency audit. This narrowly supersedes the prior factual statement that these four pilot weights remained retained.',
            requirement='Full panel retained-output forecast plus10GiB reserve (8GiB floor plus2GiB headroom).')
        put(OUT/'manifest.json',manifest)
        evidence=dict(path=str(qualifier/'result.json'),sha256=sha(qualifier/'result.json'))
        for target in targets:
            receipt=target['receipt'];path=Path(receipt['path'])
            assert path.stat().st_ino==target['inode']
            # Same exact deletion safeguards, explicitly admitted pilot directory.
            retire(receipt,OUT/'deletion-journal.jsonl',manifest['dependency_closure'],evidence)
        assert all(sha(path)==digest for path,digest in protected.items())
        put(OUT/'completed.json',dict(removed_bytes=sum(t['receipt']['size'] for t in targets),
            removed_files=len(targets),protected_files_verified=len(protected),
            free_after=shutil.disk_usage(ROOT).free,manifest_sha256=sha(OUT/'manifest.json')))
        print((OUT/'completed.json').read_text())


if __name__=='__main__':main()
