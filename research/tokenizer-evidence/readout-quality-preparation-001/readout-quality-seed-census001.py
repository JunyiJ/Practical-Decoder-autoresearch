"""Prospective six scientific roles and four derived contexts; read-only text census."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
E = Path(__file__).resolve().parent
R = E/'readout-quality-seed-census001'
CAP = 32*1024**2


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def publish(name, value):
    data = (json.dumps(value, indent=2, sort_keys=True)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(data) <= CAP
    with (R/name).open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan-sha', required=True)
    arg = parser.parse_args()
    assert sha(R/'plan.json') == arg.plan_sha
    assert set(p.name for p in R.iterdir()) == {'plan.json'}
    plan = json.loads((R/'plan.json').read_text())
    for p, h in plan['sources'].items():
        assert sha(ROOT/p) == h, p
    assert len(plan['sources']) == 1126
    roles = plan['roles']
    assert len(roles) == 10 and len(set(roles.values())) == 10
    seeds = plan['seeds']
    assert [roles[n] for n in ['screen', 'evaluation', 'fresh0', 'fresh1', 'fresh2', 'bootstrap']] == [seeds['screen'], seeds['evaluation'], *seeds['fresh'], seeds['bootstrap']]
    for name, base in [('screen', seeds['screen']), *[(f'fresh{i}', seed) for i, seed in enumerate(seeds['fresh'])]]:
        assert roles[name+'_context'] == int.from_bytes(hashlib.sha256(('readout_mixture_model_v1:context:'+str(base)).encode()).digest()[:8], 'big') % 2**63
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                pattern = re.compile(rb'(?<![\w.])(?:'+b'|'.join(str(x).encode() for x in plan['roles'].values())+rb')(?![\w.])')
                files, total, output, matches = 0, 0, 0, []
                candidates = []
                for name in ('research', 'research_runs', 'research_state', 'src', 'tests', 'configs', 'scripts'):
                    base = ROOT/name
                    if base.exists():
                        candidates += [p for p in base.rglob('*') if p.is_file() and not p.is_symlink()
                                       and '__pycache__' not in p.parts and R not in p.parents
                                       and p != Path(__file__).resolve()]
                candidates += [p for p in ROOT.iterdir() if p.is_file() and p.suffix in ('.md', '.toml', '.txt', '.yml', '.yaml')]
                with (R/'inventory.jsonl').open('xb') as inventory:
                    for p in sorted(set(candidates)):
                        with p.open('rb') as f:
                            first = f.read(8192)
                            if b'\0' in first:
                                continue  # Binary weights/arrays/images are never loaded or interpreted.
                            data = first+f.read()
                        if b'\0' in data:
                            continue
                        st = p.stat()
                        row = dict(path=str(p.relative_to(ROOT)), size=len(data), sha256=hashlib.sha256(data).hexdigest())
                        encoded = (json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n').encode()
                        output += len(encoded)
                        assert output+2*1024**2 <= CAP, 'Bounded inventory cap'
                        inventory.write(encoded)
                        files += 1
                        total += len(data)
                        for match in pattern.finditer(data):
                            matches.append(dict(path=row['path'], offset=match.start(), context=data[max(0, match.start()-70):match.end()+70].decode('utf-8', 'replace')))
                    inventory.flush()
                    os.fsync(inventory.fileno())
                publish('matches.json', dict(prior_matches=matches))
                assert not matches, 'A prospective scientific or derived context role already occurs in prior text'
                for p, h in plan['sources'].items():
                    assert sha(ROOT/p) == h, p
                publish('result.json', dict(status='reserved_unused', seeds=seeds, roles=roles,
                    recorded_text_files=files, recorded_text_bytes=total, inventory_sha256=sha(R/'inventory.jsonl'),
                    prior_matches=matches, excluded_prospective=[str(R), str(Path(__file__).resolve())],
                    binary_payloads_not_loaded=True, corpus_data_directory_not_scanned=True,
                    torch_model_optimizer_rng_seed_calls=0, source_pins_unchanged=True))
            publish('STOP', dict(status='complete_new_readout_scientific_and_context_role_census', no_replay=True))
            print(json.dumps(dict(status='complete', files=files, bytes=total, prior_matches=0)), flush=True)
        except BaseException as exc:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', no_retry=True, exception=repr(exc)))
            raise


if __name__ == '__main__':
    main()
