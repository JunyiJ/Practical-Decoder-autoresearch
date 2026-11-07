"""New six-role serialized-resume census; no Torch/model/data-loader imports."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
CAP = 32*1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name, value):
    raw = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir())+len(raw) < CAP
    with (R/name).open('xb') as f:
        f.write(raw); f.flush(); os.fsync(f.fileno())


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--plan-sha', required=True); arg = parser.parse_args()
    assert sha(R/'plan.json') == arg.plan_sha and set(p.name for p in R.iterdir()) == {'plan.json', 'source-review.json'}
    plan = json.loads((R/'plan.json').read_text())
    assert len(plan['sources']) == 1191 and 'torch' not in sys.modules
    for p, h in plan['sources'].items():
        assert sha(ROOT/p) == h, p
    base, private = plan['roles']['initializer'], plan['roles']['private_ungated']
    assert set(plan['roles']) == {'initializer', 'private_ungated', 'train', 'eval', 'global_cpu', 'global_mps'}
    assert len(set(plan['roles'].values())) == 6 and all(type(v) is int and 0 <= v < 2**63 for v in plan['roles'].values())
    assert int.from_bytes(hashlib.sha256(('relu_squared_model_v1:ungated:'+str(base)).encode()).digest()[:8], 'big') % 2**63 == private
    assert shutil.disk_usage(ROOT).free >= plan['complete_next_chain_target_bytes']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        publish('attempt.json', dict(pid=os.getpid(), plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25):
                # Numeric boundaries also catch a role embedded in a labelled filename/string.
                pattern = re.compile(rb'(?<![0-9])(?:'+b'|'.join(str(x).encode() for x in plan['roles'].values())+rb')(?![0-9])')
                candidates = []
                for name in ('research', 'research_runs', 'research_state', 'src', 'tests', 'configs', 'scripts'):
                    folder = ROOT/name
                    if folder.exists():
                        candidates.extend(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()
                                          and '__pycache__' not in p.parts and R not in p.parents
                                          and p != Path(__file__).resolve())
                candidates.extend(p for p in ROOT.iterdir() if p.is_file() and not p.is_symlink()
                                  and p.suffix in ('.md', '.toml', '.txt', '.yml', '.yaml'))
                files = total = inventory_bytes = binary_skips = 0
                matches = []
                with (R/'inventory.jsonl').open('xb') as inventory:
                    for p in sorted(set(candidates)):
                        h = hashlib.sha256(); length = 0; tail = b''; found = []; binary = False; processed = 0
                        with p.open('rb') as f:
                            first = f.read(8192)
                            if b'\0' in first:
                                binary_skips += 1
                                continue
                            chunk = first
                            while chunk:
                                if b'\0' in chunk:
                                    binary = True
                                    break
                                h.update(chunk); length += len(chunk)
                                scan = tail+chunk
                                next_chunk = f.read(1024**2)
                                boundary = length if not next_chunk else length-32
                                for match in pattern.finditer(scan):
                                    offset = length-len(chunk)-len(tail)+match.start()
                                    end = length-len(chunk)-len(tail)+match.end()
                                    if processed < end <= boundary:
                                        found.append(offset)
                                processed = boundary
                                tail = scan[-64:]
                                chunk = next_chunk
                        if binary:
                            binary_skips += 1
                            continue
                        row = dict(path=str(p.relative_to(ROOT)), size=length, sha256=h.hexdigest())
                        encoded = (json.dumps(row, sort_keys=True, separators=(',', ':'))+'\n').encode()
                        inventory_bytes += len(encoded)
                        assert inventory_bytes+2*1024**2 < CAP, 'Complete census metadata cap'
                        inventory.write(encoded); files += 1; total += length
                        matches.extend(dict(path=row['path'], offset=offset) for offset in found)
                    inventory.flush(); os.fsync(inventory.fileno())
                if matches:
                    publish('prior-matches.json', dict(matches=matches))
                    raise ValueError('Proposed role already occurs in prior recorded text')
                for p, h in plan['sources'].items():
                    assert sha(ROOT/p) == h, p
                assert 'torch' not in sys.modules
                publish('result.json', dict(status='new_ffn_resume_roles_unused_before_construction',
                    roles=plan['roles'], recorded_text_files=files, recorded_text_bytes=total,
                    inventory_sha256=sha(R/'inventory.jsonl'), prior_matches=matches, binary_skips=binary_skips,
                    excluded_prospective=[str(R), str(Path(__file__).resolve())],
                    source_pins_unchanged=True, torch_model_optimizer_rng_seed_calls=0,
                    corpus_data_directory_not_scanned=True, binary_payloads_not_loaded_as_tensors=True))
            publish('STOP', dict(status='complete_new_ffn_resume_role_census', no_replay=True, no_retry=True))
            print(json.dumps(dict(status='complete', text_files=files, text_bytes=total, prior_matches=0)), flush=True)
        except BaseException as exc:
            if not (R/'STOP').exists():
                publish('STOP', dict(status='closed_failed', error=repr(exc), no_retry=True))
            raise


if __name__ == '__main__':
    main()
