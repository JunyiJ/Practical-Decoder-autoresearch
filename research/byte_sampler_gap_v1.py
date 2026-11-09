"""Exact target-arrival probabilities for the incumbent paired-block sampler."""
import collections
import fcntl
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import traceback

from research.output_anchored_tying_frequency_v1 import bin_for, digest, read, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/tokenizer-evidence/byte-residual-readout-001/quality001'
OUT = ROOT/'research/tokenizer-evidence/byte-sampler-gap-001'
CAP = 1024**2
LABELS = ('0', '1-9', '10-99', '100-999', '1000+')


def intervals(sequence, block):
    n = len(sequence)-block
    assert n > 0
    positions = collections.defaultdict(list)
    for j, token in enumerate(sequence): positions[token].append(j)
    result = {}
    for token, indices in positions.items():
        covered = total = 0; right = -1
        for j in indices:
            lo, hi = max(0, j-block), min(n-1, j-1)
            if hi < lo: continue
            total += hi-lo+1
            covered += max(0, hi-max(lo, right+1)+1)
            right = max(right, hi)
        assert 0 <= covered <= n and covered <= total
        result[token] = (covered, total)
    return result


def verify_toy():
    cases = list(itertools.product(range(2), repeat=7))+[tuple(range(7))]
    for seq in cases:
        exact = intervals(seq, 3); presence = collections.Counter(); counts = collections.Counter()
        for a, b in itertools.product(range(4), repeat=2):
            targets = seq[a+1:a+4]+seq[b+1:b+4]
            presence.update(set(targets)); counts.update(targets)
        for token, (covered, total) in exact.items():
            assert presence[token] == 8*covered-covered**2
            assert counts[token] == 8*total
    return len(cases)


def summary(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values)) if values else None


def run():
    toy_cases = verify_toy()
    expected = read(BASE/'00/config.json')['data']
    import tiktoken
    assert importlib.metadata.version('tiktoken') == expected['tokenizer']['version']
    enc = tiktoken.get_encoding('gpt2')
    vocabulary = json.dumps(sorted((t.hex(), r) for t, r in enc._mergeable_ranks.items())).encode()
    assert hashlib.sha256(vocabulary).hexdigest() == expected['tokenizer']['vocabulary_sha256']
    corpus = Path(expected['path']); assert sha(corpus) == expected['corpus_sha256']
    text = corpus.read_text(); assert len(text) == expected['raw_characters']
    train = enc.encode(text[:expected['raw_train_end']], disallowed_special=())
    val = enc.encode(text[expected['raw_train_end']:expected['raw_val_end']], disallowed_special=())
    assert digest(train) == expected['train_ids_sha256']
    assert digest(val) == expected['validation_ids_sha256']
    assert digest(val[1:]) == expected['validation_target_ids_sha256']
    assert len(train) == 267688 and len(val) == 34279 and enc.n_vocab == 50257
    freq, vf = collections.Counter(train), collections.Counter(val[1:])
    n = len(train)-512; exact = intervals(train, 512)
    expected_total = math.fsum(2*total/n for covered, total in exact.values())
    assert abs(expected_total-1024) < 1e-10
    rows = []
    for label in LABELS:
        ids = [i for i in range(enc.n_vocab) if bin_for(freq[i]) == label]
        presence = {i: 2*(exact.get(i, (0, 0))[0]/n)-(exact.get(i, (0, 0))[0]/n)**2 for i in ids}
        positive = {i: p for i, p in presence.items() if p > 0}
        assert all(0 <= p <= 1 for p in presence.values())
        beta_rows = {}
        for beta in (.999, .95):
            kappas = {i: (1-beta)/p for i, p in positive.items()}
            high = [i for i, k in kappas.items() if k > 1]
            beta_rows[str(beta)] = dict(kappa=summary(list(kappas.values())),
                types_above_one=len(high), validation_targets_above_one=sum(vf[i] for i in high))
        rows.append(dict(bin=label, types=len(ids), validation_targets=sum(vf[i] for i in ids),
            zero_presence_types=len(ids)-len(positive),
            zero_presence_validation_targets=sum(vf[i] for i, p in presence.items() if p == 0),
            mean_gap_updates=summary([1/p for p in positive.values()]), beta2=beta_rows,
            expected_types_never_targeted_1200=math.fsum((1-p)**1200 for p in presence.values()),
            expected_validation_target_mass_never_trained_1200=math.fsum(vf[i]*(1-p)**1200 for i, p in presence.items())/34278))
    assert sum(r['types'] for r in rows) == enc.n_vocab
    assert sum(r['validation_targets'] for r in rows) == 34278
    assert sha(corpus) == expected['corpus_sha256']
    return dict(completed=True, toy_exhaustive_cases=toy_cases, rows=rows, starts=n,
        expected_targets_per_update=expected_total, train_tokens=len(train),
        training_types=len(freq), validation_targets=34278,
        data_identity={k: expected[k] for k in ('corpus_sha256', 'train_ids_sha256',
            'validation_ids_sha256', 'validation_target_ids_sha256')},
        no_model_execution=True, no_random_draws=True, no_new_losses=True, holdout_encoded=False,
        scope='Sampler expectations only; no actual old seed replay, probability-bias prediction or quality claim.')


def put(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    assert len(raw)+sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < CAP
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


if __name__ == '__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(BASE/'controller-complete.json')['completed']
        review = read(BASE/'final-review.json')
        assert review['accepted'] and review['final_sha256'] == sha(BASE/'final.json')
        assert shutil.disk_usage(ROOT).free >= 10*1024**3+CAP
        OUT.mkdir()
        paths = [Path(__file__), ROOT/'research/BYTE_SAMPLER_GAP_PROTOCOL.md',
            ROOT/'research/output_anchored_tying_frequency_v1.py',
            ROOT/'src/research/cpu_prerequisite_guard.py',
            ROOT/'research/block_norm_v1/runtime/data/loaders.py',
            ROOT/'research/byte_input_model_v6/optimizer.py',
            ROOT/'research/tokenizer-evidence/byte-residual-readout-frequency-001/result.json',
            *(BASE/name for name in ('00/config.json', 'final.json', 'final-review.json', 'controller-complete.json'))]
        pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        put(OUT/'attempt.json', dict(input_sources=pins, output_cap_bytes=CAP))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120,
                    rss_limit_bytes=1024**3, reserve_bytes=10*1024**3):
                result = run()
                assert all(sha(ROOT/path) == value for path, value in pins.items())
            result.update(attempt_sha256=sha(OUT/'attempt.json'), guard_sha256=sha(OUT/'guard.json'))
            put(OUT/'result.json', result)
            print(json.dumps(result, sort_keys=True), flush=True)
        except BaseException:
            put(OUT/'failure.json', dict(traceback=traceback.format_exc())); raise
