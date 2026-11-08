"""All-seed saved-loss frequency attribution, after the frozen panel closes."""
import collections
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import traceback

from research.output_anchored_tying_frequency_v1 import bin_for, digest, read, sha

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/tokenizer-evidence/byte-residual-readout-001/quality001'
OUT = ROOT/'research/tokenizer-evidence/byte-residual-readout-frequency-001'
PLAN_SHA = '5a5e11ee79335ff63cfe6761378cb6fc46fa7d4eacd3df6bf7bddb82ad3615a6'
LABELS = ('0', '1-9', '10-99', '100-999', '1000+')
CAP = 1024**2


def put(path, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    assert len(raw)+sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < CAP
    with path.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def run():
    plan = read(BASE/'quality-plan.json'); final = read(BASE/'final.json')
    review = read(BASE/'final-review.json')
    assert sha(BASE/'quality-plan.json') == PLAN_SHA
    assert read(BASE/'controller-complete.json')['completed']
    assert review['accepted'] and review['plan_sha256'] == PLAN_SHA
    assert review['final_sha256'] == sha(BASE/'final.json')
    analysis = read(BASE/'analysis-3-bundles001/result.json')
    assert analysis['bundles'] == 3 and analysis['plan_sha256'] == PLAN_SHA
    assert analysis['final_review']['sha256'] == sha(BASE/'final-review.json')
    assert final['completed'] and final['trajectories'] == 9 and final['updates'] == 10800
    expected = read(BASE/'00/config.json')['data']
    import tiktoken
    assert importlib.metadata.version('tiktoken') == expected['tokenizer']['version']
    enc = tiktoken.get_encoding('gpt2')
    import hashlib
    vocabulary = json.dumps(sorted((t.hex(), r) for t, r in enc._mergeable_ranks.items())).encode()
    assert hashlib.sha256(vocabulary).hexdigest() == expected['tokenizer']['vocabulary_sha256']
    corpus = Path(expected['path']); assert sha(corpus) == expected['corpus_sha256']
    text = corpus.read_text(); assert len(text) == expected['raw_characters']
    train = enc.encode(text[:expected['raw_train_end']], disallowed_special=())
    validation = enc.encode(text[expected['raw_train_end']:expected['raw_val_end']], disallowed_special=())
    assert digest(train) == expected['train_ids_sha256']
    assert digest(validation) == expected['validation_ids_sha256']
    assert digest(validation[1:]) == expected['validation_target_ids_sha256']
    assert len(train) == 267688 and len(validation) == 34279
    frequency = collections.Counter(train); panels = [dict() for _ in range(3)]
    receipts = {}
    for index, spec in enumerate(plan['specs']):
        folder = BASE/f'{index:02d}'
        accepted = read(folder/'acceptance.json'); result = read(folder/'result.json')
        assert accepted['accepted'] and accepted['result_sha256'] == sha(folder/'result.json')
        assert result['spec'] == spec and result['steps'] == 1200 and result['averaging_snapshots'] == 41
        proof = review['receipts'][index]; assert proof['slot'] == index
        assert proof['acceptance_sha256'] == sha(folder/'acceptance.json')
        assert proof['validation_sha256'] == sha(folder/'validation-mps.json')
        score = read(folder/'validation-mps.json'); losses = score['per_target_losses']
        assert len(losses) == score['target_count'] == 34278 and not score['holdout_scored']
        assert all(math.isfinite(x) and x >= 0 for x in losses)
        assert math.fsum(losses)/34278 == score['ce'] == result['ce']
        panels[index//3][spec['arm']] = losses
        for name in ('acceptance.json', 'result.json', 'validation-mps.json'):
            receipts[str((folder/name).relative_to(ROOT))] = sha(folder/name)
    output = {}
    for candidate, control in (('residual_tanh', 'byte'), ('residual_linear', 'byte'),
                               ('residual_tanh', 'residual_linear')):
        deltas = [[a-b for a, b in zip(p[candidate], p[control], strict=True)] for p in panels]
        overall = math.fsum(math.fsum(row)/34278 for row in deltas)/3
        assert abs(overall-(final['means'][candidate]-final['means'][control])) < 1e-12
        views = {}
        for name, ids in (('target_frequency', validation[1:]),
                          ('preceding_input_frequency', validation[:-1])):
            groups = collections.defaultdict(list)
            for i, token in enumerate(ids): groups[bin_for(frequency[token])].append(i)
            rows = []
            for label in LABELS:
                indices = groups[label]; assert indices
                totals = [math.fsum(delta[i] for i in indices) for delta in deltas]
                rows.append(dict(bin=label, targets=len(indices),
                    seed_deltas=[total/len(indices) for total in totals],
                    mean_delta=math.fsum(totals)/(3*len(indices)),
                    contribution_to_global_delta=math.fsum(totals)/(3*34278)))
            assert sum(r['targets'] for r in rows) == 34278
            assert abs(math.fsum(r['contribution_to_global_delta'] for r in rows)-overall) < 1e-12
            views[name] = rows
        output[candidate+'_vs_'+control] = dict(mean_delta=overall, views=views)
    assert sha(corpus) == expected['corpus_sha256']
    assert all(sha(ROOT/path) == value for path, value in receipts.items())
    return dict(completed=True, comparisons=output, input_receipts=receipts,
        training_tokens=len(train), training_types=len(frequency), validation_targets=34278,
        no_model_execution=True, no_new_losses=True, no_new_training=True, holdout_encoded=False,
        scope='Exploratory all-three-seed saved-loss attribution with prior fixed bins. No subgroup significance, causal effect, new selection, checkpoint reload, or change to quality criteria.')


if __name__ == '__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert read(BASE/'controller-complete.json')['completed']
        assert read(BASE/'final-review.json')['accepted']
        assert (BASE/'analysis-3-bundles001/result.json').is_file()
        assert shutil.disk_usage(ROOT).free >= 10*1024**3+CAP
        OUT.mkdir()
        paths = [Path(__file__), ROOT/'research/BYTE_RESIDUAL_FREQUENCY_PROTOCOL.md',
                 ROOT/'research/output_anchored_tying_frequency_v1.py',
                 ROOT/'src/research/cpu_prerequisite_guard.py',
                 *(BASE/name for name in ('quality-plan.json', 'final.json', 'final-review.json',
                                          'controller-complete.json', 'analysis-3-bundles001/result.json'))]
        pins = {str(p.relative_to(ROOT)): sha(p) for p in paths}
        put(OUT/'attempt.json', dict(input_sources=pins, output_cap_bytes=CAP))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json', ROOT, timeout_seconds=120,
                                     rss_limit_bytes=1024**3, reserve_bytes=10*1024**3):
                result = run()
                assert all(sha(ROOT/path) == value for path, value in pins.items())
            result['attempt_sha256'] = sha(OUT/'attempt.json')
            result['guard_sha256'] = sha(OUT/'guard.json')
            put(OUT/'result.json', result)
            print(json.dumps(result['comparisons'], sort_keys=True), flush=True)
        except BaseException:
            put(OUT/'failure.json', dict(traceback=traceback.format_exc())); raise
