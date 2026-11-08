"""Exploratory frequency attribution of accepted saved losses; no model execution."""
import collections
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'research/tokenizer-evidence/output-anchored-tying-001'
OUT = BASE/'frequency-diagnostic001'


def read(path): return json.loads(path.read_text())
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def bin_for(count):
    return '0' if count == 0 else '1-9' if count < 10 else '10-99' if count < 100 else '100-999' if count < 1000 else '1000+'


def run():
    assert sha(BASE/'plan.json') == 'cc34e2b43349f2281b4a13303080b689bb34fb0ebe4060934a573dfc105921b6'
    plan = read(BASE/'plan.json'); review = read(BASE/'final-review.json')
    assert review['accepted'] and review['final_sha256'] == sha(BASE/'final.json')
    expected = read(BASE/'00/config.json')['data']
    import tiktoken
    assert importlib.metadata.version('tiktoken') == expected['tokenizer']['version']
    enc = tiktoken.get_encoding('gpt2')
    assert hashlib.sha256(json.dumps(sorted((t.hex(),r) for t,r in enc._mergeable_ranks.items())).encode()).hexdigest() == expected['tokenizer']['vocabulary_sha256']
    corpus = Path(expected['path']); assert sha(corpus) == expected['corpus_sha256']
    text = corpus.read_text(); assert len(text) == expected['raw_characters']
    train = enc.encode(text[:expected['raw_train_end']],disallowed_special=())
    validation = enc.encode(text[expected['raw_train_end']:expected['raw_val_end']],disallowed_special=())
    assert digest(train) == expected['train_ids_sha256'] and digest(validation) == expected['validation_ids_sha256']
    assert digest(validation[1:]) == expected['validation_target_ids_sha256'] and len(validation)-1 == 34278
    frequency = collections.Counter(train); panels = [dict() for _ in range(3)]
    for index, spec in enumerate(plan['specs']):
        path = BASE/f'{index:02d}'/'validation.json'
        assert sha(path) == review['receipts'][index]['validation_sha256']
        losses = read(path)['per_target_losses']
        assert len(losses) == 34278 and all(math.isfinite(x) and x >= 0 for x in losses)
        panels[index//3][spec['arm']] = losses
    output = {}
    for candidate, control in (('tied','native'),('tied','matched'),('matched','native')):
        deltas = [[a-b for a,b in zip(p[candidate],p[control],strict=True)] for p in panels]
        overall = math.fsum(math.fsum(row)/34278 for row in deltas)/3
        contrasts = dict(mean_delta=overall,views={})
        for name, ids in (('target_frequency',validation[1:]),('preceding_input_frequency',validation[:-1])):
            groups = collections.defaultdict(list)
            for i, token in enumerate(ids): groups[bin_for(frequency[token])].append(i)
            rows = []
            for label in ('0','1-9','10-99','100-999','1000+'):
                indices = groups[label]; assert indices
                totals = [math.fsum(delta[i] for i in indices) for delta in deltas]
                rows.append(dict(bin=label,targets=len(indices),
                    seed_deltas=[total/len(indices) for total in totals],
                    mean_delta=math.fsum(totals)/(3*len(indices)),
                    contribution_to_global_delta=math.fsum(totals)/(3*34278)))
            assert sum(r['targets'] for r in rows) == 34278
            assert abs(math.fsum(r['contribution_to_global_delta'] for r in rows)-overall) < 1e-12
            contrasts['views'][name] = rows
        output[candidate+'_vs_'+control] = contrasts
    assert sha(corpus) == expected['corpus_sha256']
    return dict(kind='post_aggregate_exploratory_saved_loss_frequency_analysis',comparisons=output,
        sources=dict(script=sha(Path(__file__)),plan=sha(BASE/'plan.json'),review=sha(BASE/'final-review.json')),
        training_tokens=len(train),training_types=len(frequency),validation_targets=34278,
        bins='Prior byte-input diagnostic bins: 0,1-9,10-99,100-999,1000+ training occurrences.',
        no_model_execution=True,no_new_losses=True,no_new_training=True,holdout_encoded=False,
        limitations=['Exploratory after aggregate results; no subgroup significance or causal gradient claim.',
                    'Frequency correlates with token identity and context; preceding input is only the final context token.',
                    'All three fresh seeds retained; this analysis changes no quality gate or scientific outcome.'])


if __name__ == '__main__':
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        OUT.mkdir()
        with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=1024**3):
            result = run()
        with (OUT/'result.json').open('x') as stream:
            json.dump(result,stream,sort_keys=True,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
