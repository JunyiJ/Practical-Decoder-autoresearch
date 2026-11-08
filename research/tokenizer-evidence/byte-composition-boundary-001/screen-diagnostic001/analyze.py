"""Descriptive saved-loss stratification of the first rate; no selection."""
import collections
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'research/tokenizer-evidence/byte-composition-boundary-001'
Q = BASE / 'quality001'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def frequency_bin(count):
    return '0' if count == 0 else '1-9' if count < 10 else '10-99' if count < 100 else '100-999' if count < 1000 else '1000+'

def length_bin(length):
    return '1' if length == 1 else '2-3' if length <= 3 else '4-7' if length <= 7 else '8+'

def main():
    # Bins are specified here before inspecting any stratified outcome.
    plan = json.loads((Q/'plan.json').read_text())
    expected = plan['data']
    inputs = [Q/'plan.json']
    saved = []
    reviews = []
    for slot in ('00','01'):
        folder = Q/'slots'/slot
        acceptance = json.loads((folder/'acceptance.json').read_text())
        assert acceptance['accepted'] and acceptance['slot'] == int(slot)
        for name, bound in acceptance['evidence'].items():
            assert sha(folder/name) == bound
        assert json.loads((folder/'guard.json').read_text())['status'] == 'finished'
        assert json.loads((folder/'STOP').read_text())['status'] == 'complete_original_scientific_slot'
        reviews.append(json.loads((folder/'review.json').read_text()))
        saved.append(json.loads((folder/'validation.json').read_text()))
        inputs += [folder/'acceptance.json', folder/'review.json', folder/'validation.json']
    assert [r['variant'] for r in reviews] == ['native','raw_bytes']
    assert all(r['learning_rate'] == .000625 and r['bundle'] == 'screen' and not r['quality_selected'] for r in reviews)
    assert reviews[0]['sampler_hashes'] == reviews[1]['sampler_hashes']
    assert saved[0]['report']['target'] == saved[1]['report']['target']
    for asset in plan['codebook_assets']:
        assert Path(asset['path']).stat().st_size == asset['size'] and sha(Path(asset['path'])) == asset['sha256']
    os.environ['TIKTOKEN_CACHE_DIR'] = str(Path(plan['codebook_assets'][0]['path']).parent)
    import tiktoken
    assert importlib.metadata.version('tiktoken') == expected['tokenizer']['version']
    enc = tiktoken.get_encoding('gpt2')
    assert hashlib.sha256(json.dumps(sorted((raw.hex(), rank) for raw,rank in enc._mergeable_ranks.items())).encode()).hexdigest() == expected['tokenizer']['vocabulary_sha256']
    corpus = Path(expected['path']);inputs.append(corpus)
    assert sha(corpus) == expected['corpus_sha256']
    text = corpus.read_text()
    train = enc.encode(text[:expected['raw_train_end']], disallowed_special=())
    validation = enc.encode(text[expected['raw_train_end']:expected['raw_val_end']], disallowed_special=())
    assert digest(train) == expected['train_ids_sha256'] and digest(validation) == expected['validation_ids_sha256']
    assert digest(validation[1:]) == saved[0]['report']['target']['target_ids_sha256']
    frequencies = collections.Counter(train)
    byte_lengths = {rank:len(raw) for raw,rank in enc._mergeable_ranks.items()}
    assert all(token in byte_lengths for token in validation)
    a,b = [s['per_target_losses'] for s in saved]
    assert len(a) == len(b) == len(validation)-1 == 34278
    assert all(math.isfinite(x) and x >= 0 for x in a+b)
    for losses, score in zip((a,b),saved):
        assert abs(math.fsum(losses)-score['report']['total_loss']) < 1e-8
    delta = [y-x for x,y in zip(a,b)]
    global_delta = math.fsum(delta)/len(delta)
    def stratify(ids, classifier):
        groups = collections.defaultdict(list)
        for index, token in enumerate(ids):groups[classifier(token)].append(index)
        rows = []
        for label, indices in groups.items():
            total = math.fsum(delta[i] for i in indices)
            rows.append(dict(bin=label, targets=len(indices),
                native_ce=math.fsum(a[i] for i in indices)/len(indices),
                raw_byte_ce=math.fsum(b[i] for i in indices)/len(indices),
                raw_minus_native_ce=total/len(indices), contribution_to_global_delta=total/len(delta),
                raw_lower_loss_fraction=sum(delta[i]<0 for i in indices)/len(indices)))
        assert sum(r['targets'] for r in rows)==len(delta)
        assert abs(math.fsum(r['contribution_to_global_delta'] for r in rows)-global_delta)<1e-12
        return rows
    result = dict(kind='post_aggregate_exploratory_first_rate_saved_loss_stratification',
        learning_rate=.000625, paired_screen_bundle=True, global_raw_minus_native_ce=global_delta,
        inputs=[dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for path in inputs],
        target_frequency=stratify(validation[1:],lambda token:frequency_bin(frequencies[token])),
        preceding_input_frequency=stratify(validation[:-1],lambda token:frequency_bin(frequencies[token])),
        target_byte_length=stratify(validation[1:],lambda token:length_bin(byte_lengths[token])),
        preceding_input_byte_length=stratify(validation[:-1],lambda token:length_bin(byte_lengths[token])),
        no_model_execution=True, no_new_losses=True, holdout_encoded=False, holdout_scored=False,
        limitations=['One rate and one paired screening bundle; no rate selection or fresh confirmation.',
                    'Exploratory after aggregate result was known; no confirmatory subgroup claim or confidence interval.',
                    'Frequency and length correlate with difficulty, context and token identity; no causal attribution.',
                    'Preceding input is the last token of the model context, not the whole context.',
                    'Keep the complete frozen scientific panel and adoption criteria unchanged.'])
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == '__main__':main()
