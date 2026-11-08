"""Training-only byte/fragment accounting for the existing screening codebooks.

Widths 1..6 match the six composer filters. Count raw content bytes only:
exclude boundary/padding/special markers. Cross-token sharing means a fragment
occurs in at least two distinct token IDs, not merely repeated training tokens.
No model execution, numerical role assignment or scientific selection.
"""
from array import array
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / 'research/tokenizer-evidence/byte-composition-boundary-001'
Q = BASE / 'quality001'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def summarize(sequences, weights):
    byte_counts = Counter()
    for token, weight in weights.items():
        for byte in sequences[token]:
            byte_counts[byte] += weight
    total_bytes = sum(byte_counts.values())
    assert total_bytes == sum(len(sequences[t])*w for t,w in weights.items())
    byte_probabilities = [count/total_bytes for count in byte_counts.values()]
    byte_summary = dict(total_bytes=total_bytes, distinct_bytes=len(byte_counts),
        entropy_bits=-math.fsum(p*math.log2(p) for p in byte_probabilities),
        top10_mass=sum(n for _,n in byte_counts.most_common(10))/total_bytes,
        counts={str(b):byte_counts[b] for b in range(256)})
    windows = []
    for width in range(1,7):
        document_counts = Counter()
        occurrence_counts = Counter()
        for token, weight in weights.items():
            sequence = sequences[token]
            fragments = [sequence[i:i+width] for i in range(len(sequence)-width+1)]
            document_counts.update(set(fragments))
            for fragment in fragments:
                occurrence_counts[fragment] += weight
        occurrences = sum(occurrence_counts.values())
        incidences = sum(document_counts.values())
        shared = {fragment for fragment,count in document_counts.items() if count >= 2}
        assert occurrences == sum(max(0,len(sequences[t])-width+1)*w for t,w in weights.items())
        assert incidences <= sum(max(0,len(sequences[t])-width+1) for t in weights)
        windows.append(dict(width=width, distinct_fragments=len(document_counts),
            shared_fragments=len(shared), distinct_token_fragment_incidences=incidences,
            cross_token_shared_incidence_fraction=sum(document_counts[f] for f in shared)/incidences,
            weighted_fragment_occurrences=occurrences,
            weighted_cross_token_shared_fraction=sum(occurrence_counts[f] for f in shared)/occurrences))
    return dict(token_types=len(weights), token_occurrences=sum(weights.values()),
        byte_distribution=byte_summary, fragments=windows)


def main():
    started = time.perf_counter()
    plan = json.loads((Q/'plan.json').read_text())
    source = ROOT/'research/byte_input_model_v3/codebook.py'
    source_record = next(r for r in plan['sources'] if r['path']==str(source))
    assert sha(source)==source_record['sha256']
    # Load the pure standard-library builder without importing its model package.
    spec = importlib.util.spec_from_file_location('control_structure_codebook',source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    inputs = [Q/'plan.json',source,Path(__file__).resolve()]
    assets = [Path(a['path']) for a in plan['codebook_assets']]
    for p,a in zip(assets,plan['codebook_assets']):
        assert p.stat().st_size==a['size'] and sha(p)==a['sha256']
    inputs += assets
    os.environ['TIKTOKEN_CACHE_DIR'] = str(assets[0].parent)
    import tiktoken
    data = plan['data']
    assert importlib.metadata.version('tiktoken')==data['tokenizer']['version']
    enc = tiktoken.get_encoding('gpt2')
    assert hashlib.sha256(json.dumps(sorted((raw.hex(), rank) for raw,rank in enc._mergeable_ranks.items())).encode()).hexdigest()==data['tokenizer']['vocabulary_sha256']
    corpus = Path(data['path']); inputs.append(corpus)
    assert sha(corpus)==data['corpus_sha256']
    train = enc.encode(corpus.read_text()[:data['raw_train_end']],disallowed_special=())
    assert digest(train)==data['train_ids_sha256'] and 50256 not in train
    counts = Counter(train)
    views = dict(full_regular_vocabulary={t:1 for t in range(50256)},
        training_token_types={t:1 for t in counts}, training_occurrences=dict(counts))
    results = {}; identities = {}; lengths_reference = None
    for slot,variant in [('01','raw_bytes'),('02','nonlexical_codes')]:
        folder = Q/'slots'/slot
        acceptance = json.loads((folder/'acceptance.json').read_text())
        assert acceptance['accepted'] and acceptance['slot']==int(slot)
        assert sha(folder/'artifacts.json')==acceptance['evidence']['artifacts.json']
        original = json.loads((folder/'artifacts.json').read_text())['identity']
        assert original['roles']==plan['bundles']['screen']
        inputs += [folder/'acceptance.json',folder/'artifacts.json']
        key = None if variant=='raw_bytes' else plan['bundles']['screen']['nonlexical_key']
        book = module.build_codebook(assets[0].read_bytes(),assets[1].read_bytes(),variant,key)
        assert list(book.identity)==original['model'][3]
        identities[variant] = list(book.identity)
        packed = array('H');packed.frombytes(book.codes)
        lengths = array('H');lengths.frombytes(book.lengths)
        if sys.byteorder!='little':packed.byteswap();lengths.byteswap()
        if lengths_reference is None:lengths_reference = list(lengths)
        else:assert list(lengths)==lengths_reference
        sequences = [bytes(list(packed[t*130+1:t*130+lengths[t]-1])) for t in range(50256)]
        assert len(set(sequences))==50256
        assert all(len(sequence)==lengths[t]-2 for t,sequence in enumerate(sequences))
        if variant=='raw_bytes':
            assert all(sequences[token]==raw for raw,token in enc._mergeable_ranks.items())
        results[variant] = {name:summarize(sequences,weights) for name,weights in views.items()}
        del book,packed,lengths,sequences
    for name in views:
        raw,control = [results[v][name] for v in ('raw_bytes','nonlexical_codes')]
        assert raw['byte_distribution']['total_bytes']==control['byte_distribution']['total_bytes']
        assert [r['weighted_fragment_occurrences'] for r in raw['fragments']]==[r['weighted_fragment_occurrences'] for r in control['fragments']]
    result = dict(kind='existing_screen_codebook_training_only_structure_analysis',
        elapsed_seconds=time.perf_counter()-started, codebook_identities=identities,
        inputs=[dict(path=str(p),sha256=sha(p)) for p in inputs], views=results,
        no_model_execution=True,no_new_random_draws=True,no_training_updates=True,
        validation_encoded=False,holdout_encoded=False,holdout_scored=False,
        limitations=['One existing screening code key; no fresh confirmation keys examined.',
            'Content fragments exclude boundary and padding markers; this is not activation or gradient accounting.',
            'Training-occurrence weights count the corpus split, not the realized sampled stream.',
            'Pattern availability does not establish learned use or explain a causal loss difference.',
            'No changed control, quality endpoint, rate, seed or frozen acceptance rule.'])
    print(json.dumps(result,indent=2,sort_keys=True,allow_nan=False))


if __name__=='__main__':main()
