"""One fresh, guarded train-only provenance artifact; no model or launch.

The output's meta directory can later be copied byte-for-byte into a new
campaign. Binding still requires the same source/runtime and corpus identities.
Existing output is refused, including failed partial attempts.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.protocol import digest, file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_wide_hybrid_config import rank_config
from src.research.tokenizer_wide_hybrid_driver import CORPUS_SHA256
from src.research.tokenizer_wide_hybrid_timing import token_digest
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder


def publish(path, value):
    with path.open('x') as stream:
        stream.write(json_encoder().encode(value)+'\n')
    return dict(path='meta/'+path.name, sha256=file_digest(path))


def prepare(output, cache):
    cache = Path(cache).resolve()
    # Refuse implicit network downloads or mutation of an invalid cache.
    for name, expected in (
        ('vocab.bpe', '1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5'),
        ('encoder.json', '196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783')):
        url = 'https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/'+name
        assert file_digest(cache/hashlib.sha1(url.encode()).hexdigest()) == expected
    os.environ['TIKTOKEN_CACHE_DIR'] = str(cache)
    output = Path(output).absolute()
    if output != output.resolve():
        raise ValueError('Canonical fresh output required')
    output.mkdir(exist_ok=False)
    meta = output/'meta'
    meta.mkdir()
    with CpuPrerequisiteGuard(output/'guard.json', ROOT, timeout_seconds=180):
        import tiktoken
        source, runtime = source_fingerprint(), runtime_identity()
        cfg = rank_config(output/'config-only', 'hybrid', device='cpu')
        corpus = Path(cfg['data']['path']).resolve()
        assert file_digest(corpus) == CORPUS_SHA256
        assert cfg['data']['tokenizer'] == 'gpt2'
        assert cfg['data']['train_fraction'] == .8 and cfg['data']['val_fraction'] == .1
        # Read for integrity and boundary location; tokenize only the train slice.
        # No loader is instantiated: the normal loader encodes every split.
        text = corpus.read_text(encoding='utf-8')
        train_end = int(len(text)*.8)
        train = text[:train_end]
        del text
        enc = tiktoken.get_encoding('gpt2')
        ranks = [(token.hex(), rank) for token, rank in enc._mergeable_ranks.items()]
        tokenizer = dict(name='gpt2', implementation='tiktoken',
            version=importlib.metadata.version('tiktoken'),
            vocabulary_sha256=hashlib.sha256(json.dumps(sorted(ranks)).encode()).hexdigest(),
            pattern=enc._pat_str, special_tokens=enc._special_tokens,
            special_token_policy='ordinary_text', split_unit='raw_text', vocab_size=enc.n_vocab)
        assert tokenizer == cfg['model']['output_bias_prior']['tokenizer']
        tokens = enc.encode(train, disallowed_special=())
        assert tokens == enc.encode_ordinary(train)
        assert enc.decode(tokens) == train
        ids = tokens[:96]
        sha = token_digest(ids)
        context = dict(source_sha256=digest(source), runtime_sha256=digest(runtime),
                       corpus_sha256=CORPUS_SHA256)
        trace_ref = publish(meta/'training-input-trace.json', dict(
            kind='wide_hybrid_train_only_encoding_trace_v2', **context,
            source=source, runtime=runtime, script_sha256=file_digest(Path(__file__)),
            corpus_path=str(corpus), train_fraction=.8, val_fraction=.1,
            train_end_character=train_end, train_token_count=len(tokens),
            train_utf8_sha256=hashlib.sha256(train.encode()).hexdigest(),
            tokenizer=tokenizer, encode_ordinary_equal=True, full_train_roundtrip_equal=True,
            validation_encoded=False, test_encoded=False, model_loaded=False,
            checkpoint_loaded=False, token_selection='first_96_train_tokens'))
        ids_ref = publish(meta/'training-input-ids.json', dict(split='train',
            corpus_sha256=CORPUS_SHA256, tokenizer_sha256=digest(tokenizer), ids=ids))
        assert file_digest(corpus) == CORPUS_SHA256
        assert source_fingerprint() == source and runtime_identity() == runtime
        report = dict(kind='wide_hybrid_training_input_v2', status='passed', **context,
            tokenizer_sha256=digest(tokenizer), training_token_sha256=sha,
            split='train', token_count=96, holdout_consumed=False,
            input_ids_ref=ids_ref,
            artifacts={ref['path']:ref['sha256'] for ref in (trace_ref, ids_ref)})
        ref = publish(meta/'training-input.json', report)
    assert json.loads((output/'guard.json').read_text())['status'] == 'finished'
    print(json.dumps(dict(input_ref=ref, token_sha256=sha, files=3,
        metadata_bytes=sum(p.stat().st_size for p in meta.iterdir()),
        plan_frozen=False, training_launched=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', required=True)
    args = parser.parse_args()
    prepare(args.output, args.cache)
