# General English pretraining corpus v1

Prepared on 2026-11-08 for the user's M4 Mac mini (16 GB). This is a new,
quality-filtered corpus of existing writing, not newly generated synthetic text.
It is independent of the historical Shakespeare campaigns. No model training or
quality-improvement claim follows from preparing the data.

## Data and use

Local payload directory: `data/raw/rsi-general-10m-v1/` (gitignored).
See `manifest.json` in this report directory for the final measured document,
token, source, rejection, file-size and hash counts. `verification.json` records
the complete-data checks. The corpus uses the existing GPT-2 vocabulary, with
little-endian uint16 token IDs and an end-of-text token after each document.
The requested budgets are 10 million training tokens and 100,000 tokens each
for validation and final test; whole documents cause small overshoots.

- `train.bin`, `val.bin`, `test.bin`: token IDs, not PyTorch pickles.
- `*.jsonl.gz`: normalized text, source URLs/IDs, pinned-shard row locations,
  quality scores where available, document hashes and split identity.
- `*.index.jsonl`: document offsets and lengths in the token files.
- `*-card.md`: dataset cards from the exact upstream revisions.
- `builder-source.py`: exact builder source used for this artifact.
- `manifest.json`: provenance, versions, counts, and SHA-256 payload inventory.

The disk-backed loader reads only the requested windows into tensors and samples
uniformly over eligible windows within documents. It does not concatenate
unrelated documents into a training example. The end-of-text token can be a
prediction target. Validation has its own RNG; the training interface does not
open or expose the final test split.

```python
from src.data.pretraining_corpus import PretrainingCorpus

data = PretrainingCorpus(
    'data/raw/rsi-general-10m-v1',
    batch_size=8, block_size=128, device='mps', seed=42,
)
x, y = data.get_batch('train')  # torch.int64, shape [8, 128]
vx, vy = data.get_batch('val')
```

This is a standalone loader. Frozen training contexts still enforce their
original corpus hashes and must not be repointed to this dataset. A new training
entry point must explicitly consume this loader and record this manifest hash.

Rebuild to a **new** directory (network access required):

```sh
.venv/bin/python research/build_pretraining_corpus_v1.py --output data/raw/rsi-general-10m-rebuild
.venv/bin/python -m research.verify_pretraining_corpus_v1 data/raw/rsi-general-10m-rebuild
.venv/bin/python -m pytest -q tests/test_pretraining_corpus.py
```

Source revisions, random selection and token contents are reproducible. Timing,
filesystem paths, gzip timestamps and environment-dependent manifest fields are
not promised to produce byte-identical archives across rebuilds.

## Sources and quality policy

The mix targets 80% FineWeb-Edu and 20% English Wikipedia by tokens within each
split. This is a pragmatic starting mix for educational and encyclopedic prose,
not an experimentally optimal recipe or a balanced sample of all English.

- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
  revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`, `sample/10BT/`.
  Retain upstream rounded educational score >=3, English language label and
  language score >=0.8. Four deterministic shuffled shards, with shuffled row
  groups and rows, avoid taking only the beginning of one archive.
- [Wikimedia Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia),
  revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`, `20231101.en/`.
  Four deterministic shuffled shards and row groups; full article documents.

Normalization retains paragraph breaks, applies Unicode NFC, normalizes line
endings and spaces, and removes control characters. Conservative heuristic
filters reject excessive extraction markup, replacement characters, very short
or long text, low prose content, repetitive lines/trigrams, and extreme word
repetition. Retained documents contain 256–8,192 content tokens. Long documents
are rejected rather than silently truncated.

Global deduplication rejects normalized exact text duplicates, canonical URL
duplicates and similar-length SimHash matches at Hamming distance <=3. Direct
Wikipedia/Wikimedia domains are excluded from the web portion. A web hostname
cap of roughly 1% of the training budget limits individual-site concentration.
This cap uses hostnames, not registered parent domains; subdomains can therefore
have separate caps. Wikipedia is intentionally exempt.

Splits are deterministic hashes of canonical document URLs (98/1/1 before token
quota stopping). Query strings and fragments do not create separate documents.
These are document-separated, same-mixture evaluations, not unseen-domain tests.

An additional passage filter queries all 64-token windows of each candidate
against sampled training shingles and all evaluation shingles. Training stores
every 32nd window; evaluation stores every window. It rejects cross-split copied
passages of at least 95 exact tokens and many shorter passages, independent of
document arrival order. The independent verifier also checks all evaluation
windows against stride-32 training windows. Shorter overlap, paraphrases and
semantic duplication may remain. No benchmark-specific decontamination is claimed.

## Evidence and limits

Local basis: `research/figures/gqa-overfitting-2026-11-01/analysis.json` shows
falling training loss and rising validation loss on the old corpus. That corpus
has 267,688 training tokens. This new pool is about 37 times larger; it supports
fresh-text sampling without increasing model memory proportionally.

External basis: [The FineWeb Datasets, v2](https://arxiv.org/html/2406.17557v2),
§4, supports educational filtering at much larger training scales; §§4.1–4.2
also show topic and domain biases and weaker fit on some broad web domains.
Those published gains are not gains demonstrated for this local model. The
80/20 mix and extra local heuristics have not been ablated.

Mechanistic hypothesis for a future comparison: repeated exposure to a tiny
corpus contributed to overfitting and the advantage of parameter-sharing input
representations. A diverse fresh-text pool should change those dynamics. Compare
native and byte-composed inputs with matched data streams, tuning budgets and
held-out documents; report both equal-token and equal-compute results. A similar
or larger byte advantage would support its usefulness beyond the old sparse
corpus; a shrinking/reversed advantage would narrow the earlier claim. Neither
outcome alone establishes transfer to billion-parameter models.

Preparation only: a 1 GiB disk-output forecast plus a 2 GiB reserve was checked
before building. No full parquet shards are cached on disk. Final payload sizes
are measured in the manifest. Any future training panel needs its own complete
checkpoint/serialization/output budget and explicit recipe; none was launched.

Manual inspection of training excerpts found varied educational, historical,
scientific, cultural and encyclopedic writing, but also opinion, promotional
framing and dated claims. This is **quality-filtered pretraining text, not
fact-checked reference material**. It is English-centric, education-heavy, and
not a code, multilingual, instruction-following or comprehensive LLM corpus.
The small validation/test samples support an initial screen, not strong claims
about general language ability. Do not repeatedly use the final test for tuning.

Licenses and attribution: the FineWeb-Edu card declares ODC-By-1.0 and Common
Crawl terms; this does not replace underlying content rights. The pinned
Wikipedia card lists CC-BY-SA-3.0/GFDL with its stated exceptions. Original URLs
and source cards are retained. Raw text payloads are not committed or uploaded
to the Git repository.

## Verification

The verifier checks every document's tokenization, all file hashes/sizes, token
range, end-of-text boundaries, index coverage, source counts, deterministic
splits, global normalized-text/URL uniqueness, retained SimHash thresholds and
cross-split passage overlap. Tests cover shifted copied passages in either
arrival order, URL variants, duplicate handling, deterministic sampling,
document boundaries, independent validation RNG, malformed payload lengths and
unavailable test access. CPU and MPS batch delivery are checked separately.

The next executable research step is a fresh, separately budgeted native-vs-byte
screen using this corpus; historical studies remain closed.
