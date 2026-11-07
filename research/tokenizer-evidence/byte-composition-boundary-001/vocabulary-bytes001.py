"""One original, guarded, read-only public GPT2 vocabulary metadata inspection.

No tokenizer/package imports or functions, corpus, model, RNG or network access.
"""
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import time
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/tokenizer-evidence/byte-composition-boundary-001'
OUT = BASE / 'vocabulary-bytes001'
LIMITS = {'timeout_seconds': 120, 'rss_limit_bytes': 4294967296,
          'reserve_bytes': 8589934592, 'output_bytes': 2097152,
          'input_bytes': 8388608, 'entry_bytes': 10739515392}
VISIBLE_RANGES = ((33, 127), (161, 173), (174, 256))
N_REGULAR = 50256
N_TOTAL = 50257
N_MERGES = 50000
EXPECTED_NATIVE_HASH = 'bdc599f0a294cff99d60cf83aed3870857b30bc0d1d06164cae137831b55257c'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def write_new(path, value):
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    require(sum(p.stat().st_size for p in OUT.iterdir()) + len(payload) <= LIMITS['output_bytes'],
            'Whole metadata output cap before publication')
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class MetadataGuard(CpuPrerequisiteGuard):
    def _sample(self):
        sample, reason = super()._sample()
        entries = list(OUT.iterdir())
        if any(p.is_symlink() or not p.is_file() for p in entries):
            reason = 'unsafe_metadata_output'
        sample['output_bytes'] = sum(p.lstat().st_size for p in entries)
        if sample['output_bytes'] > LIMITS['output_bytes']:
            reason = 'whole_metadata_output_cap'
        return sample, reason


def stat_identity(path):
    for part in (path, *path.parents):
        require(not part.is_symlink(), 'No symlink asset or ancestor')
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_nlink == 1, 'Single-link regular asset')
    return {'path': str(path), 'device': st.st_dev, 'inode': st.st_ino,
            'size': st.st_size, 'nlink': st.st_nlink, 'uid': st.st_uid,
            'mtime_ns': st.st_mtime_ns}


def read_asset(spec):
    path = Path(spec['path'])
    before = stat_identity(path)
    require(before == spec['identity'], 'Pinned asset identity before read')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        st = os.fstat(fd)
        require((st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns) ==
                (before['device'], before['inode'], before['size'], before['mtime_ns']),
                'Asset descriptor identity')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            payload = stream.read(before['size'] + 1)
        require(len(payload) == before['size'] and digest(payload) == spec['sha256'],
                'Exact public asset size and SHA before parsing')
        after = stat_identity(path)
        st_after = os.fstat(fd)
        require(after == before and (st_after.st_size, st_after.st_mtime_ns) ==
                (before['size'], before['mtime_ns']), 'Asset preserved during read')
    finally:
        os.close(fd)
    return payload, {'identity': before, 'sha256': digest(payload)}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key')
        result[key] = value
    return result


def inspect_assets(encoder_payload, merges_payload):
    # Fixed public byte alphabet, independent of installed tokenizer functions.
    visible = [b for lo, hi in VISIBLE_RANGES for b in range(lo, hi)]
    omitted = [b for b in range(256) if b not in visible]
    order = visible + omitted
    alphabet = {chr(b): b for b in visible}
    alphabet.update({chr(256 + i): b for i, b in enumerate(omitted)})
    require(len(alphabet) == 256 and set(alphabet.values()) == set(range(256)),
            'Exact bijective public byte alphabet')
    encoder = json.loads(encoder_payload.decode('utf8', errors='strict'),
                         object_pairs_hook=unique_object)
    require(type(encoder) is dict and len(encoder) == N_TOTAL, 'Full encoder count')
    require(all(type(k) is str and type(v) is int for k, v in encoder.items()),
            'Exact encoder key/ID types')
    require(set(encoder.values()) == set(range(N_TOTAL)), 'Dense unique full encoder IDs')
    require(encoder.get('<|endoftext|>') == 50256, 'Explicit special ID')
    regular = {}
    for spelling, idx in encoder.items():
        if idx == 50256:
            continue
        require(spelling and all(ch in alphabet for ch in spelling), 'Nonempty mapped regular token')
        raw = bytes(alphabet[ch] for ch in spelling)
        require(raw not in regular and idx < N_REGULAR, 'Unique regular bytes and ID')
        regular[raw] = idx
    require(len(regular) == N_REGULAR, 'Full regular byte-string count')
    lines = merges_payload.decode('utf8', errors='strict').splitlines()
    require(lines[0] == '#version: 0.2' and len(lines) == N_MERGES + 1,
            'Exact public merge header and count')
    merged = {bytes([b]): rank for rank, b in enumerate(order)}
    for rank, line in enumerate(lines[1:], start=256):
        pair = line.split(' ')
        require(len(pair) == 2 and all(pair), 'Two exact merge components')
        require(all(all(ch in alphabet for ch in item) for item in pair), 'Merge alphabet')
        left, right = (bytes(alphabet[ch] for ch in item) for item in pair)
        require(left in merged and right in merged, 'Merge operands already ranked')
        raw = left + right
        require(raw not in merged, 'No duplicate merged byte string')
        merged[raw] = rank
    require(merged == regular, 'All encoder IDs equal independently constructed merge ranks')
    # Exact native descriptor serialization, distinct from prefix codec's compact JSON.
    native_hash = digest(json.dumps(sorted((raw.hex(), idx) for raw, idx in regular.items())).encode())
    require(native_hash == EXPECTED_NATIVE_HASH, 'Accepted native vocabulary identity')
    tokens = [None] * N_REGULAR
    for raw, idx in regular.items():
        tokens[idx] = raw
    lengths = [len(raw) for raw in tokens]
    invalid_utf8 = []
    for idx, raw in enumerate(tokens):
        try:
            raw.decode('utf8', errors='strict')
        except UnicodeDecodeError:
            invalid_utf8.append(idx)
    maximum = max(lengths)
    require(len([n for n in lengths if n == 1]) == 256, 'Exactly all 256 one-byte units')
    require(b'<|endoftext|>' not in regular, 'Special is distinct from every whole regular token')
    return {
        'regular_count': N_REGULAR, 'total_count': N_TOTAL, 'merge_count': N_MERGES,
        'native_vocabulary_sha256': native_hash, 'all_encoder_merge_IDs_equal': True,
        'lengths_by_regular_ID': lengths, 'length_histogram': dict(collections.Counter(lengths)),
        'total_regular_bytes': sum(lengths), 'min_bytes': min(lengths), 'max_bytes': maximum,
        'max_length_IDs': [i for i, n in enumerate(lengths) if n == maximum],
        'invalid_standalone_utf8_IDs': invalid_utf8,
        'leading_space_regular_count': sum(raw.startswith(b' ') for raw in tokens),
        'contains_NUL_regular_count': sum(b'\0' in raw for raw in tokens),
        'one_byte_IDs_by_byte': [regular[bytes([b])] for b in range(256)],
        'special': {'id': 50256, 'spelling': '<|endoftext|>', 'raw_hex': b'<|endoftext|>'.hex(),
                    'byte_length': 13, 'is_mergeable_regular_token': False,
                    'ordinary_literal_policy_preserved': True},
        'symbolic_buffers': {'uint8_regular_payload_bytes': sum(lengths),
                             'int64_regular_offsets_50257_bytes': 50257 * 8,
                             'int64_full_lengths_bytes': N_TOTAL * 8,
                             'int64_full_padded_byte_table_bytes': N_TOTAL * maximum * 8,
                             'full_B8T128_padded_positions': 1024 * maximum,
                             'with_two_boundaries_positions': 1024 * (maximum + 2),
                             'fp32_byte_embedding_activation_bytes_per_e': 1024 * (maximum + 2) * 4,
                             'distinguishable_planning_codes_only': {'bytes': [0, 255], 'pad': 256,
                                'begin': 257, 'end': 258, 'special': 259}},
        'not_a_model_component_device_state_or_quality_qualification': True}


def main(plan_sha):
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require(set(p.name for p in OUT.iterdir()) == {'plan.json'}, 'One original entry only')
        write_new(OUT / 'attempt.json', {'pid': os.getpid(), 'plan_sha256': plan_sha,
                                       'no_replay': True, 'no_retry': True})
        try:
            with MetadataGuard(OUT / 'guard.json', ROOT, timeout_seconds=120,
                               rss_limit_bytes=4294967296, reserve_bytes=8589934592,
                               poll_seconds=.1):
                require(sys.version.split()[0] == '3.13.12', 'Pinned stdlib interpreter version')
                raw = (OUT / 'plan.json').read_bytes()
                require(digest(raw) == plan_sha, 'Immutable original plan')
                plan = json.loads(raw)
                require(plan['limits'] == LIMITS, 'Fixed resources')
                require(os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize >= LIMITS['entry_bytes'],
                        'Output plus reserve and extra headroom')
                gate = BASE / 'vocabulary-source001'
                receipt = json.loads((gate / 'review.json').read_text())
                require(receipt['accepted'] is True and receipt['consumer_plan_sha256'] == plan_sha,
                        'Independent source admission first')
                require(json.loads((gate / 'guard.json').read_text())['status'] == 'finished' and
                        json.loads((gate / 'STOP').read_text())['status'] == 'complete_source_only',
                        'Original source gate and guard complete')
                for pin in plan['source_and_metadata_pins']:
                    p = ROOT / pin['path']
                    payload = p.read_bytes()
                    require(len(payload) == pin['size'] and digest(payload) == pin['sha256'],
                            'Exact source or accepted-metadata pin')
                require(sum(x['identity']['size'] for x in plan['assets']) +
                        sum(x['size'] for x in plan['source_and_metadata_pins']) < LIMITS['input_bytes'],
                        'Whole anticipated input bound')
                descriptor = json.loads((ROOT / plan['native_descriptor_path']).read_text())['data']['tokenizer']
                require(descriptor == plan['native_descriptor'], 'Accepted descriptor metadata unchanged')
                require(descriptor['vocabulary_sha256'] == EXPECTED_NATIVE_HASH and
                        descriptor['vocab_size'] == N_TOTAL and descriptor['special_token_policy'] == 'ordinary_text',
                        'Unchanged native identity and ordinary-text policy')
                payloads = [read_asset(spec) for spec in plan['assets']]
                report = inspect_assets(payloads[0][0], payloads[1][0])
                for spec in plan['assets']:
                    require(stat_identity(Path(spec['path'])) == spec['identity'], 'Final asset identity preserved')
                report.update(status='complete_original_public_vocabulary_byte_metadata',
                              plan_sha256=plan_sha, assets=[x[1] for x in payloads],
                              native_descriptor=descriptor, no_tokenizer_corpus_model_RNG_calls=True)
                write_new(OUT / 'result.json', report)
            write_new(OUT / 'STOP', {'status': 'complete_original_public_metadata',
                                    'result_sha256': digest((OUT / 'result.json').read_bytes()),
                                    'no_replay': True, 'no_retry': True})
        except BaseException as exc:
            if not (OUT / 'STOP').exists():
                write_new(OUT / 'STOP', {'status': 'closed_failed_original_public_metadata',
                                        'error': repr(exc), 'no_replay': True, 'no_retry': True})
            raise


if __name__ == '__main__':
    main(sys.argv[1])
