"""Independent AST admission then saved-metadata reconciliation; no parser import."""
import ast
import collections
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'research/tokenizer-evidence/byte-composition-boundary-001'
PARSER = BASE / 'vocabulary-bytes001.py'
LIMITS = {'timeout_seconds': 120, 'rss_limit_bytes': 4294967296,
          'reserve_bytes': 8589934592, 'output_bytes': 2097152,
          'input_bytes': 8388608, 'entry_bytes': 10739515392}
EXPECTED_HASH = 'bdc599f0a294cff99d60cf83aed3870857b30bc0d1d06164cae137831b55257c'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def publish(out, name, value):
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    require(sum(p.stat().st_size for p in out.iterdir()) + len(payload) < 2097152,
            'Reviewer whole metadata output cap')
    with (out / name).open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def pins(plan):
    for pin in plan['pins']:
        payload = (ROOT / pin['path']).read_bytes()
        require(len(payload) == pin['size'] and sha(payload) == pin['sha256'], 'Frozen review pin')


def source_review(plan):
    # Parse only. No compile, eval, exec, runpy, imports or extracted functions from candidate.
    text = PARSER.read_text()
    tree = ast.parse(text)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(x.name for x in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(node.module)
    require(imports == {'collections', 'fcntl', 'hashlib', 'json', 'os', 'pathlib',
                        'stat', 'sys', 'time', 'src.research.cpu_prerequisite_guard'},
            'Exact stdlib/CPU-guard import boundary')
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    require(not any(isinstance(n.func, ast.Name) and n.func.id in
                    {'eval', 'exec', 'compile', '__import__'} for n in calls), 'No dynamic code execution')
    require(not any(isinstance(n.func, ast.Attribute) and n.func.attr in
                    {'unlink', 'remove', 'rmdir', 'rename', 'replace', 'write_bytes', 'write_text'}
                    for n in calls), 'No asset mutation or generic file overwrite API')
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                constants[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    require(constants['LIMITS'] == LIMITS and constants['N_REGULAR'] == 50256 and
            constants['N_TOTAL'] == 50257 and constants['N_MERGES'] == 50000 and
            constants['EXPECTED_NATIVE_HASH'] == EXPECTED_HASH and
            constants['VISIBLE_RANGES'] == ((33, 127), (161, 173), (174, 256)),
            'Exact resource/alphabet/count/hash literals')
    required_fragments = [
        "os.O_RDONLY | os.O_NOFOLLOW", "Single-link regular asset", "No symlink asset or ancestor",
        "Pinned asset identity before read", "Asset descriptor identity",
        "Exact public asset size and SHA before parsing", "Asset preserved during read",
        "Duplicate JSON key", "Dense unique full encoder IDs", "Explicit special ID",
        "Unique regular bytes and ID", "Exact public merge header and count",
        "Merge operands already ranked", "No duplicate merged byte string",
        "All encoder IDs equal independently constructed merge ranks", "Accepted native vocabulary identity",
        "Exactly all 256 one-byte units", "Special is distinct from every whole regular token",
        "errors='strict'", "Final asset identity preserved", "Independent source admission first",
        "Original source gate and guard complete", "Whole anticipated input bound",
        "fcntl.LOCK_EX | fcntl.LOCK_NB", "One original entry only", "path.open('xb')",
        "stream.flush()", "os.fsync(stream.fileno())", "CpuPrerequisiteGuard",
        "timeout_seconds=120", "reserve_bytes=8589934592", "rss_limit_bytes=4294967296",
        "closed_failed_original_public_metadata", "no_replay", "no_retry"]
    require(all(fragment in text for fragment in required_fragments), 'All read-only/full-consumer/guard predicates present')
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    read_calls = [ast.unparse(n.func) for n in ast.walk(funcs['read_asset']) if isinstance(n, ast.Call)]
    require('os.open' in read_calls and 'os.fstat' in read_calls and 'os.close' in read_calls,
            'Explicit read-only descriptor lifecycle')
    main = ast.unparse(funcs['main'])
    require(main.index('with MetadataGuard') < main.index("raw =") < main.index('read_asset(spec)'),
            'CPU/output guard before plan and asset content')
    consumer_path = BASE / 'vocabulary-bytes001/plan.json'
    raw = consumer_path.read_bytes()
    require(sha(raw) == plan['consumer_plan_sha256'], 'Immutable consumer plan')
    consumer = json.loads(raw)
    require(consumer['limits'] == LIMITS and consumer['scope'] == 'public_vocabulary_metadata_only',
            'No model/quality admission')
    require(consumer['native_descriptor']['vocabulary_sha256'] == EXPECTED_HASH,
            'Accepted native descriptor bound')
    require(len(consumer['assets']) == 2 and len({s['path'] for s in consumer['assets']}) == 2,
            'Exactly two unique public assets')
    require(sum(s['identity']['size'] for s in consumer['assets']) +
            sum(s['size'] for s in consumer['source_and_metadata_pins']) < 8388608,
            'Complete anticipated read budget')
    require(consumer['forecast']['seconds'] == 30 and consumer['forecast']['RSS_bytes'] == 268435456,
            'Prospective bounded metadata cost only')
    return {'status': 'complete_AST_public_vocabulary_source_only', 'accepted': True,
            'consumer_plan_sha256': sha(raw), 'imports': sorted(imports),
            'no_candidate_import_extracted_function_or_asset_content_read': True,
            'scope': 'Only one original read-only public vocabulary metadata inspection'}


def saved_review(plan):
    out = BASE / 'vocabulary-bytes001'
    require({p.name for p in out.iterdir()} == {'plan.json', 'attempt.json', 'guard.json', 'result.json', 'STOP'},
            'Exact original complete metadata output set')
    raw_plan = (out / 'plan.json').read_bytes()
    require(sha(raw_plan) == plan['consumer_plan_sha256'], 'Original consumer plan')
    consumer = json.loads(raw_plan)
    raw = (out / 'result.json').read_bytes()
    report = json.loads(raw)
    stop = json.loads((out / 'STOP').read_text())
    guard = json.loads((out / 'guard.json').read_text())
    require(stop['status'] == 'complete_original_public_metadata' and stop['result_sha256'] == sha(raw),
            'Original completion and exact saved result hash')
    require(guard['status'] == 'finished' and guard['limits'] ==
            {k: LIMITS[k] for k in ['timeout_seconds', 'rss_limit_bytes', 'reserve_bytes']},
            'All original fixed resource guards finished')
    require(report['plan_sha256'] == plan['consumer_plan_sha256'] and
            report['native_vocabulary_sha256'] == EXPECTED_HASH and
            report['native_descriptor'] == consumer['native_descriptor'], 'Saved native identity')
    require(report['regular_count'] == 50256 and report['total_count'] == 50257 and
            report['merge_count'] == 50000 and report['all_encoder_merge_IDs_equal'] is True,
            'Full original vocabulary coverage')
    lengths = report['lengths_by_regular_ID']
    require(type(lengths) is list and len(lengths) == 50256 and
            all(type(x) is int and 1 <= x <= 1042301 for x in lengths), 'Every regular-ID length')
    histogram = {str(k): v for k, v in collections.Counter(lengths).items()}
    require(histogram == report['length_histogram'] and sum(lengths) == report['total_regular_bytes'],
            'All length coordinates and histogram arithmetic')
    maximum = max(lengths)
    require(report['min_bytes'] == min(lengths) and report['max_bytes'] == maximum and
            report['max_length_IDs'] == [i for i, x in enumerate(lengths) if x == maximum],
            'Exact minimum/maximum/all maximum IDs')
    units = report['one_byte_IDs_by_byte']
    require(len(units) == len(set(units)) == 256 and all(type(i) is int and 0 <= i < 50256 for i in units)
            and set(units) == {i for i, x in enumerate(lengths) if x == 1}, 'Complete unique one-byte IDs')
    bad = report['invalid_standalone_utf8_IDs']
    require(bad == sorted(set(bad)) and all(type(i) is int and 0 <= i < 50256 for i in bad),
            'Explicit unique partial/invalid standalone UTF8 ID metadata')
    require(report['special'] == {'id': 50256, 'spelling': '<|endoftext|>',
            'raw_hex': '3c7c656e646f66746578747c3e', 'byte_length': 13,
            'is_mergeable_regular_token': False, 'ordinary_literal_policy_preserved': True},
            'Special and ordinary policy kept distinct')
    require(all(r['identity'] == s['identity'] and r['sha256'] == s['sha256']
                for r, s in zip(report['assets'], consumer['assets'])) and len(report['assets']) == 2,
            'Both original expected public hashes/identities reported')
    buffers = report['symbolic_buffers']
    require(buffers['uint8_regular_payload_bytes'] == sum(lengths) and
            buffers['int64_regular_offsets_50257_bytes'] == 50257*8 and
            buffers['int64_full_lengths_bytes'] == 50257*8 and
            buffers['int64_full_padded_byte_table_bytes'] == 50257*maximum*8 and
            buffers['full_B8T128_padded_positions'] == 1024*maximum and
            buffers['with_two_boundaries_positions'] == 1024*(maximum+2) and
            buffers['fp32_byte_embedding_activation_bytes_per_e'] == 1024*(maximum+2)*4,
            'Independent symbolic buffer/activation arithmetic')
    require(report['no_tokenizer_corpus_model_RNG_calls'] is True and
            report['not_a_model_component_device_state_or_quality_qualification'] is True,
            'Metadata-only scope')
    return {'status': 'accepted_original_public_vocabulary_metadata_only', 'accepted': True,
            'consumer_plan_sha256': plan['consumer_plan_sha256'], 'result_sha256': sha(raw),
            'regular_count': 50256, 'max_bytes': maximum, 'total_regular_bytes': sum(lengths),
            'invalid_standalone_utf8_count': len(bad), 'no_asset_parser_tokenizer_or_model_replay': True,
            'scope_limit': 'Reconciles saved metadata/arithmetic; does not rederive byte mapping or UTF8 flags from digest alone'}


def main(mode, expected_sha):
    require(mode in {'source', 'saved'}, 'Declared original review mode')
    out = BASE / ('vocabulary-source001' if mode == 'source' else 'vocabulary-saved-review001')
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        require({p.name for p in out.iterdir()} == {'plan.json'}, 'One original review attempt only')
        publish(out, 'attempt.json', {'pid': os.getpid(), 'mode': mode, 'plan_sha256': expected_sha,
                                      'no_replay': True, 'no_retry': True})
        try:
            with CpuPrerequisiteGuard(out/'guard.json', ROOT, timeout_seconds=120,
                                     rss_limit_bytes=4294967296, reserve_bytes=8589934592, poll_seconds=.1):
                raw = (out/'plan.json').read_bytes()
                require(sha(raw) == expected_sha, 'Immutable review plan')
                plan = json.loads(raw)
                require(plan['mode'] == mode and plan['limits'] == LIMITS, 'Fixed mode/resources')
                require(os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize >= LIMITS['entry_bytes'],
                        'Metadata/reserve/headroom entry bound')
                pins(plan)
                result = source_review(plan) if mode == 'source' else saved_review(plan)
                publish(out, 'review.json', result)
            publish(out, 'STOP', {'status': 'complete_source_only' if mode == 'source' else 'complete_saved_metadata_review',
                                   'no_replay': True, 'no_retry': True})
        except BaseException as exc:
            publish(out, 'STOP', {'status': 'closed_failed_original_review', 'error': repr(exc),
                                  'no_replay': True, 'no_retry': True})
            raise


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
