"""NEW TOP copied-readout/shared-body fixture initializer census, stdlib only.

One original text census; no model/numerical/RNG imports or role draws.
Accepted bounded scanner is used on NEW role/text inputs, never its old fixtures.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from research.recorded_role_scan_v1 import scan
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')
CAP = 33554432
MAX_FILES = 75000
MAX_SCAN_BYTES = 8589934592
SCOPES = ('research', 'research_runs', 'research_state', 'src', 'tests', 'configs', 'scripts')
FORBIDDEN = ('torch', 'numpy', 'research.top_proximity_model_v1',
             'research.top_proximity_targets_v1', 'research.top_proximity_objective_v1')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def forbidden_imports():
    return [name for name in sys.modules if any(name == prefix or name.startswith(prefix + '.') for prefix in FORBIDDEN)]


def write(name, payload):
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + '\n').encode()
    if sum(p.stat().st_size for p in R.iterdir() if p.is_file()) + len(raw) + 65536 > CAP:
        raise ValueError('Original32MiB census metadata cap; no expansion/retry')
    with (R / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def identity(value): return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def enumerate_paths():
    found, missing, skipped_links = [], [], []
    def add(scope, base, path):
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            skipped_links.append({'scope': scope, 'path': str(path.relative_to(base)), 'reason': 'symlink_not_followed'})
        elif stat.S_ISREG(info.st_mode):
            if R not in path.parents:
                found.append((scope, base, path, info))
    for label in SCOPES + ('automations',):
        base = ROOT / label if label != 'automations' else Path.home() / '.codex/automations'
        if not base.exists():
            if label not in ('configs', 'scripts'): raise ValueError('Required census root absent: ' + label)
            missing.append(label); continue
        if base.is_symlink() or not base.is_dir(): raise ValueError('Unsafe census root: ' + label)
        def walk_error(error): raise error
        for current, dirs, files in os.walk(base, followlinks=False, onerror=walk_error):
            for directory in list(dirs):
                path = Path(current) / directory
                if directory == '__pycache__': dirs.remove(directory)
                elif path.is_symlink():
                    skipped_links.append({'scope':label,'path':str(path.relative_to(base)),'reason':'symlink_directory_not_followed'})
                    dirs.remove(directory)
                elif path == R: dirs.remove(directory)
            for name in files: add(label, base, Path(current) / name)
    for path in ROOT.iterdir():
        if path.is_file() or path.is_symlink(): add('root-files', ROOT, path)
    found.sort(key=lambda entry: (entry[0], str(entry[2])))
    if len(found) > MAX_FILES: raise ValueError('Prospective file bound exceeded')
    if len({str(item[2]) for item in found}) != len(found): raise ValueError('Duplicate census path')
    upper = 0
    for label, base, path, info in found:
        row = {'scope':label,'path':str(path.relative_to(base)),'size':info.st_size,
               'device':info.st_dev,'inode':info.st_ino,'mtime_ns':info.st_mtime_ns,
               'classification':'text','sha256':'0' * 64,'scanned_bytes':info.st_size}
        upper += len((json.dumps(row,sort_keys=True,separators=(',',':')) + '\n').encode()) + 32
    if upper + 2097152 > CAP: raise ValueError('Complete anticipated inventory cannot fit original cap')
    return found, missing, skipped_links, upper


def census(plan):
    if forbidden_imports(): raise ValueError('Forbidden numerical/model import')
    roles = plan['roles']
    if type(roles) is not dict or set(roles) != {'initializer'}: raise ValueError('One exact fixture initializer role required')
    role = roles['initializer']
    if type(role) is not int or not 0 < role < 2**63: raise ValueError('Exact positive unused role proposal required')
    needles = {str(role).encode('ascii')}
    files, missing, skipped, upper = enumerate_paths()
    write('inventory-preflight.json', {'candidate_files':len(files),'complete_inventory_upper_bytes':upper,
          'optional_absent_roots':missing,'skipped_links':skipped,'current_proposal_and_output_exclusion':str(R.relative_to(ROOT)),
          'no_content_has_yet_been_scanned':True})
    text_files = binary_files = text_bytes = scanned_bytes = inventory_bytes = 0
    scope_counts, matches = {}, []
    with (R / 'inventory.jsonl').open('xb') as inventory:
        for label, base, path, before in files:
            flags = os.O_RDONLY | os.O_NOFOLLOW
            descriptor = os.open(path, flags)
            with os.fdopen(descriptor, 'rb') as stream:
                if identity(os.fstat(stream.fileno())) != identity(before): raise ValueError('File changed before read: ' + str(path))
                def chunks():
                    nonlocal scanned_bytes
                    first = True
                    while True:
                        chunk = stream.read(8192 if first else 1048576); first = False
                        if not chunk: break
                        scanned_bytes += len(chunk)
                        if scanned_bytes > MAX_SCAN_BYTES: raise ValueError('Original8GiB streamed-read bound exceeded')
                        yield chunk
                result = scan(chunks(), needles)
                after = os.fstat(stream.fileno())
            if identity(after) != identity(before) or identity(path.lstat()) != identity(before):
                raise ValueError('File changed during read: ' + str(path))
            row = {'scope':label,'path':str(path.relative_to(base)),'size':before.st_size,
                   'device':before.st_dev,'inode':before.st_ino,'mtime_ns':before.st_mtime_ns,
                   'classification':'binary' if result['binary'] else 'text',
                   'sha256':result['sha256'],'scanned_bytes':result['bytes']}
            raw = (json.dumps(row,sort_keys=True,separators=(',',':')) + '\n').encode()
            inventory_bytes += len(raw)
            if inventory_bytes > upper or inventory_bytes + 2097152 > CAP: raise ValueError('Inventory output bound exceeded')
            inventory.write(raw)
            scope_counts[label] = scope_counts.get(label, 0) + 1
            if result['binary']: binary_files += 1
            else:
                if result['bytes'] != before.st_size: raise ValueError('Text coverage differs')
                text_files += 1; text_bytes += result['bytes']
                matches.extend({'scope':label,'path':row['path'],**match} for match in result['matches'])
                if matches:
                    # Preserve original first matches and inventory BEFORE closing failure.
                    inventory.flush(); os.fsync(inventory.fileno())
                    write('prior-matches.json', {'matches':matches,'roles':roles,'consumer':plan['consumer']})
                    raise ValueError('Proposed initializer appears in prior recorded text; permanently close original proposal')
        inventory.flush(); os.fsync(inventory.fileno())
    for name, digest in plan['execution_sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('Executed source changed')
    if forbidden_imports(): raise ValueError('Forbidden numerical/model import after census')
    write('result.json', {'accepted':True,'roles':roles,'prior_matches':0,'status':'NEW_TOP_fixture_initializer_unused_in_declared_recorded_text',
        'recorded_text_files':text_files,'recorded_text_bytes':text_bytes,'binary_classifications':binary_files,
        'candidate_files':len(files),'scope_file_counts':scope_counts,'streamed_bytes':scanned_bytes,
        'inventory_bytes':inventory_bytes,'inventory_sha256':sha(R / 'inventory.jsonl'),
        'optional_absent_roots':missing,'excluded_current_proposal_and_outputs':str(R.relative_to(ROOT)),
        'role_status':'RESERVED_EXCLUSIVE_TOP_MODEL_MATH001_UNUSED_NOT_DRAWN',
        'consumer':plan['consumer'],'no_Torch_NumPy_model_oracle_RNG_seed_calls':True,
        'scope':'Declared project history/root files/automation files, NUL binary classification and no symlinks or __pycache__; not a universal unrecorded-history claim',
        'model_numerical_execution_admitted':False})


def main():
    if len(sys.argv) != 2 or sha(R / 'plan.json') != sys.argv[1]: raise ValueError('Exact original plan SHA required')
    if set(path.name for path in R.iterdir()) != {'plan.json'}: raise ValueError('Original command closed or unexpected outputs; no retry/replay')
    plan = json.loads((R / 'plan.json').read_text())
    if plan['kind'] != 'NEW_TOP_model_math_exclusive_initializer_text_census': raise ValueError('Census identity differs')
    if plan['limits'] != {'CPU_seconds':120,'RSS_bytes':4294967296,'reserve_bytes':8589934592,'metadata_bytes':CAP,
                          'max_candidate_files':MAX_FILES,'max_streamed_bytes':MAX_SCAN_BYTES}: raise ValueError('Original resource contract differs')
    if not plan['no_retry'] or not plan['no_replay']: raise ValueError('Once-only contract required')
    if shutil.disk_usage(ROOT).free < 10770972672: raise ValueError('32MiB+reserve8GiB+extra2GiB entry required')
    for name, digest in plan['execution_sources'].items():
        if sha(ROOT / name) != digest: raise ValueError('Execution source differs')
    gate_path = ROOT / plan['source_gate_review']
    gate = json.loads(gate_path.read_text())
    if gate['status'] != 'complete_NEW_TOP_initializer_census_SOURCE_review_only' or gate['census_plan_sha256'] != sys.argv[1]:
        raise ValueError('Original source gate does not admit this text census')
    scanner = ROOT / plan['accepted_scanner_receipt']['path']
    if sha(scanner) != plan['accepted_scanner_receipt']['sha256']: raise ValueError('Scanner receipt pin differs')
    if json.loads(scanner.read_text())['status'] != 'passed_new_literal_bounded_role_stream_oracle': raise ValueError('Scanner acceptance scope differs')
    if forbidden_imports(): raise ValueError('Forbidden import before original census')
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write('attempt.json', {'pid':os.getpid(),'plan_sha256':sys.argv[1],'no_retry':True,'no_replay':True})
        try:
            with CpuPrerequisiteGuard(R / 'guard.json', ROOT, timeout_seconds=120, poll_seconds=.25): census(plan)
        except BaseException as error:
            write('STOP', {'status':'closed_failed_NEW_TOP_initializer_census','error':repr(error),'no_retry':True,'no_replay':True})
            raise
        else:
            write('STOP', {'status':'complete_original_NEW_TOP_initializer_census','no_retry':True,'no_replay':True})


if __name__ == '__main__': main()
