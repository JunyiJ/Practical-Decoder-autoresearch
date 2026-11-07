"""NEW independent TOP initializer-census AST/literal gate; no census execution."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, value):
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    assert sum(path.stat().st_size for path in R.iterdir() if path.is_file()) + len(raw) + 65536 < 2097152
    with (R / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def review(plan):
    for path, digest in plan['sources'].items(): assert sha(ROOT / path) == digest
    path = ROOT / plan['census_plan']; assert sha(path) == plan['census_plan_sha256']
    census = json.loads(path.read_text())
    assert census['kind'] == 'NEW_TOP_model_math_exclusive_initializer_text_census'
    roles = census['roles']
    assert type(roles) is dict and set(roles) == {'initializer'}
    assert type(roles['initializer']) is int and 0 < roles['initializer'] < 2**63
    assert census['role_status'] == 'PROPOSED_EXCLUSIVE_TOP_MODEL_MATH001_NOT_AUDITED_NOT_DRAWN'
    assert census['consumer'] == 'research/tokenizer-evidence/future-token-preparation-001/top-model-math001.py'
    assert census['limits'] == {'CPU_seconds':120,'RSS_bytes':4294967296,'reserve_bytes':8589934592,
                               'metadata_bytes':33554432,'max_candidate_files':75000,'max_streamed_bytes':8589934592}
    assert census['complete_entry_target_bytes'] == 10770972672
    assert census['no_retry'] and census['no_replay'] and census['no_model_numerical_execution_admission']
    assert census['source_gate_review'] == str(R.relative_to(ROOT) / 'review.json')
    for source, digest in census['execution_sources'].items(): assert sha(ROOT / source) == digest
    anticipated = census['anticipated_metadata_only_inventory']
    assert anticipated['metadata_only_no_content_or_role_scan'] is True
    assert anticipated['all_class_inventory_upper_bytes'] == 22895648
    assert anticipated['all_class_inventory_upper_bytes'] + 2097152 < 33554432
    assert sum(value.get('files',0) for value in anticipated['scope_counts'].values()) < 75000
    assert anticipated['scope_counts']['configs']['present'] is False and anticipated['scope_counts']['scripts']['present'] is False
    assert census['forecast'] == {'seconds':110,'RSS_bytes':268435456,'inventory_upper_bytes':25165824,'text_bytes_allowance':6442450944}
    source = (ROOT / plan['census_source']).read_text(); tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        assert isinstance(node,(ast.Expr,ast.Import,ast.ImportFrom,ast.Assign,ast.FunctionDef,ast.If))
        if isinstance(node,ast.Assign) and node.targets[0].id != 'ROOT' and node.targets[0].id != 'R':
            constants[node.targets[0].id] = ast.literal_eval(node.value)
        if isinstance(node,ast.If): assert ast.unparse(node.test) == "__name__ == '__main__'"
    assert constants['CAP'] == 33554432 and constants['MAX_FILES'] == 75000 and constants['MAX_SCAN_BYTES'] == 8589934592
    assert constants['SCOPES'] == ('research','research_runs','research_state','src','tests','configs','scripts')
    assert constants['FORBIDDEN'] == ('torch','numpy','research.top_proximity_model_v1','research.top_proximity_targets_v1','research.top_proximity_objective_v1')
    imports = []
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.extend(alias.name for alias in node.names)
        if isinstance(node,ast.ImportFrom): imports.append(node.module)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name): assert node.func.id not in {'exec','eval','compile','__import__'}
    assert set(imports) <= {'fcntl','hashlib','json','os','pathlib','shutil','stat','sys','research.recorded_role_scan_v1','src.research.cpu_prerequisite_guard'}
    assert 'research.recorded_role_scan_v1' in imports
    functions = {node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    assert set(functions) == {'sha','forbidden_imports','write','identity','enumerate_paths','census','main'}
    def body(name): return ast.get_source_segment(source,functions[name])
    enumeration = body('enumerate_paths')
    assert all(term in enumeration for term in ("SCOPES + ('automations',)","Path.home() / '.codex/automations'",'followlinks=False',
        "label not in ('configs', 'scripts')", "directory == '__pycache__'", 'path == R', 'R not in path.parents',
        'stat.S_ISLNK','stat.S_ISREG','MAX_FILES','upper + 2097152 > CAP','ROOT.iterdir()'))
    scan = body('census')
    assert all(term in scan for term in ('enumerate_paths()',"str(role).encode('ascii')",'os.O_RDONLY | os.O_NOFOLLOW',
        'identity(os.fstat(stream.fileno())) != identity(before)','stream.read(8192 if first else 1048576)',
        'scanned_bytes > MAX_SCAN_BYTES','result = scan(chunks(), needles)',
        'identity(path.lstat()) != identity(before)',"'classification':'binary' if result['binary'] else 'text'",
        "'scanned_bytes':result['bytes']","result['bytes'] != before.st_size",'inventory.flush(); os.fsync(inventory.fileno())',
        "write('prior-matches.json'","if matches:","'accepted':True","'prior_matches':0",
        'RESERVED_EXCLUSIVE_TOP_MODEL_MATH001_UNUSED_NOT_DRAWN',"'model_numerical_execution_admitted':False"))
    assert scan.index("write('prior-matches.json'") < scan.index("raise ValueError('Proposed initializer appears")
    main = body('main')
    assert all(term in main for term in ("!= {'plan.json'}",'LOCK_EX | fcntl.LOCK_NB',"write('attempt.json'",'timeout_seconds=120',
        "plan['source_gate_review']", "gate['census_plan_sha256'] != sys.argv[1]",'accepted_scanner_receipt',
        "write('STOP'",'closed_failed_NEW_TOP_initializer_census','complete_original_NEW_TOP_initializer_census'))
    assert 'manual_seed' not in source and 'random.' not in source and 'import torch' not in source
    assert plan['future_source_bindings'] == 1264 and plan['predecessor_bindings'] == 1262
    assert plan['limits'] == {'CPU_seconds':120,'RSS_bytes':4294967296,'reserve_bytes':8589934592,'metadata_bytes':2097152}
    write('review.json', {'status':'complete_NEW_TOP_initializer_census_SOURCE_review_only',
          'census_plan_sha256':plan['census_plan_sha256'],'sources':plan['sources'],
          'future_source_bindings':1264,'one_proposed_role_not_audited_or_drawn':True,
          'no_census_scan_or_Torch_model_oracle_extracted_function_execution':True,
          'admitted_action':'One original declared-root stdlib text census only; no model/numerical admission'})


def main():
    assert len(sys.argv) == 2 and sha(R / 'plan.json') == sys.argv[1]
    assert set(path.name for path in R.iterdir()) == {'plan.json'}
    plan = json.loads((R / 'plan.json').read_text())
    assert shutil.disk_usage(ROOT).free >= 10739515392
    for path, digest in plan['sources'].items(): assert sha(ROOT / path) == digest
    with (ROOT / 'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
        write('attempt.json',{'pid':os.getpid(),'plan_sha256':sys.argv[1],'no_retry':True,'no_replay':True})
        try:
            with CpuPrerequisiteGuard(R / 'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25): review(plan)
        except BaseException as error:
            write('STOP',{'status':'closed_failed_NEW_TOP_initializer_SOURCE_review','error':repr(error),'no_retry':True,'no_replay':True})
            raise
        else:
            write('STOP',{'status':'complete_NEW_TOP_initializer_SOURCE_review','no_retry':True,'no_replay':True})


if __name__ == '__main__': main()
