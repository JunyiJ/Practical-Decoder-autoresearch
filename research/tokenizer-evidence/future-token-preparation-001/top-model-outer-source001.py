"""NEW TOP outer-entry AST/source gate only; no wrapper/proof/Torch execution."""
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
    raw = (json.dumps(value,sort_keys=True,allow_nan=False) + '\n').encode()
    assert sum(path.stat().st_size for path in R.iterdir() if path.is_file()) + len(raw) + 65536 < 2097152
    with (R / name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def review(plan):
    for path, digest in plan['sources'].items(): assert sha(ROOT / path) == digest
    source = (ROOT / plan['outer_source']).read_text(); tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node,ast.Import): imports.extend(alias.name for alias in node.names)
        if isinstance(node,ast.ImportFrom): imports.append(node.module)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name): assert node.func.id not in {'exec','eval','compile','__import__'}
    assert set(imports) == {'time','hashlib','json','os','pathlib','runpy','sys','src.research.cpu_prerequisite_guard'}
    assert source.index('BOOT_MONOTONIC = time.monotonic()') < source.index('from src.research.cpu_prerequisite_guard')
    constants = {node.targets[0].id:ast.literal_eval(node.value) for node in tree.body
                 if isinstance(node,ast.Assign) and node.targets[0].id in ('RUN_RELATIVE','OUTER_RESOURCES')}
    assert constants['RUN_RELATIVE'] == 'research/tokenizer-evidence/future-token-preparation-001/top-model-math001'
    assert constants['OUTER_RESOURCES'] == {'elapsed_seconds':600,'RSS_bytes':4294967296,'reserve_bytes':8589934592,'output_bytes':536870912}
    functions = {node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    assert set(functions) == {'sha','output_observation','write','main'}
    main = ast.get_source_segment(source,functions['main'])
    assert main.index('with OuterGuard(run):') < main.index('plan = json.loads') < main.index('runpy.run_path(')
    assert all(term in main for term in ("plan.get('execution_admitted') is not True", "plan['outer_resources'] != OUTER_RESOURCES",
        "plan['outer_entry'] != self_path", "plan['sources'].get(self_path) != sha(__file__)", "sha(PROOF)",
        "plan['runtime_admission_review']", "admission['accepted'] is not True", "admission['numerical_plan_sha256'] != original_argv[2]",
        "runpy.run_path(str(PROOF),run_name='__main__')", "complete_NEW_TOP_model_math_finite_scope_only",
        "write(run,'outer-STOP'", 'except BaseException as error:', 'finally:', 'sys.argv = original_argv'))
    assert "'outer-attempt.json','outer-guard.json','outer-STOP','attempt.json','STOP','result.json'" in main
    guard = next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name == 'OuterGuard')
    methods = {node.name:ast.get_source_segment(source,node) for node in guard.body if isinstance(node,ast.FunctionDef)}
    assert set(methods) == {'__init__','_sample'}
    assert 'timeout_seconds=600,poll_seconds=.25' in methods['__init__']
    assert all(term in methods['_sample'] for term in ('super()._sample()', 'time.monotonic() - BOOT_MONOTONIC',
        'elapsed >= 600', "size >= OUTER_RESOURCES['output_bytes']", 'whole_entry_wall_time_limit',
        'unsafe_nonflat_or_symlink_output','whole_JSON_output_limit'))
    observation = ast.get_source_segment(source,functions['output_observation'])
    assert 'path.is_symlink() or not path.is_file()' in observation and 'path.stat().st_size' in observation
    census = ROOT / plan['accepted_census_result']
    assert sha(census) == plan['accepted_census_result_sha256']
    receipt = json.loads(census.read_text())
    assert receipt['accepted'] and receipt['prior_matches'] == 0 and receipt['model_numerical_execution_admitted'] is False
    assert receipt['role_status'] == 'RESERVED_EXCLUSIVE_TOP_MODEL_MATH001_UNUSED_NOT_DRAWN'
    assert plan['predecessor_bindings'] == 1264 and plan['future_source_bindings'] == 1266
    assert plan['numerical_execution_admitted'] is False and plan['runtime_plan'] is None
    assert plan['limits'] == {'CPU_seconds':120,'RSS_bytes':4294967296,'reserve_bytes':8589934592,'metadata_bytes':2097152}
    write('review.json',{'status':'complete_NEW_TOP_outer_entry_SOURCE_AST_review_only','sources':plan['sources'],
        'future_source_bindings':1266,'numerical_execution_admitted':False,'runtime_plan':None,
        'no_outer_wrapper_proof_Torch_or_extracted_function_execution':True,
        'admission_scope':'Next independent saved-report reviewer/runtime admission SOURCE preparation only'})


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
            write('STOP',{'status':'closed_failed_NEW_TOP_outer_SOURCE_review','error':repr(error),'no_retry':True,'no_replay':True})
            raise
        else:
            write('STOP',{'status':'complete_NEW_TOP_outer_SOURCE_review','no_retry':True,'no_replay':True})


if __name__ == '__main__': main()
