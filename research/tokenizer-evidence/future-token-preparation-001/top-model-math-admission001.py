"""NEW independent TOP runtime admission: AST/literals/source pins only.

Never import candidate/proof/reviewer/outer or execute extracted functions.
Preserved historical source bindings do not qualify their failed experiments.
"""
import ast
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')


def require(value,message):
    if not value: raise ValueError(message)


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path): return json.loads(Path(path).read_text())


def write(name,value):
    raw = (json.dumps(value,sort_keys=True,allow_nan=False) + '\n').encode()
    require(sum(p.stat().st_size for p in R.iterdir() if p.is_file()) + len(raw) + 65536 < 2097152,'Original2MiB admission cap')
    with (R/name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def constants(path):
    source = Path(path).read_text(); tree = ast.parse(source); values = {}
    for node in tree.body:
        if isinstance(node,ast.Assign) and isinstance(node.targets[0],ast.Name):
            try: values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError,TypeError): pass
        elif isinstance(node,ast.AnnAssign) and isinstance(node.target,ast.Name):
            try: values[node.target.id] = ast.literal_eval(node.value)
            except (ValueError,TypeError): pass
    return source,tree,values


def review(capsule):
    require(capsule['kind'] == 'NEW_TOP_runtime_admission_AST_source_only' and capsule['no_retry'] and capsule['no_replay'],'Original admission identity')
    require(capsule['limits'] == {'CPU_seconds':120,'RSS_bytes':4294967296,'reserve_bytes':8589934592,'metadata_bytes':2097152},'Original admission resources')
    require(capsule['future_source_bindings'] == 1268 and capsule['execution_source_bindings'] == 1274,'Exact project and runtime binding counts')
    path = ROOT/capsule['numerical_plan']; require(sha(path) == capsule['numerical_plan_sha256'],'New original numerical plan SHA')
    numerical = read(path); run = path.parent
    require({p.name for p in run.iterdir()} == {'plan.json'},'No numerical entry before complete admission')
    require(numerical['execution_admitted'] is True,'Plan requires independent receipt before entry')
    require(numerical['runtime_admission_review'] == str(R.relative_to(ROOT)/'review.json'),'Exact independent admission receipt')
    proof_path = ROOT/numerical['proof_source']; source,tree,c = constants(proof_path)
    require(numerical['resources'] == c['RESOURCE_CONTRACT'] and numerical['expected_reports'] == c['EXPECTED_REPORTS'],'Unchanged original cases/resource contract')
    require(c['EXPECTED_REPORTS'] == {'arms':10,'transactions':89,'tail_pairs':4,'updates':4},'Exact finite counts')
    case_map = {name:(context,window,rows) for name,context,window,rows in c['CASES']}
    arms = [[device+'-'+variant+'-'+name,variant,name,len(case_map[name][2]),case_map[name][0],case_map[name][1]]
            for device in c['BACKENDS'] for variant,name in c['ARMS']]
    negative_ids = [device+'-negative-'+name for device in c['BACKENDS']
                    for name in c['NEGATIVE_IDS'] + (c['CPU_ONLY_NEGATIVE_IDS'] if device == 'cpu' else ())]
    tail_ids = [device+'-tail-only-'+kind for device in c['BACKENDS'] for kind in ('primary','auxiliary')]
    files = {'plan.json','attempt.json','STOP','result.json','mps-guard.json','outer-attempt.json','outer-guard.json','outer-STOP','preservation-and-counts.json'}
    for ident,variant,name,_,_,_ in arms:
        files.update((ident+'-math.json',ident+'-phase-guard.json'))
        if variant != 'native' and name == 'repeated_tail_001': files.add(ident+'-update.json')
    files.update(ident+'.json' for ident in negative_ids+tail_ids)
    expected = {'arms':arms,'negative_ids':negative_ids,'tail_ids':tail_ids,'output_files':sorted(files),
                'tolerances':json.loads(json.dumps(c['TOLERANCES']))}
    require(numerical['report_contract'] == expected and len(files) == 126 and len(negative_ids) == 89,'Complete original report identity/coverage')
    require(len(arms) == 10 and len(set(row[0] for row in arms)) == 10,'Unique original arms')
    require(all(1 <= batch <= 2 and 1 <= context <= 3 and window in (1,4) for _,_,_,batch,context,window in arms),'Finite actual-model slice only')
    require('manual_seed(initializer)' in source and 'independent_targets' in source,'Original new consumer remains unchanged')
    # Literal config only; no candidate or contract import.
    _,_,cfg = constants(ROOT/'research/top_proximity_model_v1/contract.py')
    require(numerical['native_config'] == json.loads(cfg['NATIVE_MODEL_JSON']),'Exact genuine native config literal')
    outer_source,outer_tree,outer = constants(ROOT/numerical['outer_entry'])
    require(numerical['outer_resources'] == outer['OUTER_RESOURCES'] == {'elapsed_seconds':600,'RSS_bytes':4294967296,'reserve_bytes':8589934592,'output_bytes':536870912},'Outer whole-entry contract')
    require(outer_source.index('with OuterGuard(run):') < outer_source.index('runpy.run_path('),'Guard precedes functional entry')
    # Derive additive manifest from immutable SOURCE ledger only. The original
    # old model proof failed; none of its numerical outputs are consumed here.
    previous = ROOT/numerical['binding_provenance']['predecessor_manifest']
    require(sha(previous) == numerical['binding_provenance']['predecessor_manifest_sha256'],'Original source-ledger metadata pin')
    inherited = read(previous)['execution_sources']
    future = numerical['future_sources']; additions = numerical['binding_provenance']['additions']
    require(len(inherited) == 1236 and len(future) == 1268 and len(additions) == 32,'Exact source composition')
    require(all(future.get(name) == digest for name,digest in inherited.items()),'Preserve every historical source binding')
    require(set(future)-set(inherited) == set(additions),'Only explicit32 new source versions')
    require(numerical['sources'] == dict(future,**numerical['runtime_source_pins']),'Full execution pins include runtime source versions')
    for name,digest in numerical['sources'].items(): require(sha(ROOT/name) == digest,'New consumer execution pin differs: '+name)
    selected = {'gpt':'research/ffn_structure_v1/runtime/models/gpt.py','blocks':'research/local_mix_v1/runtime/models/blocks.py',
        'mlp':'research/ffn_structure_v1/runtime/models/mlp.py','attention':'research/block_norm_v1/runtime/attention/mla.py',
        'cache':'research/local_mix_v1/runtime/attention/cache.py','embedding':'research/block_norm_v1/runtime/models/grouped_embedding.py'}
    require(numerical['selected_static_native_sources'] == selected and all(name in future for name in selected.values()),'Static selected native owner pins')
    runtime = numerical['runtime']
    actual = {'python':sys.version,'executable':sys.executable,'machine':platform.machine(),'platform':platform.platform(),
              'packages':{name:importlib.metadata.version(name) for name in ('torch','numpy','omegaconf','hydra-core','tiktoken')}}
    require(runtime['identity'] == actual,'Actual current interpreter/package versions')
    _,_,torch_version = constants(ROOT/runtime['torch_version_source'])
    require(torch_version['__version__'] == runtime['torch_version'] == actual['packages']['torch']
            and torch_version['git_version'] == runtime['torch_git'] == '5c4886908584029761b579af026dcfb627c84070','Installed Torch version/git source, no binary/kernel profile')
    require(actual['packages']['torch'] == '2.14.1' and sys.version_info[:3] == (3,13,12),'Declared local runtime only')
    environment = {'PYTHONPATH':'.','PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    require(numerical['environment'] == environment and all(os.environ.get(k) == v for k,v in environment.items()),'Strict original launch environment')
    census_path = ROOT/numerical['initializer_census']; census = read(census_path)
    require(sha(census_path) == numerical['initializer_census_sha256'] and census['roles'] == {'initializer':numerical['initializer']},'Same original exclusive initializer receipt')
    require(census['accepted'] is True and census['prior_matches'] == 0 and census['role_status'] == 'RESERVED_EXCLUSIVE_TOP_MODEL_MATH001_UNUSED_NOT_DRAWN','Accepted unused fixture role, no role draw')
    require(census['consumer'] == numerical['proof_source'],'Exclusive original model-math consumer')
    require(read(census_path.parent/'STOP')['status'] == 'complete_original_NEW_TOP_initializer_census'
            and read(census_path.parent/'guard.json')['status'] == 'finished','Full original census acceptance')
    metadata = read(ROOT/numerical['census_metadata_review'])
    require(metadata['accepted_unused_exclusive_role_only'] is True and metadata['roles'] == census['roles'],'First saved census metadata reconciliation')
    for name,digest in numerical['metadata_pins'].items(): require(sha(ROOT/name) == digest,'Original gate/census metadata binding differs')
    # Independent reviewer is SOURCE here; do not execute its decoder, shape
    # function, comparisons or any synthetic/new/old numerical fixture.
    reporter = ROOT/numerical['report_reviewer_source']; reporter_source,reporter_tree,_ = constants(reporter)
    imports = []
    for node in ast.walk(reporter_tree):
        if isinstance(node,ast.Import): imports.extend(a.name for a in node.names)
        if isinstance(node,ast.ImportFrom): imports.append(node.module)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name): require(node.func.id not in {'eval','exec','compile','__import__'},'No extracted/hidden imports')
    require(set(imports) == {'array','base64','fcntl','hashlib','json','math','os','pathlib','shutil','sys','zlib','src.research.cpu_prerequisite_guard'},'Independent stdlib-only saved-report reviewer')
    require(all(term in reporter_source for term in ('decoder.decompress(compressed,expected + 1)','not decoder.unused_data','not decoder.unconsumed_tail',
       'error <= atol + rtol * abs(expected)','predicate == bool(mask[index])','mask.count(0)','len(expected_files) == 126',
       "len(names) == len(set(names))", "set(init['parameters']) == set(native)","set(value['parameter_gradients']) == set(inventory)",
       "same(caller['before'],caller['after'])", "same(value['before'],value['after'])", "'outer-STOP'",'no_Torch_model_oracle_target_objective_scoring_or_RNG_replay')),'All-coordinate evidence/ownership source checks')
    reviewer_plan = ROOT/capsule['reviewer_plan']
    require(sha(reviewer_plan) == capsule['reviewer_plan_sha256'] and read(reviewer_plan)['numerical_plan_sha256'] == capsule['numerical_plan_sha256'],'Prospective original metadata reviewer plan')
    require(numerical['complete_entry_target_bytes'] == 11280580608 and shutil.disk_usage(ROOT).free >= numerical['complete_entry_target_bytes'],'512MiB+three2MiB metadata caps+reserve8GiB+extra2GiB whole consumer')
    require(numerical['cost']['model_forecast_seconds'] == 360 and numerical['cost']['postreview_forecast_seconds'] == 30
            and numerical['cost']['postreview_hard_CPU_seconds'] == 120 and numerical['cost']['whole_model_hard_seconds'] == 600,'Full serial time budget')
    require(numerical['no_retry'] and numerical['no_replay'] and numerical['no_science_state_cache_quality_admission'],'No rescue or scientific admission')
    require(not any(name == 'torch' or name.startswith('torch.') for name in sys.modules),'No Torch/candidate functional execution by gate')
    write('review.json',{'status':'complete_NEW_TOP_model_math_runtime_admission','accepted':True,
          'numerical_plan_sha256':capsule['numerical_plan_sha256'],'future_source_bindings':1268,
          'execution_source_bindings':len(numerical['sources']),'one_original_finite_entry_only':True,
          'no_Torch_candidate_oracle_reviewer_or_extracted_function_execution':True,
          'scope':'Only original outer-guarded NEW finite model-math fixture; no old proof/state/cache/scoring/quality qualification'})


def main():
    require(len(sys.argv) == 2 and sha(R/'plan.json') == sys.argv[1],'Exact original admission capsule SHA')
    require({p.name for p in R.iterdir()} == {'plan.json'},'Original admission gate closed/attempted; no replay')
    capsule = read(R/'plan.json')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write('attempt.json',{'pid':os.getpid(),'plan_sha256':sys.argv[1],'no_retry':True,'no_replay':True})
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25): review(capsule)
        except BaseException as error:
            write('STOP',{'status':'closed_failed_NEW_TOP_runtime_admission','error':repr(error),'no_retry':True,'no_replay':True})
            raise
        else:
            write('STOP',{'status':'complete_original_NEW_TOP_runtime_admission','no_retry':True,'no_replay':True})


if __name__ == '__main__': main()
