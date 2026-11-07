"""NEW finite exact-integer T+2 layout oracle; no Torch/module/sampler/RNG import."""
import argparse
import ast
import copy
import fcntl
import hashlib
import json
import operator
import os
from pathlib import Path
import shutil
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3]
R=Path(__file__).with_suffix('')
CAP=2*1024**2


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=CAP
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def extracted_positions(path):
    tree=ast.parse(path.read_text())
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','positions')]
    assert [n.name for n in functions]==['require','positions']
    assert all(not n.decorator_list and not n.returns and not n.args.defaults
               and not any(n.args.kw_defaults) for n in functions)
    allowed={'type','len','all','list','tuple','range','require','ValueError'}
    calls=[n for f in functions for n in ast.walk(f) if isinstance(n,ast.Call)]
    assert all(isinstance(n.func,ast.Name) and n.func.id in allowed for n in calls)
    assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Attribute,ast.Global,ast.Nonlocal))
                   for f in functions for n in ast.walk(f))
    namespace={'__builtins__':dict(type=type,int=int,list=list,tuple=tuple,len=len,all=all,range=range,ValueError=ValueError)}
    isolated=ast.Module(body=functions,type_ignores=[])
    exec(compile(isolated,str(path),'exec'),namespace)
    assert set(namespace)=={'__builtins__','require','positions'}
    return namespace['positions'],hashlib.sha256(ast.dump(isolated,include_attributes=False).encode()).hexdigest()


class ListSubclass(list):pass


def materialize(value):
    if type(value) is dict and len(value)==1:
        if 'as_tuple' in value:return tuple(value['as_tuple'])
        if 'as_iterator' in value:return iter(value['as_iterator'])
        if 'as_set' in value:return set(value['as_set'])
        if 'as_subclass' in value:return ListSubclass(value['as_subclass'])
    return copy.deepcopy(value)


def before(value):
    return ('iterator',operator.length_hint(value)) if type(value) is type(iter([])) else (type(value),copy.deepcopy(value))


def unchanged(value,saved):
    if saved[0]=='iterator':return type(value) is type(iter([])) and operator.length_hint(value)==saved[1]
    return type(value) is saved[0] and value==saved[1]


def positive(positions,case):
    starts=materialize(case['starts']);saved=before(starts)
    actual=positions(case['length'],starts,split=case['split'])
    # Independent declared literal row bases. The source function's row-start
    # expression is never reused to generate these expected row bases.
    expected=tuple(tuple(range(base,base+130)) for base in case['expected_row_starts'])
    flags=dict(exact_all1040_coordinates=actual==expected,
        tuple_output=type(actual) is tuple and all(type(row) is tuple for row in actual),
        exact_8_by_130=len(actual)==8 and all(len(row)==130 for row in actual),
        every_index_inside_split=all(type(i) is int and 0<=i<case['length'] for row in actual for i in row),
        all1024_input_and_both_target_positions=sum(len(row[:128]) for row in actual)==1024
            and sum(len(row[1:129]) for row in actual)==1024 and sum(len(row[2:]) for row in actual)==1024,
        exact_literal_endpoints=[(row[0],row[-1]) for row in actual]==[tuple(p) for p in case['expected_endpoints']],
        source_starts_preserved=unchanged(starts,saved))
    if case['split']=='train':
        # Separate oracle expands each physical514-ID block, then checks the
        # three flattened512-ID streams. It does not reshape514 into rows.
        inputs=[i for row in actual for i in row[:128]]
        nexts=[i for row in actual for i in row[1:129]]
        futures=[i for row in actual for i in row[2:]]
        flags.update(exact_two_input512_streams=inputs==[i for anchor in starts for i in range(anchor,anchor+512)],
            exact_two_next512_streams=nexts==[i for anchor in starts for i in range(anchor+1,anchor+513)],
            exact_two_future512_streams=futures==[i for anchor in starts for i in range(anchor+2,anchor+514)],
            adjacent_row_two_ID_overlap=all(actual[4*group+j][-2:]==actual[4*group+j+1][:2]
                for group in (0,1) for j in (0,1,2)))
    else:
        flags['eight_declared_independent_starts']=[row[0] for row in actual]==list(starts)
    report=dict(name=case['name'],length=case['length'],split=case['split'],starts_spec=case['starts'],
        actual_positions=actual,independent_expected_positions=expected,predicates=flags,
        passed=all(flags.values()),absolute_coordinate_error_max=max(abs(a-e) for ar,er in zip(actual,expected) for a,e in zip(ar,er)))
    publish(case['name']+'.json',report)
    assert report['passed'],case['name']
    return dict(name=case['name'],passed=True,predicates=flags,coordinates=1040)


def negative(positions,case):
    arguments={k:materialize(case[k]) for k in ('length','starts','split')};snapshots={k:before(v) for k,v in arguments.items()}
    rejected=False;error=None
    try:positions(arguments['length'],arguments['starts'],split=arguments['split'])
    except ValueError as exc:rejected=True;error=str(exc)
    preservation={k:unchanged(v,snapshots[k]) for k,v in arguments.items()}
    report=dict(name=case['name'],case=case,rejected_ValueError=rejected,caller_inputs_preserved=preservation,
                error=error,passed=rejected and all(preservation.values()))
    publish(case['name']+'.json',report)
    assert report['passed'],case['name']
    return dict(name=case['name'],passed=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json','source-review.json'}
    plan=json.loads((R/'plan.json').read_text())
    assert len(plan['positive_cases'])==14 and len(plan['negative_cases'])==44
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
    assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                positions,ast_sha=extracted_positions(ROOT/plan['sampler_source'])
                assert ast_sha==plan['extracted_AST_sha256']
                positives=[positive(positions,case) for case in plan['positive_cases']]
                negatives=[negative(positions,case) for case in plan['negative_cases']]
                assert 'torch' not in sys.modules and 'numpy' not in sys.modules
                for name,h in plan['execution_sources'].items():assert sha(ROOT/name)==h,name
                publish('result.json',dict(status='complete_new_literal_split_local_Tplus2_index_oracle',positive_cases=positives,negative_cases=negatives,
                    exact_coordinate_comparisons=14*1040,extracted_AST_sha256=ast_sha,source_pins_unchanged=True,
                    no_torch_numpy_sampler_generator_rng_model_corpus_checkpoint_or_scorer_calls=True,
                    scope='pure_positions_only;not_sampler_draw_state_or_physical_split_or_model_qualification'))
            publish('STOP',dict(status='complete_original_future_token_index_oracle',no_retry=True,no_replay=True))
        except BaseException as error:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
            raise


if __name__=='__main__':main()
