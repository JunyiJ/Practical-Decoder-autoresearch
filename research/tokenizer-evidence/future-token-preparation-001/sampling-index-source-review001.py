"""NEW static review of isolated pure-index extraction and finite case/resource plan."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3]
R=Path(__file__).with_suffix('')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(folder,name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in folder.iterdir() if p.is_file())+len(raw)<=2*1024**2
    with (folder/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json'}
    plan=json.loads((R/'plan.json').read_text());publish(R,'attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
    try:
        with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            for name,h in plan['sources'].items():assert sha(ROOT/name)==h,name
            target=ROOT/plan['index_plan']['path'];assert sha(target)==plan['index_plan']['sha256']
            ip=json.loads(target.read_text());p=ROOT/ip['sampler_source'];tree=ast.parse(p.read_text())
            functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','positions')]
            assert [n.name for n in functions]==['require','positions']
            assert all(not n.decorator_list and not n.returns and not n.args.defaults and not any(n.args.kw_defaults) for n in functions)
            calls=[n for f in functions for n in ast.walk(f) if isinstance(n,ast.Call)]
            assert all(isinstance(n.func,ast.Name) and n.func.id in {'type','len','all','list','tuple','range','require','ValueError'} for n in calls)
            assert not any(isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Attribute,ast.Global,ast.Nonlocal)) for f in functions for n in ast.walk(f))
            isolated=ast.Module(body=functions,type_ignores=[])
            assert hashlib.sha256(ast.dump(isolated,include_attributes=False).encode()).hexdigest()==ip['extracted_AST_sha256']
            compile(isolated,str(p),'exec')  # declarations only compiled, never exec or call
            runner=ROOT/plan['runner'];rt=ast.parse(runner.read_text());compile(rt,str(runner),'exec')
            imported=[]
            for n in ast.walk(rt):
                if isinstance(n,ast.Import):imported.extend(a.name for a in n.names)
                if isinstance(n,ast.ImportFrom):imported.append(n.module)
            assert all(n and n.split('.')[0] not in ('torch','numpy','random') for n in imported)
            assert not any(n and n.startswith('research.') for n in imported)
            exec_calls=[n for n in ast.walk(rt) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='exec']
            assert len(exec_calls)==1 and ast.unparse(exec_calls[0].args[0])=="compile(isolated, str(path), 'exec')"
            assert len(ip['positive_cases'])==14 and len(ip['negative_cases'])==44
            assert len({c['name'] for c in ip['positive_cases']+ip['negative_cases']})==58
            assert all(len(c['expected_row_starts'])==len(c['expected_endpoints'])==8 for c in ip['positive_cases'])
            assert {c['length'] for c in ip['positive_cases']} >= {514,130,2**24}
            assert ip['limits']['CPU_seconds']==120 and ip['limits']['rss_bytes']==4*1024**3 and ip['limits']['reserve_bytes']==8*1024**3
            assert ip['complete_target_bytes']==4*1024**2+128*1024+10*1024**3
            assert ip['metadata_cap_bytes']==2*1024**2 and ip['forecast']['seconds']<120 and ip['forecast']['rss_bytes']<4*1024**3
            for name,h in plan['sources'].items():assert sha(ROOT/name)==h,name
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            result=dict(status='complete_static_pure_index_extraction_case_resource_review',sources=plan['sources'],index_plan_sha256=plan['index_plan']['sha256'],
                no_component_extraction_execution_or_torch_numpy_rng_sampler_model_calls=True,
                extraction_exactly_two_pure_functions=True,positive_cases=14,negative_cases=44,
                scope='admits_one_original_exact_integer_index_oracle_only;no_model_or_sampler_runtime')
            publish(R,'review.json',result)
            publish(target.parent,'source-review.json',dict(status='source_reviewed_one_original_pure_integer_index_oracle',review_sha256=sha(R/'review.json'),review_plan_sha256=arg.plan_sha))
        publish(R,'STOP',dict(status='complete_original_index_source_review',no_retry=True,no_replay=True))
    except BaseException as error:
        if not (R/'STOP').exists():publish(R,'STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
        raise


if __name__=='__main__':main()
