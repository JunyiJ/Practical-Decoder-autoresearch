"""NEW static admission review of generated proof/census; no Torch/model imports."""
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
    plan=json.loads((R/'plan.json').read_text())
    publish(R,'attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
    try:
        with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            for name,h in plan['new_execution_sources'].items():assert sha(ROOT/name)==h,name
            for ref in plan['reviewed_plans']:
                assert sha(ROOT/ref['path'])==ref['sha256']
            proof=ROOT/plan['proof_source'];census=ROOT/plan['census_source']
            pt,ct=ast.parse(proof.read_text()),ast.parse(census.read_text())
            compile(pt,str(proof),'exec');compile(ct,str(census),'exec')
            fn={n.name:n for n in pt.body if isinstance(n,ast.FunctionDef)}
            assert {'native_pair','auxiliary_pair','backend_and_full_batch','roundtrip_and_invalids','exact_tree','cache_and_causality'}<=set(fn)
            calls=[ast.unparse(n.func) for n in ast.walk(pt) if isinstance(n,ast.Call)]
            forbidden={'torch.load','torch.save','torch.Generator','torch.rand','torch.randn','torch.randint'}
            assert not forbidden.intersection(calls)
            assert calls.count('complete_step')==4  # two sites native, two sites in two-arm loop =>6 updates
            assert calls.count('torch.nn.utils.clip_grad_norm_')==1
            assert calls.count('optimizer.step')==1
            assert [ast.unparse(n) for n in ast.walk(fn['ids']) if isinstance(n,ast.BinOp) and isinstance(n.op,ast.Mod)]==['(211 + 173 * b + 37 * j + 19 * j * j) % 50257']
            checked=fn['checked'];order=[ast.unparse(n.func) for n in ast.walk(checked) if isinstance(n,ast.Call)]
            assert order.index('publish')<order.index('all')
            cache_text=ast.get_source_segment(proof.read_text(),fn['cache_and_causality'])
            assert 'x[:,:61]' in cache_text and 'x[:,61:]' in cache_text and '(1,8)' in cache_text and '196608*batch' in cache_text
            assert '3e-5,3e-5' in cache_text and "for reset in (0,1)" in cache_text
            joint=fn['auxiliary_pair'];jc=[ast.unparse(n.func) for n in ast.walk(joint) if isinstance(n,ast.Call)]
            assert jc.count('joint_reference')==1 and jc.count('sequential_backward')==1
            assert '4e-6,4e-5' in ast.get_source_segment(proof.read_text(),joint)
            backend=ast.get_source_segment(proof.read_text(),fn['backend_and_full_batch'])
            assert "del cpu,m" in backend and backend.index('del cpu,m')<backend.index("cache_and_causality(mps,device)")
            assert "sequence=ids(8,128,'mps')" in backend and 'complete_step' not in backend
            assert '4e-5,4e-4' in backend
            rejects=[]
            for n in ast.walk(fn['roundtrip_and_invalids']):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('reject','modified') and isinstance(n.args[0],ast.Constant):rejects.append(n.args[0].value)
            # One literal reject(label,...) inside a three-element tensor loop.
            assert len(rejects)==36 and plan['negative_cases_each_arm']==39
            assert len(set(rejects))==len(rejects)
            imports=[n for n in ast.walk(ct) if isinstance(n,(ast.Import,ast.ImportFrom))]
            names=[n.module if isinstance(n,ast.ImportFrom) else a.name for n in imports for a in (n.names if isinstance(n,ast.Import) else [None])]
            assert not any(n and n.split('.')[0] in ('torch','numpy') for n in names)
            assert 'research.recorded_role_scan_v1' in names
            cp=json.loads((ROOT/plan['reviewed_plans'][1]['path']).read_text())
            assert set(cp['roles_drawn'])=={'initializer'} and cp['initializer']==cp['roles_drawn']['initializer']
            pp=json.loads((ROOT/plan['reviewed_plans'][0]['path']).read_text())
            assert pp['scope']['optimizer_updates']==6 and pp['scope']['full_B8T128_backwards']==3
            assert pp['predicates']==plan['predicates']
            assert pp['forecast']['rss_bytes']<4*1024**3 and pp['forecast']['mps_bytes']<6*1024**3
            assert pp['complete_target_bytes']==42*1024**2+10*1024**3
            for name,h in plan['new_execution_sources'].items():assert sha(ROOT/name)==h,name
            result=dict(status='complete_static_generated_proof_and_census_source_review_runtime_unqualified',
                new_execution_sources=plan['new_execution_sources'],proof_and_census_plan_pins=plan['reviewed_plans'],
                literal_rejection_labels=sorted(rejects),negative_cases_each_arm=39,
                no_torch_numpy_model_optimizer_rng_census_or_fixture_execution=True,
                no_quality_or_numerical_admission=True,next='Run new census once; independent result/resource admission before generated model proof')
            publish(R,'review.json',result)
            for ref in plan['reviewed_plans']:
                folder=(ROOT/ref['path']).parent
                assert set(p.name for p in folder.iterdir())=={'plan.json'}
                publish(folder,'source-review.json',dict(status='source_reviewed_runtime_unqualified',review_sha256=sha(R/'review.json'),review_plan_sha256=arg.plan_sha))
        publish(R,'STOP',dict(status='complete_original_generated_source_review',no_retry=True,no_replay=True))
    except BaseException as error:
        if not (R/'STOP').exists():publish(R,'STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
        raise


if __name__=='__main__':main()
