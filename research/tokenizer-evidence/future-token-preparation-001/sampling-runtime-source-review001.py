"""NEW independent CPU sampler integration source admission; no Torch or RNG."""
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
            for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
            loaded=[]
            for ref in plan['plans']:
                assert sha(ROOT/ref['path'])==ref['sha256']
                loaded.append(json.loads((ROOT/ref['path']).read_text()))
            runtime,census=loaded
            assert runtime['planned']['negative_cases']==29 and len(runtime['planned']['negative_labels'])==len(set(runtime['planned']['negative_labels']))==29
            assert runtime['data']==dict(train_length=1793,train_formula='(67*i+31)%50257',val_length=419,val_formula='(71*i+43)%50257')
            assert runtime['schedules']==dict(prefix=['train','val','train','val','train'],continuation=['val','train','train','val'])
            assert runtime['planned']['actual_sampler_calls']==13 and runtime['planned']['actual_anchor_draws']==62 and runtime['planned']['reference_anchor_draws']==42
            assert set(census['roles'])=={'train','val'} and len(set(census['roles'].values()))==2
            assert all(type(v) is int and 0<=v<2**63 for v in census['roles'].values())
            assert runtime['limits']['CPU_seconds']==120 and runtime['limits']['rss_bytes']==4*1024**3 and runtime['limits']['reserve_bytes']==8*1024**3
            assert runtime['forecast']['rss_bytes']<4*1024**3 and runtime['forecast']['seconds']<120
            assert runtime['complete_target_bytes']==census['complete_target_bytes']==36*1024**2+128*1024+10*1024**3
            trees={}
            for path in plan['syntax_sources']:
                text=(ROOT/path).read_text();tree=ast.parse(text);compile(tree,path,'exec');trees[path]=tree
            tree=trees[plan['runtime_source']]
            main_fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            cpu=next(n for n in ast.walk(main_fn) if isinstance(n,ast.With) and any(isinstance(x.context_expr,ast.Call) and ast.unparse(x.context_expr.func)=='CpuPrerequisiteGuard' for x in n.items))
            torch_import=[n for n in ast.walk(tree) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names)]
            assert len(torch_import)==1 and torch_import[0] in list(ast.walk(cpu))
            assert not [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) and ((isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names)) or (isinstance(n,ast.ImportFrom) and n.module and n.module.startswith('research.')))]
            functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
            check=functions['check'];statements=check.body
            call_statements=[ast.unparse(n.value.func) for n in statements if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call)]
            assert 'publish' in call_statements and next(i for i,n in enumerate(statements) if isinstance(n,ast.Assert))>next(i for i,n in enumerate(statements) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='publish')
            assert any(isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='SERIAL' for n in statements)
            text=(ROOT/plan['runtime_source']).read_text()
            assert "f'{SERIAL:03d}-{label}.json'" in text
            ct=trees[plan['census_source']]
            assert not [n for n in ast.walk(ct) if isinstance(n,ast.Import) and any(a.name in ('torch','numpy') for a in n.names)]
            sampler=trees['research/future_token_sampling_v1.py']
            cls=next(n for n in sampler.body if isinstance(n,ast.ClassDef) and n.name=='Sampler')
            methods={n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
            next_text=ast.get_source_segment((ROOT/'research/future_token_sampling_v1.py').read_text(),methods['next'])
            assert next_text.index('working=_copy_generator')<next_text.index('torch.randint')<next_text.index("self.generators[split].set_state")
            load_text=ast.get_source_segment((ROOT/'research/future_token_sampling_v1.py').read_text(),methods['load_state_dict'])
            assert load_text.index('probe.set_state')<load_text.index('self.generators=staged')
            for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules
            publish(R,'review.json',dict(status='complete_NEW_independent_CPU_sampler_source_case_resource_gate',sources=plan['sources'],
                role_count=2,negative_cases=29,report_numeric_prefix=True,
                no_torch_numpy_sampler_rng_model_corpus_or_case_function_execution=True,
                closed_index_and_model_failures_remain_unqualified=True,scope='one_original_new_sampler_census_then_separate_CPU_runtime_admission_only'))
            for ref in plan['plans']:
                folder=(ROOT/ref['path']).parent
                assert set(p.name for p in folder.iterdir())=={'plan.json'}
                publish(folder,'source-review.json',dict(status='source_reviewed_independent_CPU_sampler_fixture_only',review_sha256=sha(R/'review.json'),review_plan_sha256=arg.plan_sha))
        publish(R,'STOP',dict(status='complete_original_CPU_sampler_source_review',no_retry=True,no_replay=True))
    except BaseException as error:
        if not (R/'STOP').exists():publish(R,'STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
        raise


if __name__=='__main__':main()
