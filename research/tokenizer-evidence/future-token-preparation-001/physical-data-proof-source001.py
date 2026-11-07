"""NEW static gate for physical data-view proof/census; executes no component."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
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
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules and 'tiktoken' not in sys.modules
            assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
            for name,h in plan['sources'].items():assert sha(ROOT/name)==h,name
            configs=[]
            for ref in plan['plans']:
                assert sha(ROOT/ref['path'])==ref['sha256'];configs.append(json.loads((ROOT/ref['path']).read_text()))
            proof,census=configs
            assert len(proof['negative_labels'])==len(set(proof['negative_labels']))==24
            assert proof['scope_counts']==dict(actual_view_calls=6,actual_anchor_draws=30,final_calls_each={'train':2,'val':2},encoder_calls=2)
            assert set(census['roles'])=={'train','val'} and len(set(census['roles'].values()))==2 and all(type(v) is int and 0<=v<2**63 for v in census['roles'].values())
            assert proof['complete_target_bytes']==census['complete_target_bytes']==38*1024**2+128*1024+10*1024**3
            assert proof['limits']['CPU_seconds']==120 and proof['limits']['rss_bytes']==4*1024**3 and proof['limits']['reserve_bytes']==8*1024**3
            assert proof['forecast']['rss_bytes']<4*1024**3 and proof['forecast']['seconds']<120
            assert proof['torch_version']=='2.14.1' and proof['tiktoken_version']=='0.14.0'
            tree=ast.parse((ROOT/plan['proof_source']).read_text());compile(tree,plan['proof_source'],'exec')
            ct=ast.parse((ROOT/plan['census_source']).read_text());compile(ct,plan['census_source'],'exec')
            functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
            main_fn=functions['main']
            guard=next(n for n in ast.walk(main_fn) if isinstance(n,ast.With) and any(isinstance(x.context_expr,ast.Call) and ast.unparse(x.context_expr.func)=='CpuPrerequisiteGuard' for x in n.items))
            ti=[n for n in ast.walk(tree) if isinstance(n,ast.Import) and any(a.name=='torch' for a in n.names)]
            assert len(ti)==1 and ti[0] in list(ast.walk(guard))
            all_imports=[]
            for n in ast.walk(tree):
                if isinstance(n,ast.Import):all_imports.extend(a.name for a in n.names)
                if isinstance(n,ast.ImportFrom):all_imports.append(n.module)
            assert not [s for s in all_imports if s and s.startswith('research.') and s!='research' and s!='src.research.cpu_prerequisite_guard']
            for n in ast.walk(tree):
                if isinstance(n,ast.ImportFrom) and n.module=='research':assert [(a.name,a.asname) for a in n.names]==[('future_token_data_v1','D')]
            calls=[(n,ast.unparse(n.func)) for n in ast.walk(tree) if isinstance(n,ast.Call)]
            assert not [s for _,s in calls if s in {'torch.load','torch.save'} or s.startswith('torch.mps.') or s.endswith(('.backward','.step'))]
            check=functions['check'];order=check.body
            pub=next(i for i,n in enumerate(order) if isinstance(n,ast.Expr) and isinstance(n.value,ast.Call) and ast.unparse(n.value.func)=='publish')
            assert pub<next(i for i,n in enumerate(order) if isinstance(n,ast.Assert))
            assert any(isinstance(n,ast.AugAssign) and isinstance(n.target,ast.Name) and n.target.id=='SERIAL' for n in order)
            text=(ROOT/plan['proof_source']).read_text()
            assert "f'{SERIAL:03d}-{label}.json'" in text
            assert "patch.object(torch,'Generator'" in text and "patch.object(tiktoken.Encoding,'encode',observed)" in text and "patch.object(tiktoken.load,'read_file',no_fetch)" in text
            assert 'return genuine(encoder,value,*args,**kwargs)' in text and "seen['encoding'].decode(list(decoded.val_ids[1:]))" in text
            assert 'RNG' in text and 'D.decode()' in text
            for ref in proof['codec_cache']:
                assert Path(ref['path']).stat().st_size==ref['size'] and sha(ref['path'])==ref['sha256']
                assert ref['sha256'] in plan['known_codec_cache_hashes']
            assert len(proof['codec_cache'])==2
            for name,h in plan['sources'].items():assert sha(ROOT/name)==h,name
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules and 'tiktoken' not in sys.modules
            publish(R,'review.json',dict(status='complete_NEW_physical_data_proof_census_source_resource_gate',sources=plan['sources'],plans=plan['plans'],
                role_count=2,negative_count=24,scope_counts=proof['scope_counts'],codec_cache=proof['codec_cache'],
                genuine_encoder_spy_delegates=True,decode_generator_trap=True,network_fetch_blocked=True,
                no_Torch_tokenizer_sampler_corpus_RNG_or_proof_function_execution=True,
                scope='one_new_two_role_census_then_separately_admitted_physical_CPU_data_proof_only;no_model_or_scoring'))
            for ref in plan['plans']:
                folder=(ROOT/ref['path']).parent;assert set(p.name for p in folder.iterdir())=={'plan.json'}
                publish(folder,'source-review.json',dict(status='source_reviewed_original_physical_data_fixture',review_sha256=sha(R/'review.json'),review_plan_sha256=arg.plan_sha))
        publish(R,'STOP',dict(status='complete_original_physical_data_proof_source_gate',no_retry=True,no_replay=True))
    except BaseException as error:
        if not (R/'STOP').exists():publish(R,'STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
        raise


if __name__=='__main__':main()
