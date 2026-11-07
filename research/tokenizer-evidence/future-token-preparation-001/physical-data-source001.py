"""NEW static review of standalone physical data-view source; no module execution."""
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


def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=2*1024**2
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def called(node):
    return [(n,ast.unparse(n.func)) for n in ast.walk(node) if isinstance(n,ast.Call)]


def imports(node):
    result=[]
    for n in ast.walk(node):
        if isinstance(n,ast.Import):result.extend((n,a.name) for a in n.names)
        if isinstance(n,ast.ImportFrom):result.append((n,n.module))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json'}
    plan=json.loads((R/'plan.json').read_text());assert shutil.disk_usage(ROOT).free>=plan['complete_target_bytes']
    publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
    try:
        with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules and 'tiktoken' not in sys.modules
            for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
            source=ROOT/plan['data_view_source'];tree=ast.parse(source.read_text());compile(tree,str(source),'exec')
            fns={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
            classes={n.name:n for n in tree.body if isinstance(n,ast.ClassDef)}
            assert {'canonical','ids_digest','reference_spec','decode','tokenizer_contract'}<=set(fns)
            assert set(classes)=={'Decoded','View'}
            permitted={'copy','dataclasses','hashlib','importlib.metadata','json','pathlib','torch','tiktoken','research.future_token_sampling_v1'}
            assert all(name in permitted for _,name in imports(tree))
            assert not [n for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)) and n in [node for node,name in imports(tree) if name in ('torch','tiktoken','research.future_token_sampling_v1')]]
            statements=[n for n in tree.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef,ast.Import,ast.ImportFrom))]
            top_calls=[name for n in statements for _,name in called(n)]
            assert all(name in {'Path','Path(__file__).resolve'} for name in top_calls)
            cs=called(tree)
            assert not [name for _,name in cs if name in {'torch.Generator','torch.load','torch.save','eval','exec'}
                        or name.rsplit('.',1)[-1] in {'manual_seed','randint','rand','randn','unlink','write_bytes','write_text'}]
            decoding=fns['decode'];encodes=[n for n,name in called(decoding) if name=='enc.encode']
            assert len(encodes)==2
            bounds=[]
            for n in encodes:
                assert len(n.args)==1 and isinstance(n.args[0],ast.Subscript) and isinstance(n.args[0].value,ast.Name) and n.args[0].value.id=='text'
                sl=n.args[0].slice;assert isinstance(sl,ast.Slice) and sl.step is None
                bounds.append((None if sl.lower is None else ast.unparse(sl.lower),None if sl.upper is None else ast.unparse(sl.upper)))
                assert {k.arg:ast.literal_eval(k.value) for k in n.keywords}=={'disallowed_special':()}
            assert bounds==[(None,'a'),('a','b')]
            assert not [n for n in ast.walk(tree) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='text'
                        and isinstance(n.slice,ast.Slice) and isinstance(n.slice.lower,ast.Name) and n.slice.lower.id=='b']
            assert len([n for n,name in called(decoding) if name=='_read_corpus'])==2
            assert not [name for _,name in called(decoding) if name in {'Sampler','View'}]
            assert len([n for n,name in called(tree) if name=='Sampler'])==1
            view_methods={n.name:n for n in classes['View'].body if isinstance(n,ast.FunctionDef)}
            assert set(view_methods)=={'__init__','validate','next_sequence','state_dict','load_state_dict'}
            init=view_methods['__init__'];sampler_call=next(n for n,name in called(init) if name=='Sampler')
            decoded_validation=next(n for n,name in called(init) if name=='decoded.validate')
            assert decoded_validation.lineno<sampler_call.lineno
            assert len([n for n,name in called(view_methods['next_sequence']) if name=='self.sampler.next'])==1
            assert len([n for n,name in called(view_methods['load_state_dict']) if name=='self.sampler.load_state_dict'])==1
            canonical=next(n for n,name in called(fns['canonical']) if name=='json.dumps')
            options={k.arg:ast.literal_eval(k.value) for k in canonical.keywords}
            assert options==dict(sort_keys=True,ensure_ascii=True,separators=(',',':'),allow_nan=False)
            original=ast.parse((ROOT/plan['id_digest_reference']).read_text())
            digest=next(n for n in original.body if isinstance(n,ast.FunctionDef) and n.name=='digest')
            old_dump=next(n for n,name in called(digest) if name=='json.dumps')
            assert options=={k.arg:ast.literal_eval(k.value) for k in old_dump.keywords}
            reference=ROOT/plan['physical_manifest']['path']
            assert sha(reference)==plan['physical_manifest']['sha256']
            constants={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                and isinstance(n.targets[0],ast.Name) and n.targets[0].id in {'POLICY','REFERENCE_SHA256'}}
            assert constants['REFERENCE_SHA256']==plan['physical_manifest']['sha256']
            cap=next(n.value for n in tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id=='CORPUS_CAP')
            assert ast.dump(cap,include_attributes=False)==ast.dump(ast.parse('8*1024**2',mode='eval').body,include_attributes=False)
            data=json.loads(reference.read_text())['data']
            scalar_inventory=dict(train_tokens=data['train_tokens'],val_tokens=data['validation_tokens'],
                train_eligible=data['train_tokens']-513,train_last=data['train_tokens']-514,
                val_eligible=data['validation_tokens']-129,val_last=data['validation_tokens']-130,
                physical_targets=data['validation_targets'],physical_target_bytes=data['validation_target_bytes'])
            assert scalar_inventory==plan['independent_scalar_inventory']
            assert (data['raw_characters'],data['raw_train_end'],data['raw_val_end'])==(1115394,892315,1003854)
            assert len(data['validation_window_target_counts'])==268 and data['validation_window_target_counts'][-1]==102
            assert plan['future_source_bindings']==1245 and plan['predecessor_bindings']==1243
            for path,h in plan['sources'].items():assert sha(ROOT/path)==h,path
            assert 'torch' not in sys.modules and 'numpy' not in sys.modules and 'tiktoken' not in sys.modules
            publish('review.json',dict(status='complete_standalone_physical_data_view_SOURCE_review_runtime_unqualified',sources=plan['sources'],
                encoder_slices=bounds,independent_scalar_inventory=scalar_inventory,digest_JSON_options_match_original=True,
                physical_manifest=plan['physical_manifest'],lazy_Torch_tiktoken_sampler_only=True,
                no_data_module_functions_extraction_Torch_tokenizer_sampler_corpus_or_RNG_calls=True,
                no_new_role_or_model_or_science_admission=True,scope='source_AST_and_original_metadata_only'))
        publish('STOP',dict(status='complete_original_physical_data_source_review',no_retry=True,no_replay=True))
    except BaseException as error:
        if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',error=repr(error),no_retry=True,no_replay=True))
        raise


if __name__=='__main__':main()
