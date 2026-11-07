"""AST/literal-only component source admission; never import the component."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'component-source001'
def need(value,reason):
    if not value:raise ValueError(reason)
def save(name,value):
    payload=(json.dumps(value,sort_keys=True)+'\n').encode()
    need(sum(p.stat().st_size for p in OUT.iterdir())+len(payload)<2097152,'Metadata cap')
    with (OUT/name).open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need({p.name for p in OUT.iterdir()}=={'plan.json'},'Original source gate only')
        save('attempt.json',dict(pid=os.getpid(),plan_sha256=expected,no_retry=True,no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.1):
                raw=(OUT/'plan.json').read_bytes();need(hashlib.sha256(raw).hexdigest()==expected,'Plan SHA')
                plan=json.loads(raw)
                for pin in plan['pins']:
                    data=(ROOT/pin['path']).read_bytes();need(len(data)==pin['size'] and hashlib.sha256(data).hexdigest()==pin['sha256'],'Source/design pin')
                review=json.loads((BASE/'component-plan-review001/review.json').read_text())
                need(review['accepted'] is True and json.loads((BASE/'component-plan-review001/guard.json').read_text())['status']=='finished' and json.loads((BASE/'component-plan-review001/STOP').read_text())['status']=='complete_original_source_plan_review','Independent source-cost plan complete first')
                source=(ROOT/'research/byte_composer_v1.py').read_text();tree=ast.parse(source)
                imports=set()
                for n in ast.walk(tree):
                    if isinstance(n,ast.Import):imports.update(a.name for a in n.names)
                    elif isinstance(n,ast.ImportFrom):imports.add(n.module)
                need(imports=={'math','torch','torch.nn'},'Minimal prospective math dependencies')
                need(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'exec','eval','compile','__import__'} for n in ast.walk(tree)),'No dynamic extraction/execution')
                cls=next(n for n in tree.body if isinstance(n,ast.ClassDef));methods={n.name:ast.unparse(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
                need(set(methods)=={'__init__','identity','validate','forward','state_dict','load_state_dict'},'Only component math API')
                init=methods['__init__'];forward=methods['forward'];validate=methods['validate']
                for item in ['torch.Generator','manual_seed(private_seed)','torch.empty','generator=generator','nn.ParameterList','-2.05','-1.95','math.sqrt(525)']:
                    need(item in init,'Private initialization source contract: '+item)
                for item in ['725235','18','torch.float32','torch.isfinite','data_ptr','requires_grad','is_leaf','No hidden']:
                    need(item in validate,'Full parameter preflight: '+item)
                need(forward.index('self.validate()')<forward.index('F.embedding') and forward.index('Special appears alone')<forward.index('F.embedding'),'Full structural preflight before differentiable lookup')
                for item in ['1024','130','F.pad','value=256','masked_fill','unfold','contiguous','15 * width','torch.tanh','-2.0','amax','torch.where','torch.cat','F.relu','torch.sigmoid','1.0 - gate','self.projection_weight']:
                    need(item in forward,'Declared one-graph equation/padding: '+item)
                need('raise ValueError' in methods['state_dict'] and 'raise ValueError' in methods['load_state_dict'],'Unqualified state operations reject')
                need('detach(' not in source and 'register_hook' not in source and 'backward(' not in source,'No hidden gradient alteration or execution')
                save('review.json',dict(status='complete_component_AST_SOURCE_only',accepted=True,no_component_Torch_import_or_extracted_function_execution=True,parameters_SOURCE_predicate_only=725235,admits='New finite proof SOURCES only; no initializer role or functional execution admitted'))
            save('STOP',dict(status='complete_original_component_source_gate',no_retry=True,no_replay=True))
        except BaseException as exc:save('STOP',dict(status='closed_failed_original_component_source_gate',error=repr(exc),no_retry=True,no_replay=True));raise
if __name__=='__main__':main(sys.argv[1])
