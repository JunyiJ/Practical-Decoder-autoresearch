"""Independent literal plan and AST source admission; no model/codebook imports."""
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
def need(value,reason):
    if not value:raise ValueError(reason)
def save(out,name,value):
    data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    need(sum(p.stat().st_size for p in out.iterdir())+len(data)<2097152,'Whole source metadata cap')
    with (out/name).open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
def checkpins(pins):
    for pin in pins:
        raw=Path(pin['path']).read_bytes();need(len(raw)==pin['size'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'Exact source/metadata pin')
def plan_review():
    p=json.loads((BASE/'model-source-plan001.json').read_text());n=p['counts'];c=p['codes']
    need(p['variants']==['native','nonlexical_codes','raw_bytes'] and p['no_TOP_auxiliary_loss'] is True,'Only three fixed primary-CE variants')
    need(n==dict(native_parameters=28492416,native_tensors=88,native_Muon=44,native_AdamW=44,composed_parameters=28492416-12865792+725235,composed_tensors=88-1+18,composed_Muon=44+9,composed_AdamW=44-1+9),'Full own parameter/group arithmetic')
    need(c['packed_u16_codes_bytes']==50257*130*2 and c['packed_u16_lengths_bytes']==50257*2 and c['int64_code_table_bytes']==50257*130*8 and c['int64_lengths_bytes']==50257*8,'Complete packed/staged lookup sizes')
    need(p['codebook_round_calls_each_nonlexical_build']==50256*16,'Full forward+inverse Feistel count')
    traffic=p['codebook_hash_traffic_training_UNMEASURED'];need(traffic['three_validations_per_step_bytes']==3*(c['int64_code_table_bytes']+c['int64_lengths_bytes']) and traffic['1200_step_bytes']==1200*traffic['three_validations_per_step_bytes'],'Preflight copy/hash traffic counted')
    need(p['prospective_resources']['lookup_allowance']>=c['int64_code_table_bytes']+c['int64_lengths_bytes'] and p['prospective_resources']['no_numerical_or_science_execution_admission'] is True,'Resources forecast only')
    accepted=json.loads((BASE/'component-math-review002/review.json').read_text())
    need(accepted['accepted'] is True and accepted['all_coordinate_comparisons']==8706916,'Accepted finite component consumer prerequisite only')
    need(json.loads((BASE/'component-math-review002/guard.json').read_text())['status']=='finished' and json.loads((BASE/'component-math-review002/STOP').read_text())['status']=='complete_original_saved_review','Original prerequisite fully closed')
    need(p['initializer_roles'].startswith('UNASSIGNED') and len(p['source_files'])==5,'Minimal sources, no roles drawn')
    return p

def source_review(p):
    files=[ROOT/x for x in p['source_files']]
    need(all(x.is_file() and x.stat().st_size<=262144 for x in files) and sum(x.stat().st_size for x in files)<524288,'Source payload bound')
    trees={x.name:ast.parse(x.read_text()) for x in files};text={x.name:x.read_text() for x in files}
    for tree in trees.values():
        need(not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in {'eval','exec','compile','__import__'} for n in ast.walk(tree)),'No dynamic extraction or execution')
    imports=set()
    for n in ast.walk(trees['codebook.py']):
        if isinstance(n,ast.Import):imports.update(x.name for x in n.names)
        elif isinstance(n,ast.ImportFrom):imports.add(n.module)
    need(imports=={'array','dataclasses','hashlib','json','sys'},'Stdlib codebook only')
    cb=text['codebook.py']
    for marker in ['def permute','def inverse','range(8)','range(7, -1, -1)','shake_256','50256','50257','130','_FACTORY_TOKEN','codes_i64_sha256','lengths_i64_sha256','Duplicate encoder key','Full public merge agreement','Exact native vocabulary identity','Canonical fixed byte codes','inverse(value, length, private_key) == ordinal']:
        need(marker in cb,'Complete codebook/control source predicate: '+marker)
    need('open(' not in cb and 'tiktoken' not in cb and 'torch' not in cb,'No hidden asset/tokenizer/Torch entry')
    contract=text['contract.py'];tree=trees['contract.py'];native=None
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NATIVE_MODEL_JSON' for t in n.targets):native=json.loads(ast.literal_eval(n.value))
    need(native==p['native_config'],'True canonical native initializer model cfg')
    need('native_initializer_recipe' in contract and 'input_embedding_kind' in contract,'Truthful replacement identity/config')
    cls=next(n for n in trees['model.py'].body if isinstance(n,ast.ClassDef) and n.name=='ByteInputModel')
    methods={n.name:ast.unparse(n) for n in cls.body if isinstance(n,ast.FunctionDef)}
    init=methods['__init__'];forward=methods['forward_train']
    need(init.index('native = Native(C.native_config())')<init.index('ByteComposer('),'Genuine native draws first')
    need('self.blocks = native.transformer.h' in init and 'self.readout = native.lm_head' in init and 'self.final_norm = native.transformer.ln_f' in init,'Own retained common modules')
    need('self.native = native' in init and "spec.variant == 'native'" in init and 'self.composer = ByteComposer' in init,'Native branch and independent composer')
    need('self.native' not in methods['_composed_hidden'] and 'self.composer(' in methods['_composed_hidden'] and 'block(hidden, None, 0)' in methods['_composed_hidden'],'One native-shaped body path, no native table access')
    need('self.native(inputs, labels)' in forward and 'F.cross_entropy' in forward and 'auxiliary' not in forward and '/ 4.0' in forward,'Only ordinary primary CE/unchanged readout scale')
    need(forward.index('self.validate()')<forward.index('self._composed_hidden') and forward.index('self._ids(labels')<forward.index('self._composed_hidden'),'Full input preflight before graph')
    need('self._buffer_digest' in methods['validate'] and 'tensor.to(device=\'cpu\')' not in methods['_buffer_digest'],'Streaming buffer preflight source')
    for marker in ['131072',"to(device='cpu')",'torch.uint8','_own_buffer_bindings','_own_expected_buffer_hashes','_semantic_inventory','Aliased parameter/buffer storage','Codebook bytes differ']:
        need(marker in text['model.py'],'Full immutable buffer/structure guard: '+marker)
    for method in ['forward','state_dict','load_state_dict']:
        need('raise ValueError' in methods[method],'Unsupported general/cache/state paths reject')
    need('detach(' not in methods['_composed_hidden'] and 'register_hook' not in text['model.py'],'No hidden derivative alteration')
    opt=text['optimizer.py']
    for marker in ['transformer.h.','composer.filters.','composer.highway_weight','composer.transform_weight','composer.projection_weight','match_rms_adamw','eps=1e-6','(.9, .999)','require_gradients=True','1. - .00006','C.counts','All finite dense gradients']:
        need(marker in opt,'Exact own optimizer policy: '+marker)
    need(opt.index('self.validate(require_gradients=True)')<opt.index('parameter.mul_')<opt.index('optimizer.step()'),'All preflight before decay and either child step')
    return dict(status='complete_NEW_byte_input_model_SOURCE_only',accepted=True,source_files=p['source_files'],no_model_component_codebook_Torch_import_or_extracted_functions=True,roles='UNASSIGNED',scope='Admits new codebook/actual-model finite proof SOURCES only, no functional execution/science')

def main(mode,expected):
    need(mode in ['plan','source'],'Declared original review mode')
    out=BASE/('model-source-plan-review001' if mode=='plan' else 'model-source001')
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need({p.name for p in out.iterdir()}=={'plan.json'},'One original source review')
        save(out,'attempt.json',dict(pid=os.getpid(),mode=mode,plan_sha256=expected,no_replay=True,no_retry=True))
        try:
            with CpuPrerequisiteGuard(out/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                raw=(out/'plan.json').read_bytes();need(hashlib.sha256(raw).hexdigest()==expected,'Review plan SHA');outer=json.loads(raw);checkpins(outer['pins'])
                p=plan_review()
                result=source_review(p) if mode=='source' else dict(status='complete_NEW_byte_input_SOURCE_plan_only',accepted=True,no_candidate_code_import_or_function_execution=True,scope='Five source modules only; no functional admission')
                if mode=='source':
                    need(json.loads((BASE/'model-source-plan-review001/guard.json').read_text())['status']=='finished' and json.loads((BASE/'model-source-plan-review001/STOP').read_text())['status']=='complete_original_source_review','Original plan review first')
                save(out,'review.json',result)
            save(out,'STOP',dict(status='complete_original_source_review',no_replay=True,no_retry=True))
        except BaseException as exc:save(out,'STOP',dict(status='closed_failed_original_source_review',error=repr(exc),no_replay=True,no_retry=True));raise
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
