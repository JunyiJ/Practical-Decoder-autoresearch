"""Independent AST admission / census runtime admission / saved array review.

Never imports candidate, oracle, harness, Torch, tokenizer or model code.
"""
from array import array
import ast
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zlib
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
TOLS=(4e-6,4e-5)
def need(value,reason):
    if not value:raise ValueError(reason)
def digest(raw):return hashlib.sha256(raw).hexdigest()
def write(out,name,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    need(sum(p.stat().st_size for p in out.iterdir())+len(raw)<2097152,'Review whole metadata cap')
    with (out/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def checkpins(pins):
    for p in pins:
        raw=Path(p['path']).read_bytes();need(len(raw)==p['size'] and digest(raw)==p['sha256'],'Complete source/runtime pin')
def literals(path):
    tree=ast.parse(path.read_text());result={}
    for n in tree.body:
        if isinstance(n,ast.Assign) and len(n.targets)==1 and isinstance(n.targets[0],ast.Name):
            try:result[n.targets[0].id]=ast.literal_eval(n.value)
            except (ValueError,TypeError):pass
    return tree,result

def source(plan):
    proof=BASE/'component-math001.py';oracle=BASE/'component_oracle001.py';census=BASE/'component-census001.py'
    trees={p.name:literals(p) for p in [proof,oracle,census]}
    cases=trees[oracle.name][1]['CASES'];names=[x[0] for x in cases]
    need(len(names)==len(set(names))==6 and max(len(s) for x in cases for s in x[1])==130,'Six exact full-shape cases and maximum token')
    need({s for x in cases for row in x[1] for s in row}>={0,32,255,257,258,259},'Required byte/special/boundary cases')
    pc=trees[proof.name][1];neg=pc['NEGATIVES'];extra=pc['CPU_ONLY']
    need(len(neg)==len(set(neg))==52 and len(extra)==9 and not set(extra)&set(neg),'Exact unique negative census')
    text=proof.read_text();ost=oracle.read_text();cst=census.read_text()
    need('ATOL, RTOL = 4e-6, 4e-5' in text and pc['CAP']==268435456,'Fixed coordinate gates/output cap')
    imports=set()
    for n in ast.walk(trees[oracle.name][0]):
        if isinstance(n,ast.Import):imports.update(x.name for x in n.names)
        elif isinstance(n,ast.ImportFrom):imports.add(n.module)
    need(imports=={'array','math','struct'},'Independent scalar oracle only')
    for fragment in ['math.fsum','math.tanh','math.exp','gradients','tie_channels','len(ties[c])','pre[r] > 0','struct.pack']:
        need(fragment in ost,'Analytic represented-literal oracle: '+fragment)
    main=next(n for n in trees[proof.name][0].body if isinstance(n,ast.FunctionDef) and n.name=='main')
    normalized=ast.unparse(main)
    need(normalized.index('with WholeGuard')<normalized.index('import component_oracle001')<normalized.index('import torch'),'Whole pre-Torch startup guard first')
    need(normalized.index('Independent complete runtime admission first')<normalized.index('import torch'),'Runtime admission required before Torch')
    need(text.index("publish(backend+'-'+name+'.json',report)") < text.index('need(init_rng_preserved'), 'Durable full coordinate report before numerical assertion')
    need('before==after' in text and 'T.autograd.grad' in text and 'initial_cpu' in text and 'T.random.set_rng_state' in text and 'T.mps.set_rng_state' in text,'Caller gradient/RNG preservation')
    need(".to(device='cpu').to(dtype=T.float64)" in text and 'MPS FP64' in text,'CPU before FP64 report')
    need('os.O_' not in cst and 'path.open(\'rb\')' in cst and 'pattern.finditer' in cst and 'pending<=m.start()<boundary' in cst,'Read-only streaming census with boundary overlap')
    need('import torch' not in cst and 'UNUSED_NOT_DRAWN' in cst,'No role draw by census')
    consumer=(BASE/'component-math001/plan.json').read_bytes();cp=(BASE/'component-census001/plan.json').read_bytes()
    need(digest(consumer)==plan['numerical_plan_sha256'] and digest(cp)==plan['census_plan_sha256'],'Fresh complete plan pins')
    numerical=json.loads(consumer);cplan=json.loads(cp)
    need(numerical['initializer']==cplan['initializer'] and cplan['no_model_numerical_execution_admission'] is True,'One exclusive role proposal only')
    need(numerical['counts']==dict(positives=12,invalids=113,parameter_tensors=18,parameter_elements=725235,reports_total=150),'Exact case/report plan census')
    need(numerical['limits']==dict(whole_seconds=600,CPU_phase_seconds=120,RSS_bytes=4294967296,MPS_bytes=6442450944,reserve_bytes=8589934592,output_bytes=268435456,entry_bytes=11005853696),'Full original hard resource gate')
    return dict(status='complete_NEW_component_proof_and_census_SOURCE_only',accepted=True,numerical_plan_sha256=plan['numerical_plan_sha256'],census_plan_sha256=plan['census_plan_sha256'],case_names=names,negative_families=52,CPU_only_families=9,no_candidate_oracle_Torch_import_or_extracted_functions=True,scope='Admits original stdlib text census only; numeric entry still requires accepted census and runtime receipt')

def admission(plan):
    raw=(BASE/'component-math001/plan.json').read_bytes();need(digest(raw)==plan['numerical_plan_sha256'],'Numerical plan unchanged')
    p=json.loads(raw);checkpins(p['execution_sources'])
    src=json.loads((BASE/'component-proof-source001/review.json').read_text())
    need(src['accepted'] is True and src['numerical_plan_sha256']==plan['numerical_plan_sha256'],'Original source gate')
    for folder,status in [('component-proof-source001','complete_original_source'),('component-census001','complete_original_exclusive_initializer_census')]:
        need(json.loads((BASE/folder/'guard.json').read_text())['status']=='finished' and json.loads((BASE/folder/'STOP').read_text())['status']==status,'Original gate/census fully complete')
    census=json.loads((BASE/'component-census001/result.json').read_text())
    need(census['initializer']==p['initializer'] and census['prior_matches']==[] and census['role_status']=='RESERVED_EXCLUSIVE_COMPONENT_MATH001_UNUSED_NOT_DRAWN','Accepted original unused exclusive constructor initializer')
    need(os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize>=11005853696,'Complete current entry capacity')
    return dict(status='complete_NEW_byte_component_runtime_admission',accepted=True,numerical_plan_sha256=plan['numerical_plan_sha256'],initializer=p['initializer'],source_pins=len(p['execution_sources']),scope='One original guarded finite component proof only; no model/state/science launch')

def decode(record,kind,count):
    need(record['format']=='little_'+kind and record['count']==count,'Exact vector dtype/count')
    size=count*(8 if kind=='d' else 1)
    decoder=zlib.decompressobj();raw=decoder.decompress(base64.b64decode(record['zlib_base64'],validate=True),size+1)
    need(decoder.eof and not decoder.unconsumed_tail and not decoder.unused_data,'Bounded complete compressed vector')
    need(len(raw)==count*(8 if kind=='d' else 1) and digest(raw)==record['sha256'],'Complete saved vector hash/shape')
    v=array(kind);v.frombytes(raw)
    if sys.byteorder!='little':v.byteswap()
    return v

def saved(plan):
    root=BASE/'component-math001';p=json.loads((root/'plan.json').read_text());names=plan['case_names']
    need({x.name for x in root.iterdir()}==set(plan['expected_files']) and json.loads((root/'STOP').read_text())['status']=='complete_original_finite_component_math','Exact complete original report census')
    need(json.loads((root/'whole-guard.json').read_text())['status']=='finished','Whole pre-Torch guard finished')
    for path in root.glob('*guard.json'):
        need(json.loads(path.read_text())['status']=='finished','Every CPU phase guard finished')
    coordinates=0;forward={};expected_hashes={}
    for backend in ['cpu','mps']:
        for name in names:
            r=json.loads((root/(backend+'-'+name+'.json')).read_text())
            need(r['id']==backend+'-'+name and r['preserved'] is True and r['before']==r['after'] and r['constructor_global_RNG_preserved'] is True and r['parameter_tensor_count']==18 and r['parameter_elements']==725235,'Full positive inventory/preservation')
            need(set(r['comparisons'])==set(plan['shapes'])|{'output'},'Every forward and all parameter gradients saved')
            for key,c in r['comparisons'].items():
                shape=c['shape'];count=math.prod(shape)
                if key!='output':need(shape==plan['shapes'][key],'Exact parameter gradient shape')
                else:need(shape==plan['output_shapes'][name],'Exact original output shape')
                need((c['atol'],c['rtol'])==TOLS,'Unchanged elementwise tolerances')
                a=decode(c['actual'],'d',count);e=decode(c['expected'],'d',count);mask=decode(c['mask'],'B',count)
                flags=[math.isfinite(x) and math.isfinite(y) and abs(x-y)<=4e-6+4e-5*abs(y) for x,y in zip(a,e)]
                need(list(mask)==flags and all(flags) and c['all_pass'] is True and c['max_abs']==max(abs(x-y) for x,y in zip(a,e)),'Recomputed ALL saved coordinate predicates')
                coordinates+=count
                if key=='output':forward[backend,name]=list(a)
                if backend=='cpu':expected_hashes[name,key]=c['expected']['sha256']
                else:need(expected_hashes[name,key]==c['expected']['sha256'],'Same independent represented oracle across backends')
            if backend=='mps':need(all(abs(a-b)<=3e-5 for a,b in zip(forward['cpu',name],forward['mps',name])) and r['CPU_MPS_pair']=={'atol':3e-5,'rtol':0.,'all_coordinates':True},'All CPU/MPS output coordinates')
    count=0
    for path in root.glob('*-invalid-*.json'):
        if path.name.endswith('-guard.json'):continue
        r=json.loads(path.read_text());need(r['ValueError'] is not None and r['before']==r['after'] and r['preserved'] is True,'All invalid transactional preservation');count+=1
    need(count==113,'All exact backend negative reports')
    result=json.loads((root/'result.json').read_text());need(result['caller_CPU_MPS_RNG_restored'] is True and result['parameter_gradient_coordinates']==12*725235,'Final original acceptance receipt')
    return dict(status='accepted_original_finite_component_math_only',accepted=True,all_coordinate_comparisons=coordinates,invalid_transactions=count,no_Torch_component_oracle_replay=True,scope='Finite literal component only; no trained model, optimizer/state/cache/scorer/quality qualification')

def main(mode,expected):
    need(mode in ['source','admit','saved'],'Declared original review')
    folder={'source':'component-proof-source001','admit':'component-runtime001','saved':'component-math-review001'}[mode];out=BASE/folder
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need({p.name for p in out.iterdir()}=={'plan.json'},'One original review command')
        write(out,'attempt.json',dict(pid=os.getpid(),mode=mode,plan_sha256=expected,no_retry=True,no_replay=True))
        try:
            with CpuPrerequisiteGuard(out/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                raw=(out/'plan.json').read_bytes();need(digest(raw)==expected,'Review plan pin');p=json.loads(raw);checkpins(p['pins'])
                result={'source':source,'admit':admission,'saved':saved}[mode](p);write(out,'review.json',result)
            write(out,'STOP',dict(status={'source':'complete_original_source','admit':'complete_original_admission','saved':'complete_original_saved_review'}[mode],no_retry=True,no_replay=True))
        except BaseException as exc:write(out,'STOP',dict(status='closed_failed_original_review',error=repr(exc),no_retry=True,no_replay=True));raise
if __name__=='__main__':main(sys.argv[1],sys.argv[2])
