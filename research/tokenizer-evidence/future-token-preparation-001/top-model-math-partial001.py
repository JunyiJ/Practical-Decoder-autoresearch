"""FIRST NEW failed-TOP saved-evidence closure; never generates missing results.

Descriptive partial evidence only, no numerical/model qualification or replay.
"""
import array
import ast
import base64
import fcntl
import hashlib
import json
import math
import os
import shutil
from pathlib import Path
import sys
import zlib
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT = Path(__file__).resolve().parents[3]
R = Path(__file__).with_suffix('')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''): digest.update(chunk)
    return digest.hexdigest()


def write(name,value):
    raw = (json.dumps(value,sort_keys=True,allow_nan=False) + '\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file()) + len(raw) + 65536 < 2097152
    with (R/name).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def expand(value,count):
    assert 0 <= count <= 102926336
    decoder = zlib.decompressobj(); raw = decoder.decompress(base64.b64decode(value,validate=True),count+1)
    assert len(raw) == count and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail
    return raw


def dense(value,digest,shape):
    assert value['shape'] == shape and value['elements'] == math.prod(shape)
    assert value['dtype'] in ('torch.float32','torch.float64') and value['encoding'] == 'zlib-base64-little-endian'
    code,width = ('f',4) if value['dtype'] == 'torch.float32' else ('d',8)
    raw = expand(value['coordinates'],value['elements']*width)
    assert hashlib.sha256(raw).hexdigest() == digest == value['sha256']
    numbers = array.array(code); numbers.frombytes(raw)
    assert all(math.isfinite(x) for x in numbers)
    return numbers


def inspect(plan):
    assert sys.byteorder == 'little' and 'torch' not in sys.modules and 'numpy' not in sys.modules
    run = ROOT/plan['original_run']; numerical = json.loads((run/'plan.json').read_text())
    assert sha(run/'plan.json') == plan['original_plan_sha256']
    assert json.loads((run/'STOP').read_text())['status'] == 'closed_failed_NEW_TOP_model_math'
    assert json.loads((run/'outer-STOP').read_text())['status'] == 'closed_failed_original_NEW_TOP_outer_entry'
    original_review = ROOT/plan['original_complete_review']
    assert json.loads((original_review/'review.json').read_text())['accepted'] is False
    manifest = []; comparisons = []; nonlocal_counts = {'dense_coordinates':0,'mask_only_coordinates':0,'comparisons':0}
    def walk(value,path,file):
        if type(value) is dict:
            if 'all_coordinate_mask' in value:
                count = math.prod(value['shape']); assert count == value['coordinates'] and count <= 12865792
                mask = expand(value['all_coordinate_mask'],count); assert set(mask) <= {0,1}
                failures = mask.count(0); assert value['failed_coordinates'] == failures and value['passed'] == (failures == 0)
                tolerance = (value['atol'],value['rtol']); assert tolerance in ((0.,0.),(3e-5,0.),(4e-5,4e-4),(4e-6,4e-5))
                recomputed = 'actual' in value
                if recomputed:
                    a = dense(value['actual'],value['actual_sha256'],value['shape']);e = dense(value['expected'],value['expected_sha256'],value['shape'])
                    maximum = 0.; first = []
                    for i,(actual,expected) in enumerate(zip(a,e)):
                        error = abs(actual-expected);maximum = max(maximum,error)
                        passed = error <= tolerance[0] + tolerance[1]*abs(expected)
                        assert passed == bool(mask[i])
                        if not passed and len(first) < 64:first.append([i,actual,expected,error])
                    assert maximum == value['max_abs'] and first == value['first_64_failures']
                    nonlocal_counts['dense_coordinates'] += count
                else:
                    assert all(type(value[k]) is str and len(value[k]) == 64 for k in ('actual_sha256','expected_sha256'))
                    nonlocal_counts['mask_only_coordinates'] += count
                nonlocal_counts['comparisons'] += 1
                if failures:comparisons.append({'file':file,'path':path,'coordinates':count,'failures':failures,'max_abs':value['max_abs'],'atol':tolerance[0],'rtol':tolerance[1],'all_dense_coordinates_and_mask_recomputed':recomputed})
                return
            for key,child in value.items():walk(child,path+'/'+key,file)
        elif type(value) is list:
            for i,child in enumerate(value):
                if type(child) in (dict,list):walk(child,path+'/'+str(i),file)
    for path in sorted(run.iterdir()):
        assert path.is_file() and not path.is_symlink() and (path.suffix == '.json' or path.name in ('STOP','outer-STOP'))
        manifest.append({'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':sha(path)})
        if path.name.endswith(('-math.json','-update.json')) or '-tail-only-' in path.name:
            walk(json.loads(path.read_text()),'',path.name)
        if '-negative-' in path.name:
            value=json.loads(path.read_text());assert value['passed'] and value['rejected_ValueError'] and value['before']==value['after']
    for path,digest in numerical['sources'].items():assert sha(ROOT/path)==digest
    counts={'math_reports':len(list(run.glob('*-math.json'))),'updates':len(list(run.glob('*-update.json'))),'transactions':len(list(run.glob('*-negative-*.json'))),'tail_pairs':len(list(run.glob('*-tail-only-*.json'))),'files':len(manifest)}
    assert counts == {'math_reports':6,'updates':2,'transactions':45,'tail_pairs':2,'files':68}
    assert len(comparisons)==1 and comparisons[0]['file']=='mps-native-repeated_tail_001-math.json' and comparisons[0]['path']=='/genuine_primary'
    assert not (run/'result.json').exists() and not (run/'preservation-and-counts.json').exists()
    write('result.json',{'status':'complete_FIRST_saved_partial_TOP_failure_evidence_only','accepted_model_or_complete_fixture':False,
          'counts':counts,'failed_comparisons':comparisons,'coordinate_reconciliation':nonlocal_counts,'original_payloads':manifest,
          'original_source_pins_unchanged':len(numerical['sources']),'no_component_target_objective_oracle_Torch_model_scoring_RNG_replay':True,
          'parameter_scope':'Saved complete masks/digests only; absent parameter vectors not recomputed',
          'unreached':'All MPS augmented cases/updates/transactions/tail pairs and final count/source/RNG acceptance; no missing output reconstruction',
          'no_quality_or_root_cause_claim':True})


def main():
    assert len(sys.argv)==2 and sha(R/'plan.json')==sys.argv[1] and {p.name for p in R.iterdir()}=={'plan.json'}
    plan=json.loads((R/'plan.json').read_text())
    assert shutil.disk_usage(ROOT).free >= 10739515392
    for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest
    tree=ast.parse(Path(__file__).read_text())
    imports=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.Import):imports.extend(x.name for x in node.names)
        if isinstance(node,ast.ImportFrom):imports.append(node.module)
    assert set(imports)=={'array','ast','base64','fcntl','hashlib','json','math','os','shutil','pathlib','sys','zlib','src.research.cpu_prerequisite_guard'}
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        write('attempt.json',{'pid':os.getpid(),'plan_sha256':sys.argv[1],'no_retry':True,'no_replay':True})
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):inspect(plan)
        except BaseException as error:
            write('STOP',{'status':'closed_failed_NEW_saved_partial_metadata_review','error':repr(error),'no_retry':True,'no_replay':True});raise
        else:write('STOP',{'status':'complete_FIRST_saved_partial_failure_reconciliation','no_retry':True,'no_replay':True})


if __name__=='__main__':main()
