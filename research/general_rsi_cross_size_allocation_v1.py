"""Read retained parameter tensors and forecast a depth-only comparison; no training."""
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import torch
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-boundary-001'
GIB=1024**3

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def main():
    rows={};bindings={}
    for arm,slot in [('native',4),('raw_bytes',5)]:
        path=ROOT/f'research/tokenizer-evidence/general-rsi-panel-001/slot-{slot:02d}/result.json'
        r=read(path);w=r['endpoints'][1]['weights'];p=Path(w['path'])
        assert p.stat().st_size==w['size'] and sha(p)==w['sha256']
        weights=torch.load(p,map_location='cpu',weights_only=True)['parameters']
        rows[arm]={k:dict(shape=list(v.shape),elements=v.numel()) for k,v in weights.items()}
        bindings[str(path.relative_to(ROOT))]=sha(path);bindings[str(p.relative_to(ROOT))]=w['sha256']
    native,byte=rows['native'],rows['raw_bytes']
    totals={k:sum(x['elements'] for x in v.values()) for k,v in rows.items()}
    assert totals==dict(native=28492416,raw_bytes=16351859)
    blocks=[]
    for i in range(4):
        prefix=f'transformer.h.{i}.'
        a={k[len(prefix):]:v for k,v in native.items() if k.startswith(prefix)}
        b={k[len(prefix):]:v for k,v in byte.items() if k.startswith(prefix)}
        assert a==b and (not blocks or a==blocks[0]);blocks.append(a)
    perblock=sum(x['elements'] for x in blocks[0].values())
    input_size=native['transformer.wte.weight']['elements'];output_size=native['lm_head.weight']['elements']
    composer=sum(v['elements'] for k,v in byte.items() if k.startswith('composer.'))
    assert input_size==output_size==50257*256 and composer==725235
    other=totals['native']-4*perblock-input_size-output_size
    sizes=[]
    for depth in [4,8,16]:
        n=input_size+output_size+other+depth*perblock;b=output_size+composer+other+depth*perblock
        sizes.append(dict(width=256,depth=depth,body_parameters=depth*perblock,other_parameters=other,native_parameters=n,byte_parameters=b,native_body_fraction=depth*perblock/n,vocabulary_table_fraction=(input_size+output_size)/n,parameter_saving_fraction=(n-b)/n,native_fp32_weight_bytes=4*n,byte_fp32_weight_bytes=4*b))
    # Planning caps. Actual serialization and timing must be measured before admission.
    # 3 pairs/size x2 endpoints/arm, 2MiB container allowance per checkpoint.
    endpoints=sum(6*(4*(s['native_parameters']+s['byte_parameters'])+4*1024**2) for s in sizes)
    largest=max(s['native_parameters'] for s in sizes)
    state_one=math.ceil((4*largest*4+64*1024**2)/(64*1024**2))*64*1024**2 # params+<=2 optimizer+EMA, buffers/metadata allowance
    parts=dict(all_36_confirmation_endpoints=endpoints,active_restart_and_serialization_overlap=2*state_one,metadata=512*1024**2,engineering_artifacts=1024**3,failed_attempt_allowance=2*1024**3,reserve=10*GIB,extra_headroom=2*GIB)
    times={}
    for arm,slots in [('native',[4,6,8]),('raw_bytes',[5,7,9])]:
        times[arm]=max(read(ROOT/f'research/tokenizer-evidence/general-rsi-panel-001/slot-{s:02d}/guard-01.json')['elapsed_seconds'] for s in slots)
    # Deliberately pessimistic depth multiplier on the whole old wall time.
    # This is a planning envelope, NOT a measured performance prediction.
    conservative_seconds=3*sum(d/4 for d in [4,8,16])*sum(times.values())*1.3
    result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),input_sha256=bindings,baseline_parameter_inventory=rows,per_block_parameters=perblock,composer_parameters=composer,candidate_sizes=sizes,planned_confirmation=dict(pairs_per_size=3,arms=2,sizes=3,runs=18,updates_per_run=10240,targets_per_update=1024,targets_per_run=10485760,all_run_targets=18*10485760,lr_screen_runs=0,fixed_recipe_scope=True),storage_planning_bytes=parts,storage_total_bytes=sum(parts.values()),free_bytes=shutil.disk_usage(ROOT).free,old_reference_max_guard_seconds=times,conservative_depth_linear_envelope_seconds=conservative_seconds,initial_tranche_seconds=36*3600,admitted=False,pending='New-depth construction/initialization, numerical execution, full restart, scorer, timing/memory and serialization qualification; prospective broader evaluation and interaction contract; complete measured admission.',test_opened=False)
    OUT.mkdir(exist_ok=True)
    with (OUT/'allocation.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['baseline_parameter_inventory','input_sha256']},indent=2))
if __name__=='__main__':main()
