"""Bounded prelaunch checks of the only changed execution/configuration paths."""
import ast
import json
import math
from pathlib import Path
from research.block_norm_v1.runtime.train.lr_schedule import multiplier
from research.general_rsi_training_v1 import sha
from research.general_rsi_exposure_worker_v1 import PANEL, RUNS
ROOT=Path(__file__).resolve().parents[1]

def verify():
    old=(ROOT/'research/general_rsi_training_v1.py').read_text()
    new=(ROOT/'research/general_rsi_training_v2.py').read_text()
    assert new==old.replace('assert horizon in (40, 2048, 10240)','assert horizon in (40, 2048, 10240, 20480)')
    # Generation is unchanged, including decoding order and generator state.
    def fn(path,name):
        return ast.dump(next(x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name==name))
    assert fn(ROOT/'research/general_rsi_worker_v1.py','generate')==fn(ROOT/'research/general_rsi_exposure_worker_v1.py','generate')
    assert PANEL.name==RUNS.name=='general-rsi-exposure-001'
    plan=json.loads((PANEL/'plan.json').read_text())
    snapshots={}
    for h,expected in [(10240,342),(20480,683)]:
        schedule=dict(kind='linear_cooldown',total_steps=h,min_ratio=.1,start_step=2*h//3)
        ratios=[multiplier(schedule,i) for i in range(h)]
        assert ratios[0]==1 and ratios[-1]==.1
        assert all(.1<=r<=1 for r in ratios)
        assert all(a>=b for a,b in zip(ratios,ratios[1:]))
        first=math.ceil((2*h/3)/10)*10
        assert len(range(first,h+1,10))==expected
        snapshots[h]=dict(first=first,count=expected,final_ratio=ratios[-1])
    all_roles=[]
    for pair in range(3):
        a,b=plan['slots'][2*pair:2*pair+2]
        assert a['roles']==b['roles'] and len(set(a['roles'].values()))==4
        assert (a['horizon'],b['horizon'])==(10240,20480)
        assert a['lr']==b['lr']==.0025 and a['arm']==b['arm']=='raw_bytes'
        assert a['generation']==b['generation']==(pair==0)
        all_roles.extend(a['roles'].values())
    assert len(set(all_roles+[plan['bootstrap_seed']]))==13
    assert sum(s['horizon'] for s in plan['slots'])==92160
    assert plan['incremental_peak_bytes']==6*132975578+529357538+64*1024**2
    assert plan['forecast_seconds']<plan['tranche_seconds']
    assert plan['free_bytes_at_freeze']>=plan['reserve_bytes']+plan['incremental_peak_bytes']
    for path,digest in plan['sources'].items():assert sha(ROOT/path)==digest,path
    return dict(status='passed',plan_sha256=sha(PANEL/'plan.json'),training_change='Only20480horizon allowlist addition',
        inherited_numerical_qualification=plan['qualification'],new_numerical_training_performed=False,
        generation_ast_identical=True,schedules=snapshots,paired_roles_and_fresh_roles_valid=True,
        complete_capacity_forecast_valid=True,source_bindings_verified=len(plan['sources']),test_opened=False)

if __name__=='__main__':
    result=verify()
    with (PANEL/'preflight.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
