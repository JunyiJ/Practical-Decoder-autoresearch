"""Saved-artifact and resource audit after both wider MPS pilots exit."""
import gc
import json
import math
import shutil
import datetime
from pathlib import Path
import torch
from src.research.tokenizer_wide_pilot import verify, ALLOWANCE, PARAMETERS, required_free, KINDS
from src.research.protocol import file_digest, digest
from src.research.runner import atomic_json
root=Path('research_runs/tokenizer-wide-pilot-001').absolute();out=Path('research/tokenizer-evidence/wide-pilot-001')
plan,state=verify(root)
assert state['status']=='completed' and state['active'] is None and state['completed']==list(KINDS)
torch.set_num_threads(2)
rows=[]
for kind in KINDS:
    trial=root/kind;result=json.loads((trial/'result.json').read_text());req=json.loads((trial/'request.json').read_text())
    assert result['status']=='completed' and result['kind']==kind and not result['holdout_consumed'] and not result['quality_selection']
    assert result['request_sha256']==file_digest(trial/'request.json')
    assert result['training']['completed_steps']==20 and result['training']['weight_averaging']['snapshots']==3
    assert result['max_rss_bytes']<4*1024**3 and result['wall_seconds']<600
    assert [x['kind'] for x in result['checkpoints']]==['raw','h200']
    metadata={};raw_state=None
    for c in result['checkpoints']:
        path=Path(c['checkpoint']);assert path.stat().st_size==c['bytes'] and file_digest(path)==c['sha256']
        assert c['unique_parameter_bytes']==4*PARAMETERS[kind]
        assert c['observed_mps_driver_bytes']<6*1024**3
        assert [x['batch'] for x in c['cache_checks']]==[1,8]
        for check in c['cache_checks']:
            assert math.isfinite(check['max_absolute_error'])
            assert check['cache_bytes']==check['batch']*4*128*4*(512 if kind=='gqa' else 160)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        assert payload['cfg']==req['config'] and payload['step']==20
        def finite(v):
            if isinstance(v,torch.Tensor):return not (v.is_floating_point() or v.is_complex()) or bool(torch.isfinite(v).all())
            if isinstance(v,dict):return all(finite(x) for x in v.values())
            if isinstance(v,(list,tuple)):return all(finite(x) for x in v)
            return True
        assert finite(payload['model_state'])
        if c['kind']=='raw':
            assert finite(payload['optimizer_state'])
            wa=payload['training_state']['weight_averaging_state'];assert wa['snapshots']==3 and finite(wa)
            token=payload['training_state']['contract']['tokenizer']
            names=wa['parameter_names']; assert sum(payload['model_state'][n].numel() for n in names)==PARAMETERS[kind]
            # Store hashes only, releasing the large raw payload before loading export.
            import hashlib
            hash_tensor=lambda t:hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
            raw_state={n:hash_tensor(wa['averages']['200'][n] if n in names else t) for n,t in payload['model_state'].items()}
            del wa
        else:
            assert payload['tokenizer_contract']==token and token['name']=='gpt2'
            assert payload['parent_checkpoint']['sha256']==result['checkpoints'][0]['sha256']
            assert payload['averaged_weights']['snapshots']==3 and payload['averaged_weights']['half_life']==200
            assert {n:hash_tensor(t) for n,t in payload['model_state'].items()}==raw_state
        metadata[c['kind']]=dict(sha256=c['sha256'],bytes=c['bytes'])
        del payload;gc.collect()
    total=sum(p.stat().st_size for p in trial.rglob('*') if p.is_file());assert total<ALLOWANCE[kind]
    dest=out/kind
    for name in ['request.json','result.json','training-summary.json','metrics.jsonl','worker.log','prelaunch.json']:
        if (dest/name).exists():assert file_digest(dest/name)==file_digest(trial/name)
        else:shutil.copyfile(trial/name,dest/name)
    rows.append(dict(kind=kind,result_sha256=file_digest(trial/'result.json'),checkpoints=metadata,
        trial_bytes=total,pair_bytes=sum(c['bytes'] for c in result['checkpoints']),wall_seconds=result['wall_seconds'],
        training_seconds=result['training']['training_time_sec'],max_rss_bytes=result['max_rss_bytes'],
        raw_h200_finite=True,ema_export_exact=True,unique_parameter_bytes=4*PARAMETERS[kind],cache_bytes_batch1=result['checkpoints'][0]['cache_checks'][0]['cache_bytes']))
free=shutil.disk_usage(root).free;assert free>=required_free(list(KINDS))
for name in ['state.json','controller.log']:shutil.copyfile(root/name,out/name)
review=dict(status='passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=rows,plan_sha256=digest(plan),
    quality_allowances=ALLOWANCE,free_bytes=free,remaining_quality_plus_reserve_bytes=required_free(list(KINDS)),
    quality_selection=False,holdout_consumed=False,helper_sha256=file_digest(Path(__file__)),
    scope='Width512 feasibility and saved-artifact audit only. No quality ranking from20-update pilot scores. Short pilot does not establish1200-update stability.')
atomic_json(out/'review.json',review)
print(json.dumps(review,indent=2))
