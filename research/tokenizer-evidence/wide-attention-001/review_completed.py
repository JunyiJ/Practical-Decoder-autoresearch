"""Read-only metadata/hash review of completed wider attention trials; no model evaluation."""
import datetime
import json
import math
import shutil
from pathlib import Path
import torch
from src.research.protocol import digest, file_digest
from src.research.runner import source_fingerprint, atomic_json
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_wide_study import config, specs, check_budget, RECIPE, PROTOCOL, RESOURCE, PILOT

root=Path('research_runs/tokenizer-wide-attention-001').absolute()
evidence=Path('research/tokenizer-evidence/wide-attention-001')
plan=json.loads((root/'plan.json').read_text());state=json.loads((root/'state.json').read_text())
assert digest(plan)==state['plan_sha256'] and source_fingerprint()==plan['source'] and runtime_identity()==plan['runtime']
for path,key in [(RECIPE,'recipe_sha256'),(PROTOCOL,'protocol_sha256'),(RESOURCE,'resource_review_sha256'),(PILOT/'plan.json','pilot_plan_sha256'),(Path(plan['config']['data']['path']),'corpus_sha256')]:
    assert file_digest(path)==plan[key]
expected={x['id']:x for x in specs(state['selection'])}
rows=[]
for result in state['records']:
    assert result['status']=='completed'
    spec=expected[result['id']];assert all(result[k]==v for k,v in spec.items())
    trial=root/spec['id'];request=json.loads((trial/'request.json').read_text())
    assert request['spec']==spec
    assert request['config']==config(trial,spec['family'],spec['seed'],spec['lr'])
    assert file_digest(trial/'request.json')==result['request_sha256']
    assert json.loads((trial/'result.json').read_text())==result and result['holdout_consumed'] is False
    assert result['training']['completed_steps']==1200 and result['training']['weight_averaging']['snapshots']==41
    token_contract=None;pair_bytes=0
    for kind in ('raw','h200'):
        score=result[kind];path=Path(score['checkpoint'])
        assert file_digest(path)==score['checkpoint_sha256']
        assert score['parameters']=={'gqa':62877696,'mla':62424192}[spec['family']]
        assert score['context']==128 and score['block_tokens']==2048 and score['target_count']==34278 and score['target_bytes']==111536
        assert sum(score['block_target_counts'])==34278
        assert all(math.isfinite(x) for x in score['block_loss_sums'])
        assert math.isclose(sum(score['block_loss_sums'])/34278,score['mean_ce'],abs_tol=1e-12)
        assert math.isclose(sum(score['block_loss_sums'])/111536/math.log(2),score['bits_per_byte'],abs_tol=1e-12)
        payload=torch.load(path,map_location='meta',weights_only=False)
        assert payload['cfg']==request['config'] and payload['step']==1200
        assert payload['cfg']['data']['tokenizer']=='gpt2'
        names=(payload['training_state']['weight_averaging_state']['parameter_names'] if kind=='raw' else payload['averaged_weights']['parameter_names'])
        assert len(names)==len(set(names))
        assert sum(payload['model_state'][n].numel() for n in names)==score['parameters']
        buffers=set(payload['model_state'])-set(names)
        assert buffers=={f'transformer.h.{i}.freq_{trig}' for i in range(4) for trig in ('cos','sin')}
        assert all(tuple(payload['model_state'][n].shape)==(128,16) for n in buffers)
        if kind=='raw':
            token_contract=payload['training_state']['contract']['tokenizer']
            assert payload['training_state']['weight_averaging_state']['snapshots']==41
        else:
            assert payload['tokenizer_contract']==token_contract
            assert payload['parent_checkpoint']['sha256']==result['raw']['checkpoint_sha256']
            assert payload['averaged_weights']['snapshots']==41 and payload['averaged_weights']['half_life']==200
        pair_bytes+=path.stat().st_size
        del payload
    used=sum(p.stat().st_size for p in trial.rglob('*') if p.is_file())
    assert used<=plan['per_trial_budget_bytes'][spec['family']]
    dest=evidence/spec['id'];dest.mkdir(exist_ok=True)
    for name in ('result.json','request.json','training-summary.json','metrics.jsonl','worker.log'):
        target=dest/name
        if target.exists():assert file_digest(target)==file_digest(trial/name)
        else:shutil.copyfile(trial/name,target)
    rows.append(dict(**spec,raw_ce=result['raw']['mean_ce'],h200_ce=result['h200']['mean_ce'],pair_bytes=pair_bytes,
                     trial_bytes=used,training_seconds=result['training']['training_time_sec'],result_sha256=file_digest(trial/'result.json')))
free=shutil.disk_usage(root).free;check_budget(plan,state['records'],sum(p.stat().st_size for p in root.rglob('*') if p.is_file()),free)
review=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='passed',completed_count=len(rows),records=rows,
            plan_sha256=state['plan_sha256'],reviewer_sha256=file_digest(Path(__file__)),free_bytes=free,
            scope='Completed trials only: checkpoint hashes, metadata/config/tokenizer/EMA lineage and saved full-validation arithmetic. No fresh tensor-finiteness or model-evaluation recomputation. No quality adoption.',holdout_consumed=False)
atomic_json(evidence/'completed-review.json',review)
progress=Path('research_state/tokenizer-quality-progress.json');q=json.loads(progress.read_text());q['wide_reviewed_count']=len(rows)
q['last_reviewed_trial']=rows[-1]['id'] if rows else None
q['last_resource_check']=dict(at=review['at'],free_bytes=free,reserve_bytes=plan['reserve_bytes'],active_workers=1 if state['status']=='running' else 0,outcome='Completed record hashes/metadata reviewed; frozen provenance and all remaining disk allocations pass. Actual process list checked separately.')
progress.write_text(json.dumps(q,indent=2)+'\n')
print(json.dumps(dict(completed_count=len(rows),records=rows,free_bytes=free),indent=2))
