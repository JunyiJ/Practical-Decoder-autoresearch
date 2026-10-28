"""Close the frozen wider attention panel from retained evidence; never dispatch training."""
import datetime
import json
import math
import shutil
import statistics as st
from pathlib import Path
from src.research.protocol import digest, file_digest
from src.research.runner import source_fingerprint, atomic_json
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_wide_study import conclusion, checked_panel, specs, RECIPE, PROTOCOL, RESOURCE

root=Path('research_runs/tokenizer-wide-attention-001')
out=Path('research/tokenizer-evidence/wide-attention-001')
read=lambda p:json.loads(p.read_text())
state=read(root/'state.json'); plan=read(root/'plan.json'); timing=read(root/'timing/state.json')
assert state['status']==timing['status']=='completed'
assert state['active'] is None and timing['completed']==[3411,3412,3413]
assert digest(plan)==state['plan_sha256'] and source_fingerprint()==plan['source'] and runtime_identity()==plan['runtime']
for p,k in [(RECIPE,'recipe_sha256'),(PROTOCOL,'protocol_sha256'),(RESOURCE,'resource_review_sha256'),(Path(plan['config']['data']['path']),'corpus_sha256')]:
    assert file_digest(p)==plan[k]
checked_panel(state['records'],specs(state['selection']))
assert conclusion(state['records'],state['selection'])==state['conclusion']
assert timing['quality_state_sha256']==file_digest(root/'state.json')
audit=read(out/'tensor-audit.json'); completed=read(out/'completed-review.json')
assert len(audit['records'])==completed['completed_count']==12
assert audit['quality_state_sha256']==file_digest(root/'state.json')
assert {r['id'] for r in audit['records']}=={r['id'] for r in state['records']}
for r in state['records']:
    a=next(x for x in audit['records'] if x['id']==r['id'])
    assert a['all_saved_float_tensors_finite'] and a['export_exactly_matches_saved_ema_and_raw_buffers']
    assert not r['holdout_consumed']
    for kind in ['raw','h200']:
        assert file_digest(Path(r[kind]['checkpoint']))==r[kind]['checkpoint_sha256']==a[kind+'_sha256']
families=['gqa','mla']
means={f:st.mean(r['h200']['mean_ce'] for r in state['records'] if r['stage']=='confirm' and r['family']==f) for f in families}
training={f:st.mean(r['training']['training_time_sec'] for r in state['records'] if r['stage']=='confirm' and r['family']==f) for f in families}
per_seed=[]; tokens=None; max_error=0
for seed in [3411,3412,3413]:
    p=root/f'timing/seed-{seed}.json'; t=read(p)
    assert t['status']=='completed' and t['device']=='mps' and t['dtype']=='float32'
    assert (t['prompt_tokens'],t['decode_tokens'],t['warmups'],t['repeats'])==(32,64,2,8)
    assert len(t['fixed_train_token_ids'])==96
    if tokens is None:tokens=t['fixed_train_token_ids']
    assert tokens==t['fixed_train_token_ids']
    assert set(t['metadata'])==set(families)
    for f in families:
        q=next(r for r in state['records'] if r['stage']=='confirm' and r['seed']==seed and r['family']==f)
        m=t['metadata'][f]
        assert m['sha256']==q['h200']['checkpoint_sha256'] and m['checkpoint']==q['h200']['checkpoint']
        assert m['parameter_bytes']=={'gqa':251510784,'mla':249696768}[f]
        assert m['parameters']*4==m['parameter_bytes'] and m['width']==512
    assert len(t['records'])==32
    assert {(r['batch'],r['family'],r['repeat']) for r in t['records']}=={(b,f,i) for b in [1,8] for f in families for i in range(8)}
    assert len(t['correctness'])==4 and {(r['batch'],r['family']) for r in t['correctness']}=={(b,f) for b in [1,8] for f in families}
    assert all(math.isfinite(r['max_absolute_error']) and r['max_absolute_error']>=0 for r in t['correctness'])
    max_error=max(max_error,max(r['max_absolute_error'] for r in t['correctness']))
    for r in t['records']:
        assert r['kv_bytes']=={'gqa':1048576,'mla':327680}[r['family']]*r['batch']
        assert all(math.isfinite(r[k]) and r[k]>0 for k in ['prefill_seconds','decode_seconds','decode_tokens_per_second'])
        assert math.isclose(r['decode_tokens_per_second'],64*r['batch']/r['decode_seconds'],rel_tol=1e-10)
    for b in [1,8]:
        med={f:{k:st.median(r[k] for r in t['records'] if r['family']==f and r['batch']==b) for k in ['prefill_seconds','decode_seconds','decode_tokens_per_second']} for f in families}
        ratios={f+'_vs_'+c:{k:med[f][k]/med[c][k] for k in ['prefill_seconds','decode_seconds']} for f,c in [('mla','gqa')]}
        cache={f:{r['kv_bytes'] for r in t['records'] if r['batch']==b and r['family']==f} for f in families}
        assert all(len(v)==1 for v in cache.values())
        reduction=1-next(iter(cache['mla']))/next(iter(cache['gqa']))
        assert reduction>=plan['min_cache_reduction']
        per_seed.append(dict(seed=seed,batch=b,medians=med,ratios=ratios,cache_reduction_fraction=reduction))
summary={str(b):{pair:{k:st.median(r['ratios'][pair][k] for r in per_seed if r['batch']==b) for k in ['prefill_seconds','decode_seconds']} for pair in per_seed[0]['ratios']} for b in [1,8]}
# Preserve all small inference evidence, never checkpoints, alongside reviewed records.
for folder in [root/'timing']:
    dest=out/folder.name;dest.mkdir(exist_ok=True)
    for p in folder.iterdir():
        if p.is_file():
            assert p.stat().st_size<4*1024**2
            target=dest/p.name
            if target.exists():assert file_digest(target)==file_digest(p)
            else:shutil.copyfile(p,target)
for name in ['plan.json','state.json','timing-controller.log']:
    target=out/('final-state.json' if name=='state.json' else name)
    if target.exists():assert file_digest(target)==file_digest(root/name)
    else:shutil.copyfile(root/name,target)
eligible=state['conclusion']['quality_eligible'] and all(r['cache_reduction_fraction']>=plan['min_cache_reduction'] for r in per_seed)
review=dict(status='completed_adopted' if eligible else 'completed_not_adopted',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  plan_sha256=state['plan_sha256'],source=plan['source'],quality_trials=12,confirmation_seeds=[3411,3412,3413],
  timing_pairs=3,selection=state['selection'],mean_h200_ce=means,mean_training_seconds=training,
  quality=state['conclusion'],parameter_bytes={'gqa':251510784,'mla':249696768},
  cache_bytes_per_batch_item={'gqa':1048576,'mla':327680},
  per_seed_inference=per_seed,median_of_seed_median_ratios=summary,max_cached_full_absolute_error=max_error,
  adopted=eligible,decision='Adopt the scoped width512 quarter-latent RoPE MLA option versus equally tuned fresh GQA.' if eligible else 'Do not adopt width512 MLA under the frozen gates.',
  scope='Three trained seeds, adaptive validation, FP32 MPS context128 batches1/8. Conditional text intervals do not measure seed-population uncertainty. Both models resident during timing; working-memory readings are not single-model peak or capacity. No larger-data/width transfer or test-holdout result. Existing width256 recipes remain separately scoped.',
  quality_state_sha256=file_digest(root/'state.json'),tensor_audit_sha256=file_digest(out/'tensor-audit.json'),
  completed_review_sha256=file_digest(out/'completed-review.json'),timing_state_sha256=file_digest(root/'timing/state.json'),
  helper_sha256=file_digest(Path(__file__)),holdout_consumed=False)
atomic_json(out/'final-review.json',review)
print(json.dumps({k:review[k] for k in ['status','mean_h200_ce','mean_training_seconds','median_of_seed_median_ratios','max_cached_full_absolute_error']},indent=2))
