"""Independent read-only review of the finished byte-context panel.

No model construction, training or validation rescoring. Reconstructs only expected
private CPU sampler state for the completed baseline whose observed digest was
not saved; source/count evidence is explicit, not recovered runtime state.
Bootstrap arithmetic is independently rebuilt with the established scalar reviewer.
"""
import fcntl
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT/'research/tokenizer-evidence/byte-context-transform-001'
PREVIOUS = ORIGINAL/'quality-recovery001'
FIX = ORIGINAL/'cpu-reference-recovery001'
BASE = ORIGINAL/'quality-recovery002'
PLAN_SHA = None


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024**2): h.update(block)
    return h.hexdigest()


def run():
    import torch
    from research.byte_input_quality_final_review_v1 import contrast
    torch.set_num_threads(4)
    assert sha(BASE/'quality-plan.json') == PLAN_SHA
    plan = read(BASE/'quality-plan.json'); final = read(BASE/'final.json')
    assert final['completed'] and final['trajectories'] == 9 and final['updates'] == 10800
    assert not final['adopted'] and not final['holdout_scored']
    for path, digest in plan['sources'].items(): assert sha(ROOT/path) == digest, path
    assert plan['primary_min_gain']==.02 and plan['bootstrap_quantile']==1-.05/6
    assert plan['cpu_mps_mean_ce_atol']==3e-5 and plan['cpu_mps_target_ce_atol']==3e-4
    for record in plan['assets']: assert sha(record['path'])==record['sha256']
    original=read(ORIGINAL/'quality-plan.json')
    assert sha(ORIGINAL/'quality-plan.json')==plan['original_plan_sha256']
    for key in ('specs','bootstrap_seed','primary_min_gain','bootstrap_quantile','cpu_mps_mean_ce_atol','cpu_mps_target_ce_atol','output_cap','reserve','extra_headroom','per_trial_seconds','total_seconds'):
        assert plan[key]==original[key],key
    assert plan['reused_training_slots']==[0,1,2] and plan['unattempted_slots']==list(range(3,9))
    assert sha(PREVIOUS/'quality-plan.json')==plan['previous_plan_sha256']
    assert sha(FIX/'plan.json')==plan['cpu_reference_plan_sha256']
    assert sha(FIX/'result.json')==plan['cpu_reference_result_sha256']
    correction=read(FIX/'acceptance.json')
    assert correction['accepted'] and correction['result_sha256']==sha(FIX/'result.json') and correction['guard_sha256']==sha(FIX/'guard.json')
    resource=read(PREVIOUS/'resource-recovery001/slot-01.json')
    roots=[ORIGINAL,ROOT/'research_runs/byte-context-transform-001',ROOT/'research_runs/byte-context-transform-recovery-001']
    assert resource['roots']==plan['resource_roots']==[str(p) for p in roots]
    assert resource['correct_total_retained_bytes']<plan['output_cap']
    assert resource['scientific_acceptance_sha256']==sha(PREVIOUS/'01/acceptance.json')
    actual_bytes=sum(p.stat().st_size for root in roots for p in root.rglob('*') if p.is_file())
    assert actual_bytes<plan['output_cap']
    expected_data = read(ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-plan.json')['data']
    receipts = []; rows = []; scores = []; cost = []; codebook_identity = None
    for index, spec in enumerate(plan['specs']):
        out = BASE/f'{index:02d}'
        receipt = read(out/'acceptance.json'); row = read(out/'result.json')
        assert read(out/'attempt.json')['plan_sha256'] == PLAN_SHA
        config = read(out/'config.json')
        assert config['spec'] == spec and config['data'] == expected_data
        assert receipt['accepted'] and receipt['result_sha256'] == sha(out/'result.json')
        assert receipt['guard_sha256'] == sha(out/'guard.json')
        guard = read(out/'guard.json')
        assert guard['status'] == 'finished' and guard['reason'] is None
        assert guard['limits'] == dict(rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,timeout_seconds=600 if index<3 else 3600)
        assert guard['elapsed_seconds'] < (600 if index<3 else 3600)
        assert guard['observation']['peak_rss_bytes'] < 4*1024**3
        if index>=3:assert guard['observation']['mps_bytes'] < 6*1024**3
        assert guard['observation']['free_bytes'] >= 8*1024**3
        assert row['completed'] and row['spec'] == spec and row['steps'] == 1200
        assert row['averaging_snapshots'] == 41 and row['targets_seen'] == 1228800
        assert not row['holdout_scored'] and not row['adopted']
        if index<3:
            old=PREVIOUS/f'{index:02d}'
            assert receipt['kind']=='completed_training_plus_verified_cpu_reference'
            assert sha(out/'training.jsonl')==sha(old/'training.jsonl')==receipt['original_training_sha256']
            assert sha(out/'guard.json')==sha(FIX/'guard.json')
            assert receipt['correction_acceptance_sha256']==sha(FIX/'acceptance.json')
            assert sha(out/'validation-cpu.json')==sha(FIX/f'{index:02d}-cpu.json')
            assert sha(out/'validation-mps.json')==sha(old/'validation-mps.json')==receipt['original_primary_sha256']
            verification=read(FIX/f'{index:02d}-verification.json')
            assert verification['original_cpu_exact'] and verification['fp64_max_error']<=3e-5
            assert verification['primary_mps_sha256']==sha(out/'validation-mps.json')
            original_guard=ORIGINAL/'00/guard.json' if index==0 else old/'guard.json'
            assert sha(original_guard)==receipt['original_guard_sha256']
            oldguard=read(original_guard)
            assert oldguard['status']==('finished' if index==1 else 'fixture_exception')
            assert oldguard['reason']==(None if index==1 else 'AssertionError()')
            assert oldguard['elapsed_seconds']<3600
            assert oldguard['observation']['peak_rss_bytes']<4*1024**3 and oldguard['observation']['mps_bytes']<6*1024**3
            assert oldguard['observation']['free_bytes']>=8*1024**3
            assert row['imported_training']
            generator=torch.Generator(device='cpu').manual_seed(spec['roles']['train_sampler'])
            for _ in range(1200):torch.randint(267688-512,(2,),generator=generator)
            expected=hashlib.sha256(generator.get_state().numpy().tobytes()).hexdigest()
            assert expected==row['expected_sampler_sha256']
            if index==1:assert row['train_sampler_sha256']==read(old/'result.json')['train_sampler_sha256']==expected
            else:assert row['train_sampler_sha256'] is None
        count = 105 if spec['arm'] == 'byte' else 106
        elements = 16351859 if spec['arm'] == 'byte' else 16417395
        assert row['parameter_count'] == elements
        history = [json.loads(s) for s in (out/'training.jsonl').read_text().splitlines()]
        assert len(history) == 1200
        for step, h in enumerate(history, 1):
            assert h['step'] == step and h['gradient_tensors'] == count and h['threshold'] == 1.
            assert all(math.isfinite(h[k]) for k in ('loss','seconds','pre_norm','post_norm','coefficient'))
            assert h['loss'] >= 0 and h['seconds'] > 0 and 0 <= h['post_norm'] <= 1.00001
            assert abs(h['coefficient']-min(1., 1./(h['pre_norm']+1e-6))) < 2e-5
            assert abs(h['post_norm']-h['pre_norm']*h['coefficient']) <= max(1e-5,h['post_norm']*2e-5)
        assert row['curves'] == [read(out/f'curve-{step}.json') for step in (300,600,900,1200)]
        saved = []
        for key, kind in (('raw','final_raw_weights_not_resume'),('h200','h200_inference_weights')):
            record = row[key]; path = Path(record['path'])
            assert path.is_file() and not path.is_symlink() and path.stat().st_nlink == 1
            assert path.stat().st_size == record['size'] and sha(path) == record['sha256']
            payload = torch.load(path, map_location='cpu', weights_only=True)
            assert payload['kind'] == kind and payload['spec'] == spec and payload['plan_sha256'] == (plan['original_plan_sha256'] if index==0 else plan['previous_plan_sha256'] if index<3 else PLAN_SHA)
            tensors = payload['weights']
            assert all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in tensors.values())
            assert len({t.untyped_storage().data_ptr() for t in tensors.values()}) == len(tensors)
            assert tensors['lm_head.weight'].shape == (50257,256)
            assert tensors['composer.byte_weight'].shape == (259,15)
            assert ('context_weight' in tensors) == (spec['arm'] != 'byte')
            if spec['arm'] != 'byte': assert tensors['context_weight'].shape == (256,256)
            assert len(tensors) == count and sum(t.numel() for t in tensors.values()) == elements
            assert payload['optimizer_resume'] is False
            if codebook_identity is None: codebook_identity = payload['codebook_identity']
            assert payload['codebook_identity'] == codebook_identity
            assert codebook_identity[1] == 'raw_bytes' and codebook_identity[4:6] == (50257,130)
            assert payload['identity'] == ('byte_context_transform_v1',spec['arm'],spec['roles']['composer_initializer'],spec['projection_seed'],codebook_identity)
            assert json.loads(json.dumps(payload['identity'])) == config['model_identity']
            saved.append(tensors)
        assert saved[0].keys() == saved[1].keys()
        assert all(saved[0][n].shape == t.shape for n,t in saved[1].items())
        assert any(not torch.equal(saved[0][n],t) for n,t in saved[1].items())
        del saved, tensors, payload
        score = read(out/'validation-mps.json'); losses = score['per_target_losses']
        assert len(losses) == score['target_count'] == 34278 and not score['holdout_scored']
        assert all(type(x) is float and math.isfinite(x) and x >= 0 for x in losses)
        assert math.fsum(losses)/34278 == score['ce'] == row['ce']
        cpu = read(out/'validation-cpu.json'); cpu_losses = cpu['per_target_losses']
        assert cpu['target_count'] == len(cpu_losses) == 34278 and not cpu['holdout_scored']
        assert all(type(x) is float and math.isfinite(x) and x >= 0 for x in cpu_losses)
        assert cpu['ce'] == math.fsum(cpu_losses)/34278
        numerical = dict(mean_ce_error=abs(cpu['ce']-score['ce']),
                         max_target_ce_error=max(abs(a-b) for a,b in zip(cpu_losses,losses)))
        assert numerical == read(out/'numerical.json') == row['numerical']
        assert numerical['mean_ce_error'] <= 3e-5 and numerical['max_target_ce_error'] <= 3e-4
        receipts.append(dict(slot=index, acceptance_sha256=sha(out/'acceptance.json'),
                             validation_sha256=sha(out/'validation-mps.json'), training_sha256=sha(out/'training.jsonl')))
        rows.append(row); scores.append(losses)
        cost.append(dict(slot=index, arm=spec['arm'], training_seconds=math.fsum(h['seconds'] for h in history),
                         clipped_fraction=sum(h['coefficient'] < 1 for h in history)/1200,
                         peak_rss_bytes=(oldguard if index<3 else guard)['observation']['peak_rss_bytes']))
    pairs = []
    for start in (0,3,6):
        assert len({r['train_sampler_sha256'] if r['train_sampler_sha256'] is not None else r['expected_sampler_sha256'] for r in rows[start:start+3]}) == 1
        assert all(r['spec']['roles'] == rows[start]['spec']['roles'] for r in rows[start:start+3])
        pairs.append({rows[i]['spec']['arm']:scores[i] for i in range(start,start+3)})
    rebuilt = {}
    for candidate, control in (('tanh','byte'),('tanh','linear')):
        # The old helper's names are labels only; it is a pure saved-loss calculation.
        mapped = [{'raw_bytes':p[candidate], 'control':p[control]} for p in pairs]
        result = contrast(mapped, 'control', plan['bootstrap_seed'], -.02)
        stored = final['contrasts'][control]
        assert all(abs(a-b) < 1e-12 for a,b in zip(result['deltas'],stored['deltas'],strict=True))
        assert abs(result['mean_delta']-stored['mean']) < 1e-12
        assert all(a['block_tokens'] == b['block_tokens'] and abs(a['upper']-b['upper']) < 1e-12
                   for a,b in zip(result['bounds'],stored['bounds'],strict=True))
        assert result['passes'] == stored['passes']
        rebuilt[candidate+'_vs_'+control] = result
    assert final['quality_success'] == all(rebuilt['tanh_vs_'+a]['passes'] for a in ('byte','linear'))
    for arm in ('byte','linear','tanh'):
        assert abs(final['means'][arm]-math.fsum(r['ce'] for r in rows if r['spec']['arm']==arm)/3) < 1e-12
    return dict(accepted=True, plan_sha256=PLAN_SHA, final_sha256=sha(BASE/'final.json'),
                reviewer_sha256=sha(Path(__file__)), actual_retained_bytes=actual_bytes, receipts=receipts, comparisons=rebuilt,
                costs=cost, quality_success=final['quality_success'], adopted=False,
                scope='Saved weights, completion/guard/history/coverage/pairing checks and independent scalar statistics; no new model execution. Final weights do not provide optimizer resume or independent reconstruction of all EMA snapshots. Slots00/02 terminal sampler states are source/count-derived because their original runtime digests were not saved. Slot01 observed digest is retained. Primary MPS reports of all three completed runs are unchanged; CPU verification uses the separately validated FP32 reference. Original failed reports and guards are retained.')


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('plan_sha256')
    PLAN_SHA=parser.parse_args().plan_sha256
    assert len(PLAN_SHA)==64 and all(c in '0123456789abcdef' for c in PLAN_SHA)
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        with CpuPrerequisiteGuard(BASE/'final-review-guard.json', ROOT, timeout_seconds=300):
            result = run()
        with (BASE/'final-review.json').open('x') as stream:
            json.dump(result,stream,sort_keys=True,allow_nan=False);stream.write('\n')
