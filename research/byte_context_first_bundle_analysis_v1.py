"""Descriptive analysis of immutable completed runs; no model execution."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/tokenizer-evidence/byte-context-transform-001'
QUALITY=BASE/'quality-recovery002'
OUT=BASE/'first-bundle-analysis001'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    assert not OUT.exists()
    inputs={};rows=[]
    for slot in range(3):
        folder=QUALITY/f'{slot:02d}'
        acceptance=read(folder/'acceptance.json');result=read(folder/'result.json')
        assert acceptance['accepted'] and acceptance['result_sha256']==sha(folder/'result.json')
        assert result['spec']['bundle']==62102200000 and result['steps']==1200
        history=[json.loads(x) for x in (folder/'training.jsonl').read_text().splitlines()]
        assert len(history)==1200
        for name in ('acceptance.json','result.json','training.jsonl'):
            inputs[str((folder/name).relative_to(ROOT))]=sha(folder/name)
        windows=[]
        for end in (300,600,900,1200):
            data=history[end-100:end]
            windows.append(dict(end_step=end,mean_training_loss=math.fsum(x['loss'] for x in data)/100,
                clipped_fraction=sum(x['coefficient']<1 for x in data)/100))
        original_guard=BASE/'00/guard.json' if slot==0 else BASE/f'quality-recovery001/{slot:02d}/guard.json'
        inputs[str(original_guard.relative_to(ROOT))]=sha(original_guard)
        curves=result['curves']
        rows.append(dict(slot=slot,arm=result['spec']['arm'],training_windows=windows,
            h200_ce=result['ce'],online_validation=curves,
            h200_minus_final_online=result['ce']-curves[-1]['ce'],
            training_update_seconds=math.fsum(x['seconds'] for x in history),
            original_guard_seconds=read(original_guard)['elapsed_seconds'],
            online_scoring_seconds=math.fsum(x['seconds'] for x in curves)))
    diagnostic=BASE/'numerical-diagnostic002/result.json'
    inputs[str(diagnostic.relative_to(ROOT))]=sha(diagnostic)
    slopes=[]
    for report in read(diagnostic)['reports']:
        rms=report['propagated_stages']['tanh']['reference_rms']
        slopes.append(dict(block_start=report['block_start'],output_rms=rms,
            inferred_mean_elementwise_tanh_derivative=1-rms*rms))
    training=math.fsum(r['training_update_seconds'] for r in rows)
    guarded=math.fsum(r['original_guard_seconds'] for r in rows)
    scoring=math.fsum(r['online_scoring_seconds'] for r in rows)
    result=dict(kind='byte_context_first_bundle_descriptive_analysis_v1',rows=rows,
        local_derivative_formula='mean(1-tanh(u)^2) = 1 - RMS(tanh(u))^2; floating-point approximation',
        local_derivative_samples=slopes,cost=dict(training_seconds=training,
            original_guard_seconds=guarded,online_scoring_seconds=scoring,
            training_fraction_of_original_guards=training/guarded,
            online_scoring_fraction_of_original_guards=scoring/guarded),
        source_sha256=sha(Path(__file__)),inputs=inputs,no_training=True,no_model_execution=True,
        no_selection_change=True,no_adoption=True,
        limitations=['One completed seed bundle, not a cross-seed or causal conclusion.',
            'The derivative sample uses H200 weights and only the first and numerically worst validation windows; it is not representative of the training trajectory.',
            'Local tanh slope is not the full network gradient; surrounding matrices can amplify or attenuate it.',
            'Slot00 online curves used the original no-grad RMSNorm path and are diagnostic only; its primary H200 score is corrected.',
            'Online final weights are not the predefined H200 endpoint; do not switch endpoints.',
            'Cost is measured execution for these three runs, not total engineering wall time or a sustained benchmark.'])
    OUT.mkdir()
    with (OUT/'result.json').open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(training_seconds=training,guarded_seconds=guarded,
        training_fraction=training/guarded,online_scoring_seconds=scoring)),flush=True)


if __name__=='__main__':run()
