"""Analyze the independently reviewed panel using saved evidence only."""
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'research/tokenizer-evidence/byte-context-transform-001'
QUALITY=BASE/'quality-recovery002'
OUT=BASE/'final-analysis001'
PLAN_SHA='2f55b9e24888232ed6c2bbe5f41ae8b68bd6e9122eeca343de1973dcefb21aef'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    assert not OUT.exists()
    inputs={}
    def read(path):
        inputs[str(path.relative_to(ROOT))]=sha(path)
        return json.loads(path.read_text())
    plan=read(QUALITY/'quality-plan.json')
    assert sha(QUALITY/'quality-plan.json')==PLAN_SHA
    final=read(QUALITY/'final.json');review=read(QUALITY/'final-review.json')
    controller=read(QUALITY/'controller-complete.json')
    review_guard=read(QUALITY/'final-review-guard.json')
    assert review['accepted'] and review['final_sha256']==sha(QUALITY/'final.json')
    assert review['plan_sha256']==PLAN_SHA and final['completed'] and controller['completed']
    assert review_guard['status']=='finished' and review_guard['reason'] is None
    assert final['trajectories']==9 and final['updates']==10800
    rows=[]
    for slot in range(9):
        folder=QUALITY/f'{slot:02d}'
        acceptance=read(folder/'acceptance.json');result=read(folder/'result.json')
        assert acceptance['accepted'] and acceptance['result_sha256']==sha(folder/'result.json')
        history_path=folder/'training.jsonl'
        inputs[str(history_path.relative_to(ROOT))]=sha(history_path)
        history=[json.loads(x) for x in history_path.read_text().splitlines()]
        assert len(history)==1200 and all(x['step']==i for i,x in enumerate(history,1))
        assert result['spec']==plan['specs'][slot]
        if slot==0:original_guard=BASE/'00/guard.json'
        elif slot<3:original_guard=BASE/f'quality-recovery001/{slot:02d}/guard.json'
        else:original_guard=folder/'guard.json'
        guard=read(original_guard)
        windows=[]
        for end in (300,600,900,1200):
            data=history[end-100:end]
            windows.append(dict(end_step=end,mean_training_loss=math.fsum(x['loss'] for x in data)/100,
                clipped_fraction=sum(x['coefficient']<1 for x in data)/100))
        curves=[read(folder/f'curve-{step}.json') for step in (300,600,900,1200)]
        assert curves==result['curves']
        rows.append(dict(slot=slot,bundle=result['spec']['bundle'],arm=result['spec']['arm'],
            h200_ce=result['ce'],training_windows=windows,online_validation=curves,
            h200_minus_final_online=result['ce']-curves[-1]['ce'],
            online_score_caveat='Original no-grad RMSNorm path; diagnostic only' if slot==0 else None,
            training_update_seconds=math.fsum(x['seconds'] for x in history),
            original_guard_seconds=guard['elapsed_seconds'],
            online_scoring_seconds=math.fsum(x['seconds'] for x in curves),
            sampler_evidence='observed' if result['train_sampler_sha256'] is not None else 'source/count-derived'))
    bundles=[]
    for start in (0,3,6):
        group={r['arm']:r for r in rows[start:start+3]}
        assert set(group)=={'byte','linear','tanh'}
        ce={a:r['h200_ce'] for a,r in group.items()}
        bundles.append(dict(bundle=rows[start]['bundle'],h200_ce=ce,
            tanh_minus_byte=ce['tanh']-ce['byte'],tanh_minus_linear=ce['tanh']-ce['linear'],
            linear_minus_byte=ce['linear']-ce['byte'],
            comparable_final_online_tanh_minus_byte=None if start==0 else group['tanh']['online_validation'][-1]['ce']-group['byte']['online_validation'][-1]['ce'],
            comparable_averaging_gap_difference=None if start==0 else group['tanh']['h200_minus_final_online']-group['byte']['h200_minus_final_online'],
            last100_tanh_minus_byte=group['tanh']['training_windows'][-1]['mean_training_loss']-group['byte']['training_windows'][-1]['mean_training_loss']))
        if start:
            row=bundles[-1]
            assert abs(row['tanh_minus_byte']-row['comparable_final_online_tanh_minus_byte']-row['comparable_averaging_gap_difference'])<1e-12
    for arm in ('byte','linear','tanh'):
        assert abs(math.fsum(r['h200_ce'] for r in rows if r['arm']==arm)/3-final['means'][arm])<1e-12
    training=math.fsum(r['training_update_seconds'] for r in rows)
    guarded=math.fsum(r['original_guard_seconds'] for r in rows)
    scoring=math.fsum(r['online_scoring_seconds'] for r in rows)
    result=dict(kind='byte_context_final_descriptive_analysis_v1',rows=rows,bundles=bundles,
        primary_means=final['means'],quality_success=final['quality_success'],
        cost=dict(training_seconds=training,original_guard_seconds=guarded,online_scoring_seconds=scoring,
            training_fraction_of_original_guards=training/guarded,online_scoring_fraction_of_original_guards=scoring/guarded,
            prior_charged_seconds=plan['carried_elapsed_seconds'],recovery_controller_seconds=controller['elapsed_seconds'],
            final_review_seconds=review_guard['elapsed_seconds'],
            total_charged_execution_seconds=plan['carried_elapsed_seconds']+controller['elapsed_seconds']+review_guard['elapsed_seconds']),
        source_sha256=sha(Path(__file__)),inputs=inputs,no_model_execution=True,no_training=True,
        no_selection_change=True,no_adoption=True,
        limitations=['Descriptive training/averaging analysis is not a causal test or a new significance claim.',
            'Use the frozen independent review for the primary contrasts and uncertainty.',
            'Online final scores do not replace the predefined saved/reloaded H200 endpoint.',
            'Slot00 online curve used the original no-grad RMSNorm path; no corrected RAW comparison is claimed.',
            'Slots00/02 have source/count-derived terminal sampler evidence, not published runtime digests.',
            'Measured execution cost excludes human/agent engineering wall time and is not a deployment benchmark.'])
    assert result['cost']['total_charged_execution_seconds']<=plan['total_seconds']
    OUT.mkdir()
    with (OUT/'result.json').open('x') as stream:
        json.dump(result,stream,indent=2,sort_keys=True,allow_nan=False);stream.write('\n')
    print(json.dumps(dict(bundles=bundles,primary_means=final['means'],cost=result['cost'])),flush=True)


if __name__=='__main__':run()
