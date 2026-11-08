"""Descriptive completed-bundle analysis; no model execution or new scores."""
import argparse
import math
import json
from pathlib import Path
import traceback
from research.byte_context_transform_pilot_v1 import ROOT, sha, read, put

BASE=ROOT/'research/tokenizer-evidence/byte-residual-readout-001/quality001'
ARMS=('byte','residual_linear','residual_tanh')


def analyze(bundles):
    assert bundles in (1,2,3)
    plan=read(BASE/'quality-plan.json'); rows=[]; receipts=[]; results=[]
    for index in range(3*bundles):
        folder=BASE/f'{index:02d}'
        accepted=read(folder/'acceptance.json'); result=read(folder/'result.json'); guard=read(folder/'guard.json')
        assert accepted['accepted'] and accepted['result_sha256']==sha(folder/'result.json')
        assert accepted['guard_sha256']==sha(folder/'guard.json') and guard['status']=='finished'
        assert result['spec']==plan['specs'][index] and result['steps']==1200 and result['averaging_snapshots']==41
        history=[json.loads(line) for line in (folder/'training.jsonl').read_text().splitlines()]
        assert len(history)==1200 and all(row['step']==step for step,row in enumerate(history,1))
        scores=read(folder/'validation-mps.json')
        assert len(scores['per_target_losses'])==34278 and math.fsum(scores['per_target_losses'])/34278==result['ce']
        curves=[read(folder/f'curve-{step}.json') for step in (300,600,900,1200)]
        assert curves==result['curves'] and not result['holdout_scored'] and not result['adopted']
        row=dict(slot=index,bundle=result['spec']['bundle'],arm=result['spec']['arm'],h200_ce=result['ce'],
            final_online_ce=curves[-1]['ce'],h200_minus_final_online=result['ce']-curves[-1]['ce'],
            training_seconds=math.fsum(x['seconds'] for x in history),scheduled_evaluation_seconds=math.fsum(x['seconds'] for x in curves),
            guard_seconds=guard['elapsed_seconds'],final_gate=history[-1].get('gate_after'),curves=curves)
        for end in (300,600,900,1200):
            window=history[end-100:end]
            row[f'train_window_ending_{end}']=dict(
                mean_ce=math.fsum(x['loss'] for x in window)/100,
                mean_preclip_norm=math.fsum(x['pre_norm'] for x in window)/100,
                mean_clip_coefficient=math.fsum(x['coefficient'] for x in window)/100,
                clipped_fraction=sum(x['coefficient']<1 for x in window)/100)
        rows.append(row); results.append(result)
        names=['acceptance.json','result.json','guard.json','training.jsonl','validation-mps.json',*(f'curve-{s}.json' for s in (300,600,900,1200))]
        receipts.append(dict(slot=index,files={str((folder/n).relative_to(ROOT)):sha(folder/n) for n in names}))
    comparisons=[]
    for start in range(0,len(rows),3):
        group=results[start:start+3]
        assert len({r['train_sampler_sha256'] for r in group})==1
        assert all(r['spec']['roles']==group[0]['spec']['roles'] and r['initial_common_sha256']==group[0]['initial_common_sha256'] for r in group)
        by_arm={r['arm']:r for r in rows[start:start+3]}; assert set(by_arm)==set(ARMS)
        for candidate,control in (('residual_linear','byte'),('residual_tanh','byte'),('residual_tanh','residual_linear')):
            a,b=by_arm[candidate],by_arm[control]
            comparisons.append(dict(bundle=a['bundle'],candidate=candidate,control=control,
                h200_delta=a['h200_ce']-b['h200_ce'],final_online_delta=a['final_online_ce']-b['final_online_ce'],
                last100_training_delta=a['train_window_ending_1200']['mean_ce']-b['train_window_ending_1200']['mean_ce']))
    final_review=None
    if bundles==3:
        proof=read(BASE/'final-review.json')
        assert proof['accepted'] and proof['plan_sha256']==sha(BASE/'quality-plan.json') and proof['final_sha256']==sha(BASE/'final.json')
        final_review=dict(path=str(BASE/'final-review.json'),sha256=sha(BASE/'final-review.json'),quality_success=proof['quality_success'])
    return dict(kind='byte_residual_readout_completed_bundle_analysis_v1',plan_sha256=sha(BASE/'quality-plan.json'),
        source_sha256=sha(Path(__file__)),bundles=bundles,rows=rows,comparisons=comparisons,receipts=receipts,
        training_updates=1200*len(rows),training_execution_seconds=math.fsum(r['training_seconds'] for r in rows),
        scheduled_evaluation_seconds=math.fsum(r['scheduled_evaluation_seconds'] for r in rows),
        guarded_execution_seconds=math.fsum(r['guard_seconds'] for r in rows),final_review=final_review,
        scope='Descriptive existing-data analysis of fully completed paired bundles. Training windows are pre-update sampled losses over steps201–300/501–600/801–900/1101–1200. Online endpoints are diagnostics; H200 remains primary. No new model execution, validation scoring, holdout, bootstrap, tuning, adoption or causal claim. Partial bundles do not replace the frozen nine-run decision. Checkpoint correctness remains supported by original numerical receipts and the separate whole-panel review.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('bundles',type=int,choices=(1,2,3)); args=parser.parse_args()
    output=BASE/f'analysis-{args.bundles}-bundles001'; output.mkdir()
    put(output/'attempt.json',dict(bundles=args.bundles,source_sha256=sha(Path(__file__)),plan_sha256=sha(BASE/'quality-plan.json')))
    try:
        result=analyze(args.bundles); put(output/'result.json',result)
        print(json.dumps(dict(bundles=args.bundles,comparisons=result['comparisons'])),flush=True)
    except BaseException:
        put(output/'failure.json',dict(traceback=traceback.format_exc())); raise
