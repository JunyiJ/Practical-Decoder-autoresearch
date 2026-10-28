"""Prospective full-panel high-water accounting; never dispatches or deletes.

Uses unchanged per-trajectory model/optimizer/EMA allowances. Every stage must
finish and receive tensor/hash/EMA and resource review before its losing weights
can be removed. Retain all final selected screens and nine confirmation pairs.
"""
import hashlib
import itertools
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
SOURCE=ROOT/'research/tokenizer-evidence/wide-hybrid-preparation-001/rank-storage-projection.json'
RESERVE=8*1024**3
METADATA=256*1024**2  # 27 bounded 4MiB process logs plus audit/results/manifests
FAMILIES=('hybrid','gated_mla','kda')


def projection():
    source=json.loads(SOURCE.read_text())
    rows=[r for r in source['records'] if r['width']==512]
    budgets={(r['family'],r['bottleneck_divisor']):r['per_trajectory_allowance_bytes'] for r in rows}
    assert len(rows)==9 and set(budgets)==set(itertools.product(FAMILIES,(8,4,2)))
    assert all(type(x) is int and x>0 for x in budgets.values())
    phases={
        'three_resource_pilots_before_audit_cleanup':sum(budgets[f,2] for f in FAMILIES),
        'nine_LR_screens_before_saved_selection_and_audit_cleanup':3*sum(budgets[f,4] for f in FAMILIES),
        'three_LR_winners_plus_six_rank_probes_before_saved_selection_and_audit_cleanup':sum(budgets[f,d] for f in FAMILIES for d in (8,4,2)),
    }
    selections=[]
    for divisors in itertools.product((8,4,2),repeat=3):
        n=4*sum(budgets[f,d] for f,d in zip(FAMILIES,divisors))
        # One selected screen and three fresh confirmation pairs per family.
        selections.append(dict(divisors=dict(zip(FAMILIES,divisors)),checkpoint_allowance_bytes=n))
    worst=max(r['checkpoint_allowance_bytes'] for r in selections)
    assert worst==4*sum(max(budgets[f,d] for d in (8,4,2)) for f in FAMILIES)
    phases['three_final_selected_screens_plus_nine_fresh_confirmations']=worst
    required=max(phases.values())+RESERVE+METADATA
    return dict(status='prospective_accounting_only_not_a_frozen_or_executable_plan',
        authorization='October 28 human authorized non-critical checkpoint cleanup moving forward.',
        width=512,trajectories=27,quality_trials=24,resource_pilots=3,
        original_all_retained_required_free_bytes=source['panels']['512']['required_free_bytes'],
        reserve_bytes=RESERVE,metadata_allowance_bytes=METADATA,
        per_trajectory_allowance_source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        phase_checkpoint_allowance_bytes=phases,all_27_final_rank_selections=selections,
        required_free_bytes=required,required_free_GiB=required/1024**3,
        assumptions=[
            'All three pilots audited and their full raw/H200 tensor/hash/EMA/cache/resource receipts retained before pilot weights deleted.',
            'All nine LR screens audited; frozen per-family LR selections persisted before losing screen pairs deleted. Three anchors retained for rank comparison.',
            'Six rank probes plus three anchors audited; frozen final selections persisted before losing rank pairs including replaced anchors deleted.',
            'All three final selected screening pairs and all nine fresh confirmation pairs retained. No deletion during an active worker or pending audit.',
            'Manifest-bound deletion receipts must replace only checkpoint-existence checks, never outcome/provenance/tensor checks; non-checkpoint artifacts remain hash-bound.',
            'Controller must check full future high-water headroom at every transition and stop on any failure; no optimistic reclaim until deletion is verified.',
            'Unexpected failures and their artifacts are preserved and stop launch; this is not automatic failed-run cleanup or retry.',
            'No rates, seeds, trial order, horizon, rank candidates, quality gates, memory threshold, holdout boundary or resource guard changed.',
            'Current v1 runners still enforce retain-all accounting; new tested retention-aware contracts and source freeze are required. No launch authorized by this calculation.'
        ])

if __name__=='__main__':
    print(json.dumps(projection(),indent=2))
