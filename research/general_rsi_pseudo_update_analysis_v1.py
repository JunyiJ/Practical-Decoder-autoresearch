"""Retain every frozen diagnostic contrast; no quality/adoption decision."""
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/tokenizer-evidence/general-rsi-pseudo-update-001'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(values):
    assert values and all(math.isfinite(x) for x in values)
    return dict(count=len(values), mean=math.fsum(values)/len(values), minimum=min(values),
                maximum=max(values), positive=sum(x>0 for x in values), negative=sum(x<0 for x in values),
                zero=sum(x==0 for x in values), values=values)


def main():
    plan = json.loads((OUT/'plan.json').read_text())
    attempt = OUT/'attempt-01'
    result = json.loads((attempt/'result.json').read_text())
    guard = json.loads((attempt/'guard.json').read_text())
    assert result['plan_sha256'] == sha(OUT/'plan.json') and guard['status'] == 'finished'
    assert result['cases'] == 24 and result['branch_observations'] == 96
    assert result['total_seconds'] < plan['timeout_seconds']
    assert guard['observation']['peak_rss_bytes'] < plan['rss_limit_bytes']
    assert guard['observation']['free_bytes'] >= plan['reserve_bytes']
    assert result['peak_mps_boundary_bytes'] < plan['mps_limit_bytes']
    receipts = {str(p.relative_to(ROOT)):sha(p) for p in (OUT/'plan.json', OUT/'admission.json', attempt/'result.json',attempt/'guard.json')}
    parents, all_contrasts = [], {c:[] for c in plan['contrasts']}
    for config in plan['parents']:
        slot = config['parent']['slot']
        parent_path = attempt/f'parent-{slot:02d}.json'
        parent = json.loads(parent_path.read_text()); receipts[str(parent_path.relative_to(ROOT))] = sha(parent_path)
        rows = []
        for case in range(8):
            branches = {}
            for branch in plan['branches']:
                path = attempt/f'p{slot:02d}-c{case:02d}-{branch}.json'
                r = json.loads(path.read_text()); receipts[str(path.relative_to(ROOT))] = sha(path)
                assert (r['parent_slot'],r['case'],r['branch']) == (slot,case,branch)
                assert r['restored_state_sha256'] == parent['state_sha256']
                assert r['input'] == config['inputs']['cases'][case]
                assert r['evaluation']['targets'] == 8192 and len(r['evaluation']['window_loss_sums']) == 64
                assert r['optimizer_steps'] == (1 if branch == 'identity' else 2)
                assert r['natural_ce_delta'] == r['evaluation']['ce'] - parent['baseline']['ce']
                if branch == 'identity': assert r['evaluation'] == parent['baseline']
                branches[branch] = r
            contrasts = {c:branches[c.split('-')[0]]['evaluation']['ce']-branches[c.split('-')[1]]['evaluation']['ce'] for c in plan['contrasts']}
            for c, v in contrasts.items(): all_contrasts[c].append(v)
            rows.append(dict(case=case, contrasts=contrasts, branches=branches))
        movements = {}
        for branch in plan['branches']:
            movements[branch] = {}
            for group in ('Muon','AdamW'):
                m = [r['branches'][branch]['displacement'][group] for r in rows]
                movements[branch][group] = {key:stats([x[key] for x in m]) for key in ('update_norm','relative_update_norm','gradient_sq','gradient_dot_update')}
                cosines = [x['gradient_update_cosine'] for x in m if x['gradient_update_cosine'] is not None]
                movements[branch][group]['gradient_update_cosine'] = stats(cosines) if cosines else None
        parents.append(dict(parent_slot=slot, baseline=parent['baseline']['ce'],
            contrasts={c:stats([r['contrasts'][c] for r in rows]) for c in plan['contrasts']},
            movement=movements, cases=[dict(case=r['case'],contrasts=r['contrasts']) for r in rows]))
    summary = dict(kind='common_state_pseudo_update_analysis_v1', sources=receipts,
        analysis_source_sha256=sha(__file__), parents=parents,
        pooled_descriptive_contrasts={c:stats(v) for c,v in all_contrasts.items()},
        total_seconds=result['total_seconds'], guard=guard, peak_mps_boundary_bytes=result['peak_mps_boundary_bytes'],
        complete=True, original_quality_decision_changed=False, test_opened=False, validation_scored=False,
        limitations='Three parent states each seeded by one CE update, eight correlated cases per state, shared8192-target training evaluation per parent. No bootstrap, population inference, final quality claim or model adoption. Identity/zero repeat the same deterministic operation; their eight cases are not independent replicates.')
    with (OUT/'analysis.json').open('x') as f: json.dump(summary,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('sources','parents')},indent=2))


if __name__ == '__main__': main()
