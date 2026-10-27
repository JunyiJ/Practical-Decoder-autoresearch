"""Audit completed saved records and publish portable evidence; no training/scoring."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
from src.research.protocol import file_digest, digest
from src.research.runner import atomic_json, source_fingerprint
from src.research.tokenizer_hybrid_study import choose

root=Path.cwd(); run=root/'research_runs/tokenizer-hybrid-001'
out=root/'research/tokenizer-evidence/hybrid-001'
plan=json.loads((run/'plan.json').read_text());state=json.loads((run/'state.json').read_text())
assert digest(plan)==state['plan_sha256'] and source_fingerprint()==plan['source']
count=0
for row in state['records']:
    if row['status']!='completed':continue
    trial=run/row['id'];target=out/row['id'];target.mkdir(exist_ok=True)
    if not (target/'review.json').exists():
        subprocess.run([sys.executable,'-m','research.tokenizer-evidence.hybrid-001.audit_trial',str(trial)],check=True)
    review=json.loads((target/'review.json').read_text())
    assert review['status']=='passed' and review['result_sha256']==file_digest(trial/'result.json')
    for name in ('request.json','result.json','worker.log','metrics.jsonl','training-summary.json'):
        if (trial/name).exists():shutil.copyfile(trial/name,target/name)
    count+=1
for name in ('plan.json','state.json','selection.json','controller-error.json'):
    if (run/name).exists():shutil.copyfile(run/name,out/name)
if state.get('selection'):
    assert choose(state['records'])==state['selection']
    atomic_json(out/'screen-review.json',dict(status='passed',selection=state['selection'],
                scope='One-screen-seed LR and control selection only; fresh pairs required.'))
lines=['# Tokenizer NoPE hybrid results','','The finite width64 comparison is '+state['status']+'. '+str(count)+' completed quality trials have passed saved-artifact audit. No adoption yet. See [protocol](TOKENIZER_HYBRID_PROTOCOL.md).','','| Stage | Family | Seed | LR | H200 CE | Training seconds |','|---|---|---:|---:|---:|---:|']
for r in state['records']:
    if r['status']=='completed':lines.append(f"| {r['stage']} | {r['family']} | {r['seed']} | {r['lr']} | {r['h200']['mean_ce']:.6f} | {r['training']['training_time_sec']:.2f} |")
lines += ['', 'The hybrid is a four-layer dense NoPE KDA/KDA/KDA/gated-MLA adaptation. All three layouts receive equal two-LR screening; only hybrid and the selected homogeneous control receive fresh confirmation. Screening alone establishes no improvement. Quality gates require all fresh pairs and conditional intervals; final serial cache/cost measurements and artifact review are required before adoption.', '', 'All research checkpoints and preflight failure evidence are preserved. Test loss remains closed; this is adaptive validation evidence at width64 only. The earlier width256 GQA and RoPE MLA options remain separate.']
(root/'research/TOKENIZER_HYBRID_RESULTS.md').write_text('\n'.join(lines)+'\n')
p=root/'research_state/tokenizer-quality-progress.json';d=json.loads(p.read_text());d.update(status='hybrid_quality_'+state['status'],hybrid_reviewed_count=count,hybrid_selection=state.get('selection'),active_run=str(run),note='Source-frozen12-trial width64 panel. Quality completion requires final serial cost review before adoption; no new source edits until finite round closes.')
atomic_json(p,d);print(json.dumps(dict(status=state['status'],reviewed_count=count,selection=state.get('selection'))))
