"""Publish reviewed GQA quality records; never edit frozen training source."""
import json
import shutil
from pathlib import Path
from src.research.protocol import file_digest
from src.research.team.knowledge import KnowledgeBank
run=Path('research_runs/tokenizer-gqa-001')
state=json.loads((run/'state.json').read_text())
assert not json.loads((run/'plan.json').read_text())['pilot']
out=Path('research/tokenizer-evidence/gqa-001');out.mkdir(exist_ok=True)
rows=[];new=[]
for record in state['records']:
    review_path=Path('research_state/tokenizer-gqa-validation')/(record['id']+'-review.json')
    if record['status']!='completed' or not review_path.exists():continue
    review=json.loads(review_path.read_text());trial=run/record['id']
    assert review['status']=='passed' and review['result_sha256']==file_digest(trial/'result.json')
    target=out/record['id']
    if not target.exists():new.append(record)
    target.mkdir(exist_ok=True)
    for name in ('request.json','result.json','worker.log','metrics.jsonl'):shutil.copyfile(trial/name,target/name)
    shutil.copyfile(review_path,target/'review.json');rows.append(record)
for name in ('plan.json','selection.json'):
    if (run/name).exists():shutil.copyfile(run/name,out/name)
text=['## Reviewed quality trajectories','','Only completed, artifact-reviewed trajectories appear here. Quality eligibility','alone does not adopt GQA: cached correctness, measured inference cost and final','review are also required. The test holdout remains closed.','',
      '| Stage | Family | Width | Seed | LR | Raw CE | H200 CE | Training seconds |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    text.append(f"| {r['stage']} | {r['family']} | {r['width']} | {r['seed']} | {r['lr']:g} | {r['raw']['mean_ce']:.6f} | {r['h200']['mean_ce']:.6f} | {r['training']['training_time_sec']:.2f} |")
text+=['','Paired GQA minus MHA H200 CE (fixed-LR screens; selected-recipe confirmation):','']
for a in rows:
    if a['family']!='gqa':continue
    controls=[b for b in rows if b['family']=='mha' and b['stage']==a['stage'] and b['width']==a['width'] and b['seed']==a['seed'] and (a['stage']=='confirm' or b['lr']==a['lr'])]
    if controls:
        b=controls[0];text.append(f"- {a['stage']}, width{a['width']}, seed{a['seed']}, GQA LR{a['lr']:g} / MHA LR{b['lr']:g}: {a['h200']['mean_ce']-b['h200']['mean_ce']:+.6f}.")
text+=['',f"Frozen family LR selection: {state['selection']}" if state.get('selection') else 'Screening incomplete; no frozen family LR choices yet.','']
report=Path('research/TOKENIZER_GQA_RESULTS.md');body=report.read_text().split('\n## Reviewed quality trajectories')[0];report.write_text(body+'\n'+'\n'.join(text)+'\n')
ids=[]
for r in new:
    p=run/r['id']/'result.json'
    ids.append(KnowledgeBank('research_state/tokenizer-knowledge').add(dict(kind='result',title='Tokenizer GQA round '+r['id'],summary=f"Completed/audited {r['id']}, H200 CE {r['h200']['mean_ce']:.6f}. Partial evidence only; finish quality and cost gates before adoption. Test closed.",outcome='reported',created_by='Codex tokenizer campaign',artifacts=[dict(path=str(p.resolve()),sha256=file_digest(p))])))
if new:
    p=Path('research_state/tokenizer-quality-progress.json');d=json.loads(p.read_text());d.update(last_reviewed_trial=rows[-1]['id'],gqa_reviewed_count=len(rows),last_learning_id=ids[-1],latest_results_commit_pending=True);p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(new=[r['id'] for r in new],reviewed=len(rows)),indent=2))
