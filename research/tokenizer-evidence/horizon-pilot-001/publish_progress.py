import json,shutil
from pathlib import Path
from src.research.protocol import file_digest
from src.research.team.knowledge import KnowledgeBank
run=Path('research_runs/tokenizer-horizon-001');state=json.loads((run/'state.json').read_text())
out=Path('research/tokenizer-evidence/horizon-001');out.mkdir(exist_ok=True)
rows=[];new=[]
for r in state['records']:
 rev=Path('research_state/tokenizer-horizon-validation')/(r['id']+'-review.json')
 if r['status']!='completed' or not rev.exists():continue
 review=json.loads(rev.read_text());trial=run/r['id']
 assert review['status']=='passed' and review['result_sha256']==file_digest(trial/'result.json')
 dest=out/r['id']
 if not dest.exists():new.append(r)
 dest.mkdir(exist_ok=True)
 for name in ['request.json','result.json','worker.log','metrics.jsonl']:shutil.copyfile(trial/name,dest/name)
 shutil.copyfile(rev,dest/'review.json');rows.append(r)
for name in ['plan.json','selection.json']:
 if (run/name).exists():shutil.copyfile(run/name,out/name)
text=['## Reviewed trajectories','','Only completed, artifact-reviewed records appear here. Test remains closed.','',
 '| Stage | Seed | Updates | Raw CE | H200 CE | Training seconds |','|---|---:|---:|---:|---:|---:|']
for r in rows:text.append(f"| {r['stage']} | {r['seed']} | {r['steps']} | {r['raw']['mean_ce']:.6f} | {r['h200']['mean_ce']:.6f} | {r['training']['training_time_sec']:.2f} |")
text+=['',f"Frozen selection: {state['selection']} updates." if state['selection'] is not None else 'Screening incomplete; no duration selected.','']
for r in rows:
 if r['stage']=='confirm' and r['steps']!=3000:
  refs=[b for b in rows if b['stage']=='confirm' and b['seed']==r['seed'] and b['steps']==3000]
  if refs:text.append(f"- Seed{r['seed']}, candidate minus3000 H200 CE: {r['h200']['mean_ce']-refs[0]['h200']['mean_ce']:+.6f}.")
p=Path('research/TOKENIZER_HORIZON_RESULTS.md');prefix=p.read_text().split('\n## Reviewed trajectories')[0];p.write_text((prefix+'\n'+'\n'.join(text)).rstrip()+'\n')
for r in new:
 artifact=run/r['id']/'result.json'
 key=KnowledgeBank('research_state/tokenizer-knowledge').add(dict(kind='result',title='Tokenizer horizon '+r['id'],summary=f"Audited {r['steps']}-update trial seed{r['seed']}, H200 CE{r['h200']['mean_ce']:.6f}. Finish frozen fresh confirmation before adoption; test closed.",outcome='reported',created_by='Codex tokenizer campaign',artifacts=[dict(path=str(artifact.resolve()),sha256=file_digest(artifact))]))
if new:
 p=Path('research_state/tokenizer-quality-progress.json');d=json.loads(p.read_text());d.update(horizon_reviewed_count=len(rows),last_reviewed_trial=rows[-1]['id'],last_learning_id=key,latest_results_commit_pending=True);p.write_text(json.dumps(d,indent=2)+'\n')
print(json.dumps(dict(new=[r['id'] for r in new],reviewed=len(rows)),indent=2))
