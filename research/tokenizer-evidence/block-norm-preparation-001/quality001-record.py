"""Record one externally accepted quality endpoint; never dispatch or adopt."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone
parser=argparse.ArgumentParser();parser.add_argument('--slot',required=True,type=int);args=parser.parse_args();slot=args.slot;assert 0<=slot<12
b=Path('research/tokenizer-evidence/block-norm-preparation-001');root=Path('research_runs/tokenizer-block-norm-v2-quality-001');sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();read=lambda p:json.loads(Path(p).read_text())
p=Path('research/TOKENIZER_RESUME_STATE.json');s=read(p)
for n,h in s['evidence_sha256'].items():assert sha(n)==h,n
anchorpath=b/f'quality001-accepted-{slot:02d}.json';anchor=read(anchorpath);a=read(anchor['acceptance']['path']);assert sha(anchor['acceptance']['path'])==anchor['acceptance']['sha256'] and a['artifact_acceptance']['accepted']
base=root/f'evidence/quality/{slot:02d}';process=read(base/'process/process.json');summary=read(base/'training-summary.json');guard=read(base/'worker-guard.json');parent=read(anchor['parent_guard']['path'])
assert process['status']=='completed' and process['returncode']==0 and process['error'] is None and guard['status']==parent['status']=='finished'
for ref in anchor['files']:assert sha(ref['path'])==ref['sha256']
portable=b/f'quality001-slot{slot:02d}-evidence';portable.mkdir();inventory={}
for f in sorted(base.rglob('*')):
 if f.is_file():
  rel=f.relative_to(base);inventory[str(rel)]={'size':f.stat().st_size,'sha256':sha(f)}
  if f.suffix!='.pt':
   dest=portable/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(f.read_bytes())
(portable/'parent-guard.json').write_bytes(Path(anchor['parent_guard']['path']).read_bytes())
now=datetime.now(timezone.utc).isoformat();spec=a['request']['trial']['spec'];ce=a['artifact_acceptance']['validation']['ce']
result=dict(status='quality_endpoint_independently_accepted',observed_at=now,slot=slot,spec=spec,anchor_sha256=sha(anchorpath),inventory=inventory,audit=a['artifact_acceptance']['audit'],h200_ce=ce,training_seconds=summary['training_time_sec'],worker_guard=guard,parent_guard=parent,process=process,free_bytes=shutil.disk_usage('.').free,quality_adopted=False,selection_saved=slot>=6,holdout_scored=False)
with (b/f'quality001-slot{slot:02d}-result.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
note=f'\nQuality001 slot{slot:02d} accepted ({now}): {spec["arm"]}, seed{spec["seed"]}, LR{spec["lr"]}, 1,200 updates/1,228,800 targets, H200 full-validation CE{ce:.12f}. Independent parent tensor/optimizer/EMA/validation/cache audits passed and the child exited cleanly. This is one endpoint; no adoption or partial-screen selection. Evidence: quality001-slot{slot:02d}-result.json and accepted-{slot:02d}.json.\n'
docs=['research/TOKENIZER_BLOCK_NORM_PREPARATION.md','research/TOKENIZER_QUALITY_RESULTS.md']
for n in docs:Path(n).write_text(Path(n).read_text()+note)
freeze=read(b/'quality001-preflight.json')['freeze'];nextstep=f'run --slot {slot+1}' if slot<5 or 6<=slot<11 else ('boundary' if slot==5 else 'FINAL_REVIEW_AND_THREE_FRESH_INFERENCE_PAIRS_NO_AUTO_ADOPTION')
next_action=(f'QUALITY001 SLOT{slot:02d} ACCEPTED. See quality001-slot{slot:02d}-result/accepted/evidence. {spec["arm"]} LR{spec["lr"]} seed{spec["seed"]}1200steps, fullvalH200CE{ce:.12f}, all parent audits passed. NEXT {nextstep} via research.block_norm_v1.quality_execution --freeze-sha256 {freeze["sha256"]}, strictMPSenv4threads. Inspect actual process/request/anchor state first; never retry partialattempt. Existing q.state validates accepted prefix, sources, files, targets and remainingcapacity; each child7200s totalparentremainingdeadline,4GiBrss6GiBdriver8GiBreserve. Complete all6screens before saving perarmLR selection and exact audited4loserpair cleanup boundary, then all6fresh endpoints. Selectedscreens/allfresh/failureweights alwaysprotected,bothpilotskept. Preserve source namespace frozen395files; recorder/evidence/docs outside namespace allowed. Original failed RMS process remainsclosed; no oldcampaign/fixture reopens. Finalqualityreview+3freshinferencepairs before anyadoption. No testloss,nopaidcompute/subagents,oneworker,oldheartbeatPAUSED;SiLUremainsadopted.')
s.update(observed_at=now,current_research_phase='block_norm_quality001_running',next_action=next_action,first_action=next_action,next_after_starter=next_action,active_run=None,active_campaign_root=str(root.absolute()),block_norm_quality_last_accepted_slot=slot)
files=docs+[str(anchorpath),str(Path(__file__))]+[str(f) for f in b.glob(f'quality001-slot{slot:02d}-*.json')]+[str(f) for f in portable.rglob('*') if f.is_file()]
for n in files:s['evidence_sha256'][n]=sha(n)
p.write_text(json.dumps(s,indent=2)+'\n');p=Path('research_state/tokenizer-quality-progress.json');d=read(p);d.update({k:s[k] for k in ['observed_at','current_research_phase','next_action','active_run','active_campaign_root','block_norm_quality_last_accepted_slot']});d.update(commit_pending=True,push_pending=True);p.write_text(json.dumps(d,indent=2)+'\n')
Path('/tmp/quality-result-stage.nul').write_bytes(b'\0'.join(x.encode() for x in sorted(set(files+['research/TOKENIZER_RESUME_STATE.json'])))+b'\0');print(json.dumps(dict(slot=slot,h200_ce=ce,training_seconds=summary['training_time_sec'],next=nextstep)))
