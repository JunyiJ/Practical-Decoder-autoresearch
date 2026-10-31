"""Preserve and close the failed pilot process; no replay, deletion or acceptance."""
import json,hashlib,shutil,os
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.block_norm_v1 import pilot_execution as p
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_serial_process import _group_exists
b=p.EVIDENCE;torch.set_num_threads(4)
with p.session_lock(p.GLOBAL_LOCK.parent) as g,p.session_lock(p.ROOT) as r:
 p.idle()
 with CpuPrerequisiteGuard(b/'pilot001-failure-guard.json',p.c.ROOT,timeout_seconds=120,poll_seconds=.25):
  plan=json.loads((b/'pilot001-plan.json').read_text());p.c.verify_sources(plan['sources']);p.c.frozen_predecessors(plan['sources'])
  trial=p.ROOT/'evidence/pilots/00';process=json.loads((trial/'process/process.json').read_text());guard=json.loads((p.ROOT/'control/worker-00-guard.json').read_text())
  assert process['status']=='failed' and process['returncode']==1 and not _group_exists(process['pid']) and not _group_exists(process['parent_pid'])
  assert 'cleanup_error' not in process and not (p.ROOT/'evidence/pilots/01').exists()
  assert not (b/'pilot001-accepted-00.json').exists() and not (trial/'accepted.json').exists()
  writer=p.MetadataWriter(p.ROOT,(g,r));writer.publish_json('STOP',dict(status='closed_failed_pilot001',reason='RMS20updates/raw/H200 completed; post-training validation adapter rejected standard train/val/test loader. No accepted pilot receipt; never retry or continue this root.',retry_allowed=False))
  inventory={};weights={};out=b/'pilot001-failure-evidence';out.mkdir()
  for path in sorted(p.ROOT.rglob('*')):
   if not path.is_file():continue
   name=str(path.relative_to(p.ROOT));weight=path.suffix=='.pt'
   row=dict(size=path.stat().st_size,sha256=p.c.file_sha(path,root=p.ROOT,max_bytes=(512 if weight else 2)*1024**2));inventory[name]=row
   if weight:weights[name]=row
   elif path.name not in ('.lock','.metadata.lock'):
    dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True)
    with dest.open('xb') as f:f.write(path.read_bytes())
    assert hashlib.sha256(dest.read_bytes()).hexdigest()==row['sha256']
  summary=json.loads((trial/'training-summary.json').read_text());export=json.loads((trial/'export.json').read_text())
  assert summary['completed_steps']==20 and summary['stop_reason']=='max_iters' and len(weights)==2
  result=dict(status='closed_failed_post_training_validation',observed_at=datetime.now(timezone.utc).isoformat(),root=str(p.ROOT),freeze=p.reference(p.FREEZE),
   process=process,guard=guard,completed_updates=20,tokens_processed=20480,training_seconds=summary['training_time_sec'],protected_weights=weights,
   inventory=inventory,export=export,slot1_started=False,accepted_pilots=0,quality_training_launched=False,holdout_scored=False,deletion_performed=False,
   diagnosis='validation.load incorrectly required exactly train/val split keys. Generated80/20 fixtures have2; actual frozen80/10/10 checkpoint_loader exposes3. Training and H200 export succeeded, but v.load raised before full validation scoring/model construction; cache and parent acceptance never ran.',
   holdout_scope='Frozen inherited loader constructs/encodes the held-out split and a private test generator; training calls only train batches and sampled val evaluation. No test loss, test batch training or test-based decision occurred. Loader presence is not evidence of test scoring; do not claim test was never tokenized.',
   preserved_source='pilot-acceptance-001-source plus388-source frozen plan; original src/requirements and13recovery modules unchanged',
   followup='Keep this root/STOP/failed process and both weights unchanged. Correct the read-only validation/inference loader interface with an explicitly train/val-only view and actual80/10/10 boundary tests. Independently audit saved RMS outputs in a separate recovery evidence namespace without retraining. A separate finite continuation may carry verified RMS evidence and run the unstarted LayerNorm pilot; it must not relabel this failed process as successful.',
   no_retry=True,free_bytes_after=shutil.disk_usage(p.ROOT).free)
  p.exclusive(b/'pilot001-failure.json',result)
 print(json.dumps(dict(status=result['status'],completed_updates=20,protected_weights=2,slot1_started=False,free_bytes=result['free_bytes_after'])))
