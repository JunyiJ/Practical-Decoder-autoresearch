"""Record a completed frozen MLA score quality slot without training, scoring or deletion."""
import argparse,json,shutil
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.mla_score_quality_v1 import quality_execution as q,campaign_contract as c
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
p=argparse.ArgumentParser();p.add_argument('--slot',type=int,required=True);args=p.parse_args();assert 0<=args.slot<12
torch.set_num_threads(4);B=q.EVIDENCE;tag=f'quality001-slot{args.slot:02d}'
with session_lock(GLOBAL_LOCK.parent) as g,session_lock(q.ROOT) as lock:
    q.idle()
    with CpuPrerequisiteGuard(B/(tag+'-record-guard.json'),c.ROOT,timeout_seconds=120,poll_seconds=.25):
        writer=MetadataWriter(q.ROOT,(g,lock));digest=reference(q.FREEZE)['sha256'];plan=q.bind(writer,digest);current=q.state(plan,digest)
        assert len(current['records'])==args.slot+1
        receipt=current['receipts'][args.slot];anchor=read(reference(q.anchor_path(args.slot)))
        worker=read(reference(q.ROOT/f'evidence/quality/{args.slot:02d}/worker-guard.json'));parent=read(anchor['parent_guard']);process=read(receipt['process'])
        assert worker['status']==parent['status']=='finished' and worker['reason'] is None and parent['reason'] is None and process['status']=='completed'
        assert not q._group_exists(process['pid'])
        accepted=receipt['artifact_acceptance'];root=q.ROOT/f'evidence/quality/{args.slot:02d}'
        snap=B/(tag+'-evidence');snap.mkdir()
        for source in root.rglob('*'):
            if source.is_file() and source.suffix!='.pt':
                assert source.stat().st_size<=4*1024**2 and not source.name.endswith('.tmp')
                used=sum(p.stat().st_size for p in B.rglob('*') if p.is_file() and p.relative_to(B).parts[0].startswith('quality001'))
                assert used+source.stat().st_size<=32*1024**2
                dest=snap/source.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        required=max(8*1024**3,q.FULL_FREE-current['used_weights']);free=shutil.disk_usage(c.ROOT).free
        result=dict(status='accepted',at=datetime.now(timezone.utc).isoformat(),slot=args.slot,spec=current['records'][args.slot],
            anchor=reference(q.anchor_path(args.slot)),freeze=reference(q.FREEZE),runner=reference(Path(__file__).resolve()),
            audit=accepted['audit'],validation_targets=accepted['validation']['target']['targets'],h200_ce=accepted['validation']['ce'],
            worker_guard=worker,parent_guard=parent,process=process,accepted_prefix=len(current['records']),
            current_retained_quality_weight_bytes=current['used_weights'],remaining_required_free_bytes=required,observed_free_bytes=free,
            capacity_fits=free>=required,quality_adopted=False,holdout_scored=False,
            scope='One accepted fixed endpoint; no interim selection, quality conclusion, seed replacement or grid extension.',
            next_action='screen boundary and audited selection/retention' if args.slot==5 else 'final paired inference and independent review' if args.slot==11 else f'frozen quality slot{args.slot+1:02d}')
        q.exclusive(B/(tag+'-result.json'),result)
    print(json.dumps(dict(status='accepted',slot=args.slot,h200_ce=result['h200_ce'],capacity_fits=result['capacity_fits'],next_action=result['next_action'])),flush=True)
