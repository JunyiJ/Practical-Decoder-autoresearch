"""Read-only pilot completion/resource review; retain every file, no quality launch."""
import json,shutil
from pathlib import Path
from datetime import datetime,timezone
import torch
from research.ffn_structure_v1 import pilot_contract as c,pilot_execution as e
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
B=c.EVIDENCE
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent),session_lock(c.RUN):
    e.idle()
    with CpuPrerequisiteGuard(B/'pilot001-review-guard.json',c.ROOT,timeout_seconds=120,poll_seconds=.25):
        plan=e.bind(reference(e.FREEZE)['sha256']);inference=read(reference(B/'pilot001-inference-result.json'))
        assert inference['status']=='resource_pilot_pair_inference_passed' and read(inference['guard'])['status']=='finished'
        rows=[];files={};used=0;weight_bytes=0
        for slot,arm in enumerate(c.ARMS):
            anchor=read(reference(e.anchor(slot)));assert anchor['status']=='accepted'
            for item in anchor['files']:
                assert e.ref(item['path'])==item
                q=Path(item['path']);size=q.stat().st_size;used+=size
                files[str(q.relative_to(c.ROOT))]=dict(size=size,sha256=item['sha256'])
                if q.suffix=='.pt':weight_bytes+=size
            receipt=read(anchor['acceptance']);accepted=receipt['artifact_acceptance'];assert accepted['accepted']
            root=c.output(slot);worker=read(e.ref(root/'worker-guard.json'));parent=read(anchor['parent_guard']);process=read(receipt['process'])
            assert worker['status']==parent['status']=='finished' and worker['reason'] is None and parent['reason'] is None
            assert process['status']=='completed' and process['returncode']==0 and not e._group_exists(process['pid'])
            assert accepted['audit']['completed_steps']==20 and accepted['audit']['tokens_processed']==20480 and accepted['audit']['ema_snapshots']==8
            assert accepted['validation']['target']['targets']==plan['data']['validation_targets']==34278
            summary=read(accepted['files']['summary'])
            rows.append(dict(slot=slot,arm=arm,acceptance=reference(e.anchor(slot)),training_seconds=summary['training_time_sec'],
                worker_seconds=worker['elapsed_seconds'],parent_inclusive_seconds=parent['elapsed_seconds'],worker_peak_rss=worker['observation']['peak_rss_bytes'],
                worker_peak_driver=worker['observation']['peak_observed_mps_driver_bytes'],audit=accepted['audit'],
                validation_targets=accepted['validation']['target']['targets'],raw_bytes=Path(accepted['files']['parent']['path']).stat().st_size,
                h200_bytes=Path(accepted['files']['h200']['path']).stat().st_size,
                quality_horizon_linear_training_projection_seconds=summary['training_time_sec']*60))
        remaining=c.REQUIRED_FREE-used;free=shutil.disk_usage(c.ROOT).free;assert free>=remaining
        snap=B/'pilot001-completed-evidence';snap.mkdir()
        for name in files:
            source=c.ROOT/name
            if source.suffix!='.pt':
                dest=snap/source.relative_to(c.RUN);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(source.read_bytes())
        result=dict(status='both_resource_pilots_and_inference_accepted',at=datetime.now(timezone.utc).isoformat(),
            source=reference(Path(__file__).resolve()),freeze=reference(e.FREEZE),rows=rows,files=files,
            inference=reference(B/'pilot001-inference-result.json'),retained_pilot_weight_bytes=weight_bytes,
            consumed_pilot_artifact_bytes=used,remaining_full_panel_free_requirement_bytes=remaining,observed_free_bytes=free,
            margin_bytes=free-remaining,retention='Both accepted pilot pairs remain retained; no cleanup credit or deletion. Preserve all selected/fresh/failed/fixture artifacts.',
            quality_scope='Resource pilots excluded from LR selection/quality inference. Linear1200step timing projection is a rough estimate, not a guarantee;7200s trial guard remains.',
            inference_scope='Two short-pilot models resident; allocation/loading excluded, descriptive timing only. Cache196608B/item. No cost-only adoption route.',
            quality_plan_frozen=False,quality_launch_permitted=False,holdout_scored=False,
            next_action='Implement/test only finite quality1200step config/worker acceptance and exact losing-screen retention using proven primitives in an additive namespace outside frozen pilot sources; freeze full3LR-per-arm plus3freshpaired-seed plan and resource caps before dispatch. Keep both pilots and all fresh/selected pairs.')
        c.verify_sources(plan['sources']);e.exclusive(B/'pilot001-review.json',result)
    e.idle();print(json.dumps(dict(status=result['status'],weights_bytes=weight_bytes,free_bytes=free,required_remaining_bytes=remaining,margin_bytes=free-remaining)),flush=True)
