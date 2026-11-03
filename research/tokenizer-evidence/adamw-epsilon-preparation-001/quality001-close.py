"""Close the fully reviewed finite panel, preserving all retained weights."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import torch
from research.adamw_epsilon_quality_v1 import quality_execution as q, campaign_contract as c
from research.block_norm_v1.fixture_dispatch import reference, read
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--freeze-sha256',required=True);SHA=parser.parse_args().freeze_sha256
B = q.EVIDENCE
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as global_fd, session_lock(q.ROOT) as campaign_fd:
    q.idle()
    guard_path = B / 'quality001-closure-guard.json'
    c.require(not guard_path.exists() and not (B / 'quality001-closure.json').exists(), 'No closure replacement')
    with CpuPrerequisiteGuard(guard_path, q.ROOT, timeout_seconds=120, poll_seconds=.25):
        writer = MetadataWriter(q.ROOT, (global_fd, campaign_fd))
        plan = q.bind(writer, SHA)
        current = q.state(plan, SHA)
        final_ref = reference(B / 'quality001-final-review.json')
        final = read(final_ref)
        anchors = [reference(q.anchor_path(n)) for n in range(6)]
        c.require(len(current['records']) == 6 and final['accepted_anchors'] == anchors, 'Complete reviewed panel required')
        c.require(final['freeze'] == dict(path=str(q.FREEZE), sha256=SHA), 'Review freeze differs')
        c.require(final['status'] == 'reviewed_not_adopted' and final['strict_quality'] is False
                  and final['quality_adopted'] is False and final['holdout_scored'] is False, 'Non-adoption closure only')
        c.require(reference(Path(final['runner']['path'])) == final['runner'], 'Review source changed')
        guard = read(final['guard'])
        c.require(guard['status'] == 'finished' and guard['reason'] is None, 'Final review guard failed')
        for index, ref in enumerate(final['inference_reviews']):
            review = read(ref)
            c.require(review['status'] == 'passed' and review['pair_index'] == index, 'Paired inference incomplete')
            read(review['result'])
            g = read(review['guard'])
            c.require(g['status'] == 'finished' and g['reason'] is None, 'Inference guard failed')
        c.require(len(final['inference_reviews']) == 3, 'Three paired inference reviews required')
        value = dict(status='closed_not_adopted', at=datetime.now(timezone.utc).isoformat(),
                     runner=reference(Path(__file__).resolve()), final_review=final_ref,
                     freeze=final['freeze'], accepted_anchors=anchors, accepted_endpoints=6, fresh_pairs=3,
                     statistics=final['statistics'], gate_checks=final['gate_checks'], decision=final['decision'],
                     inference_reviews=final['inference_reviews'], selection=None,
                     retirement=plan['pilot_retirement'], sources_unchanged=len(plan['sources']),
                     retained_quality_weight_bytes=current['used_weights'], free_bytes=shutil.disk_usage(q.ROOT).free,
                     selected_and_all_fresh_pairs_protected=True, active_worker=False,
                     training_launched=False, holdout_scored=False, quality_adopted=False, strict_quality=False,
                     next_action='Review architecture and bounded HP hypotheses; select one distinct prerequisite question outside frozen namespaces.')
        writer.publish_json('STOP', dict(status='completed_closed', final_review=final_ref,
                                        reason='Finite panel and inference complete; retain incumbent. No retry or extension.'))
    value['guard'] = reference(guard_path)
    c.require(read(value['guard'])['status'] == 'finished', 'Closure guard failed')
    q.exclusive(B / 'quality001-closure.json', value)
    print(json.dumps(dict(status=value['status'], decision=value['decision'], statistics=value['statistics'])))
