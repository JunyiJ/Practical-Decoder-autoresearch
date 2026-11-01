"""Preserve one accepted fresh endpoint and execute its frozen inference protocol."""
import argparse,importlib.util,json,shutil
from pathlib import Path
import torch
from research.ema_horizon_v1 import contract as c,execution as e
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter

p=argparse.ArgumentParser();p.add_argument('--slot',type=int,choices=range(3),required=True);args=p.parse_args()
slot=args.slot;B=e.EVIDENCE;stem=f'quality001-slot{slot:02d}'
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent) as g,session_lock(e.ROOT) as lock:
    e.idle()
    with e.Guard(B/f'{stem}-review-guard.json',600):
        plan=e.bind(MetadataWriter(e.ROOT,(g,lock)),'quality',reference(e.freeze_path('quality'))['sha256'])
        anchor=read(reference(e.anchor_path('quality',slot)));accepted=read(anchor['acceptance'])
        parent=read(anchor['parent_guard']);worker=read(reference(e.ROOT/f'control/worker-quality-{slot:02d}.json'))
        process=read(accepted['process'])
        assert accepted['stage']=='quality' and accepted['slot']==slot and accepted['artifact_acceptance']['accepted']
        assert accepted['freeze']==reference(e.freeze_path('quality'))
        for guard in (parent,worker):assert guard['status']=='finished' and guard['reason'] is None
        assert process['status']=='completed' and process['returncode']==0 and process['error'] is None
        request=read(accepted['request'])['acceptance_request']
        assert request['config']==plan['configs'][slot] and request['sources']==plan['sources'] and request['seeds']==plan['seeds']
        weights=[];portable=B/f'{stem}-evidence';portable.mkdir()
        for ref in anchor['files']:
            assert e.ref_file(ref['path'])==ref
            source=Path(ref['path'])
            if source.suffix=='.pt':weights.append(dict(ref,bytes=source.stat().st_size))
            else:
                dest=portable/source.relative_to(e.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        for source in sorted((e.ROOT/'control').glob(f'*quality-{slot:02d}.json')):
            dest=portable/source.relative_to(e.ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        assert len(weights)==3 and sum(x['bytes'] for x in weights)<c.TRAJECTORY_ALLOWANCE
        source=plan['fresh_inference']['source'];assert reference(source['path'])==source
        module_spec=importlib.util.spec_from_file_location('frozen_ema_timing',source['path'])
        module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
        launch=dict(freeze=reference(e.freeze_path('quality')),acceptance=reference(e.anchor_path('quality',slot)),
            protocol=plan['fresh_inference'],runner=reference(__file__),slot=slot,quality_adopted=False)
        e.exclusive(B/f'{stem}-inference-plan.json',launch)
        inference=module.measure(request,pair_index=slot)
        assert inference['status']=='passed' and c.source_inventory()==plan['sources'] and reference(source['path'])==source
        e.exclusive(B/f'{stem}-inference-result.json',dict(inference,plan=reference(B/f'{stem}-inference-plan.json')))
        ce={h:v['ce'] for h,v in accepted['artifact_acceptance']['validation'].items()}
        review=dict(status='accepted_with_fresh_inference',slot=slot,seed=plan['seeds']['fresh'][slot],
            acceptance=reference(e.anchor_path('quality',slot)),validation_ce=ce,h100_minus_h200=ce['100']-ce['200'],
            weights=weights,retained_weight_bytes=sum(x['bytes'] for x in weights),
            worker_guard=worker,parent_guard=parent,process_elapsed_seconds=process['elapsed_seconds'],
            inference=reference(B/f'{stem}-inference-result.json'),free_bytes=shutil.disk_usage(c.ROOT).free,
            holdout_scored=False,quality_adopted=False,note='One fresh trajectory; final decision requires all three frozen seeds and simultaneous-text statistics. No interim selection or early stopping.')
    e.exclusive(B/f'{stem}-review.json',dict(review,guard=reference(B/f'{stem}-review-guard.json')))
    print(json.dumps(dict(slot=slot,validation_ce=ce,delta=review['h100_minus_h200'],inference=inference['cells'])))
