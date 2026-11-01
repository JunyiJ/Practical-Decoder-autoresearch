"""Frozen measurement contract on both accepted scientific-corpus resource pilots."""
import json,shutil
from pathlib import Path
import torch
from research.ffn_structure_v1 import pilot_contract as c,pilot_execution as e,inference as inf
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
B=c.EVIDENCE
torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent),session_lock(c.RUN):
    e.idle();plan=e.bind(reference(e.FREEZE)['sha256']);bindings={};anchors=[]
    for slot,arm in enumerate(c.ARMS):
        anchor=read(reference(e.anchor(slot)));assert anchor['status']=='accepted' and anchor['slot']==slot
        for item in anchor['files']:assert e.ref(item['path'])==item
        assert read(anchor['parent_guard'])['status']=='finished'
        accepted=read(anchor['acceptance'])['artifact_acceptance'];assert accepted['accepted']
        bindings[arm]={k:accepted['files'][k] for k in ('config','parent','h200')};anchors.append(reference(e.anchor(slot)))
    measurement_plan=dict(contract=inf.CONTRACT,bindings=bindings,pair_index=0,runner=reference(Path(__file__).resolve()),sources=plan['sources'],
        source_acceptance=anchors,scope='Both20step scientific-corpus pilots; descriptive timing only, no quality selection.')
    e.exclusive(B/'pilot001-inference-plan.json',measurement_plan)
    with e.Guard(B/'pilot001-inference-guard.json'):
        measured=inf.measure_pair(bindings,0);review=inf.review(measured,bindings,0)
        e.exclusive(B/'pilot001-inference-measurements.json',measured);c.verify_sources(plan['sources'])
        for slot in range(2):
            anchor=read(anchors[slot])
            for item in anchor['files']:assert e.ref(item['path'])==item
        result=dict(status='resource_pilot_pair_inference_passed',plan=reference(B/'pilot001-inference-plan.json'),measurements=reference(B/'pilot001-inference-measurements.json'),
            review=review,holdout_scored=False,quality_adopted=False,free_bytes=shutil.disk_usage(c.ROOT).free)
    result['guard']=reference(B/'pilot001-inference-guard.json');e.exclusive(B/'pilot001-inference-result.json',result)
    print(json.dumps(dict(status=result['status'],cells=review['cells'])),flush=True)
