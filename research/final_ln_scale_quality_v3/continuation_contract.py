"""Exact import/source bridge for the same frozen six-trajectory panel."""
import copy
from pathlib import Path
from . import campaign_contract as c
from research.block_norm_v1.fixture_dispatch import read,reference
B=c.ROOT/'research/tokenizer-evidence/final-ln-scale-preparation-001'
OLDROOT=c.ROOT/'research_runs/tokenizer-final-ln-scale-quality-v1-001'

def verify_imports(plan):
    original=read(plan['original_plan']);failure=read(plan['entry_failure'])
    c.predecessor.verify_sources(original['sources'])
    c.require(plan['original_root']==str(OLDROOT) and plan['imported_count']==3,'Exact three-endpoint continuation required')
    c.require(plan['seeds']==original['seeds'] and plan['fresh']==original['fresh'] and plan['design']==original['design'],'Same frozen scientific panel required')
    c.require(plan['entry_headroom_bytes']==c.ENTRY_HEADROOM==failure['observed_filesystem_consumption_between_checks'],'Measured entry headroom differs')
    c.require(failure['status']=='CLOSED_FAILED_ENTRY_CAPACITY_ZERO_WORKER_UPDATES' and failure['optimizer_updates']==0 and failure['worker_spawned'] is False,'Original zero-update failure required')
    c.require(plan['original_STOP']==failure['original_STOP'] and read(plan['original_STOP'])['slot']==3,'Original STOP binding differs')
    c.require(not(OLDROOT/'evidence/quality/03').exists() and not(OLDROOT/'control/process-quality-03.json').exists(),'Original unstarted numerical endpoint changed')
    c.require(len(plan['original_anchors'])==3 and plan['original_anchors']==failure['accepted_anchors'],'Exact original accepted prefix required')
    for i,ref in enumerate(plan['original_anchors']):
        c.require(ref==reference(B/f'quality001-accepted-{i:02d}.json'),'Original anchor differs');anchor=read(ref);receipt=read(anchor['acceptance'])
        c.require(anchor['slot']==receipt['slot']==i and receipt['artifact_acceptance']['accepted'],'Original endpoint not independently accepted')
        c.require(receipt['freeze']==failure['original_freeze'],'Original scientific freeze differs')
    c.require(all(plan['sources'].get(n)==h for n,h in original['sources'].items()),'Original numerical source changed')
    return original

def resolve_request(receipt,plan,kind):
    worker=read(receipt['worker'])
    if receipt.get('imported') is True:
        original=verify_imports(plan);prior=read(receipt['original_acceptance'])
        c.require(receipt['worker']==prior['worker'] and worker['kind']==original['kind'],'Imported worker provenance differs')
        request=copy.deepcopy(worker['acceptance_request']);c.require(request['sources']==original['sources'],'Original worker source differs')
        request['sources']=plan['sources']
    else:
        c.require(worker['kind']==kind,'Continuation worker identity differs');request=worker['acceptance_request']
    c.require(request['sources']==plan['sources'] and c.fingerprint(request)==receipt['request_sha256'] and worker['holdout_scored'] is False,'Immutable request/source binding differs')
    return request
