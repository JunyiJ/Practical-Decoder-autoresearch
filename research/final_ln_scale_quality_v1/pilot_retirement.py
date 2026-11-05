"""Resolve the four accepted cross-root pilots; verify a later exact retirement.

No pilot CLI is invoked and no stopped directory is reopened for execution.
Deletion is a separate one-shot audited recovery action after integration passes.
"""
from pathlib import Path
from . import campaign_contract as c
from research.block_norm_v1.fixture_dispatch import read,reference
from research.final_ln_scale_campaign_v2 import pilot
B=pilot.B
DEST=B/'quality001-pilot-retirement001'
EXPECTED_BYTES=1118330653

def targets():
    terminal=read(reference(B/'pilot002-terminal-review.json'))
    report=read(terminal['resource_review'])
    c.require(terminal['status']=='resource_pilots_complete_quality_source_pending_capacity_blocked' and report['resource_checks_passed'] is True,'Accepted numerical resource pair required')
    plan=read(reference(B/'pilot002-plan.json'));rows=[]
    for item in terminal['accepted']:
        accepted=read(item['acceptance']);worker=read(item['worker'])
        c.require(accepted['worker']==item['worker'],'Exact immutable worker reference required')
        req=pilot.request_for(accepted,plan)
        c.require(accepted['acceptance']['accepted'] and accepted['slot']==item['slot'],'Independent accepted endpoint required')
        for kind in ('parent','h200'):
            ref=req['files'][kind]
            match=[r for r in terminal['weights'] if r['path']==ref['path'] and r['sha256']==ref['sha256'] and r['arm']==item['arm']]
            c.require(len(match)==1,'Accepted cross-root weight binding differs');rows.append(dict(match[0],acceptance=item['acceptance'],role=kind))
    c.require(len(rows)==4 and len({r['path'] for r in rows})==4 and sum(r['size'] for r in rows)==EXPECTED_BYTES,'Exact four pilot payloads required')
    allowed={str(pilot.OLDROOT/'evidence/pilots/00/checkpoints'),str(pilot.ROOT/'evidence/pilots/01/checkpoints')}
    c.require({str(Path(r['path']).parent) for r in rows}==allowed,'Only the accepted cross-root endpoints are eligible')
    return rows

def verify_retirement(ref):
    done=read(ref);manifest=read(done['manifest'])
    c.require(done['status']=='completed' and done['deleted_bytes']==EXPECTED_BYTES and done['selected_fresh_failed_unchanged'] and done['original_evidence_unchanged'],'Completed exact pilot retirement required')
    closure=read(manifest['closure']);c.require(closure['status']=='accepted_resource_pilot_tensor_dependencies_closed','Actual downstream dependency closure required')
    c.verify_sources(read(closure['plan'])['sources'])
    expected=targets();actual=manifest['delete_files']
    c.require({(x['path'],x['size'],x['sha256']) for x in actual}=={(x['path'],x['size'],x['sha256']) for x in expected},'Retirement allowlist differs')
    c.require(done['deleted_paths']==sorted(x['path'] for x in expected),'Retired paths differ')
    for row in actual:c.require(not Path(row['path']).exists(),'Permanently retired tensor unexpectedly present')
    for evidence in manifest['eligibility']:read(evidence)
    c.require(read(reference(B/'pilot001-run-0-failure.json'))['error'] and (pilot.OLDROOT/'STOP').exists(),'Original failure remains closed')
    return done
