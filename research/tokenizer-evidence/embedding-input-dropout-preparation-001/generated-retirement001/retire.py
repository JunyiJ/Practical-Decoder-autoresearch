"""Exact four obsolete generated payloads after real acceptance and pilot-source closure."""
import json,os,shutil,gc
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_quality_v1.retention import record,inventory,consumer_check,syncdir
from research.embedding_input_dropout_worker_v2.execution import sources,checked
from research.embedding_input_dropout_campaign_v1.pilot import campaign_sources
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=E.parent

def publish(p,d):
    raw=(json.dumps(d,indent=2,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=512*1024,'Bounded retirement record')
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not(E/'manifest.json').exists() and not(E/'completed.json').exists(),'Closed retirement cannot rerun')
    with CpuPrerequisiteGuard(E/'guard.json',ROOT,timeout_seconds=1800,poll_seconds=.25):
        integration_ref=reference(B/'pilot-integration001.json');integration=checked(integration_ref)
        require(integration['status']=='passed_generated_dependencies_closed_for_finite_resource_pilots' and integration['generated_tensor_dependencies_closed'] is True
            and integration['sources']==sources() and integration['campaign_sources']==campaign_sources(),'Actual complete dependency closure required')
        review_ref=reference(B/'worker002-review.json');review=checked(review_ref);require(review['status']=='two_generated_workers_independently_accepted','Actual complete generated pair required')
        expected={str(ROOT/f'research_runs/tokenizer-embedding-input-dropout-worker-v1-{attempt}/evidence/pilots/{slot}/checkpoints/{name}') for attempt,slot in [('001','00'),('002','01')] for name in ('parent.pt','h200.pt')}
        bindings=[]
        for row in review['rows']:
            w=checked(row['worker']);a=row['acceptance'];require(a['status']=='accepted' and a['step']==4 and a['targets']==4096 and w['config']['training']['seed']==600406,'Only accepted four-update fixture payloads eligible')
            bindings.extend(w[n] for n in ('parent','h200'))
        require(len(bindings)==4 and {v['path'] for v in bindings}==expected,'Complete exact cross-root allowlist required')
        stopped=[]
        for attempt in ('001','002'):
            p=ROOT/f'research_runs/tokenizer-embedding-input-dropout-worker-v1-{attempt}/STOP';require(p.exists(),'Both source executions terminal');stopped.append(reference(p))
        delete=[]
        for b in bindings:
            r=record(Path(b['path']));require(r['sha256']==b['sha256'] and r['size']==b['size'] and r['uid']==os.getuid(),'Actual generated payload differs');delete.append(r)
        paths=[r['path'] for r in delete];consumer_check(paths);before=shutil.disk_usage(ROOT).free
        survivors=[record(p) for p in inventory(E) if str(p) not in set(paths)];shards=[];total=0
        for i in range(0,len(survivors),500):
            p=E/f'survivors-{i//500:03d}.json';publish(p,survivors[i:i+500]);total+=p.stat().st_size;shards.append(reference(p))
        require(total<64*1024**2,'Complete durable retirement metadata cap exceeded')
        manifest=dict(kind='exact_successful_non_scientific_generated_fixture_retirement',source=reference(__file__),eligibility=[integration_ref,review_ref,*stopped],delete=delete,survivors=shards,
            authorization='Standing human obsolete numerical-payload authorization after actual acceptance and downstream closure. Four successful4update fixture RAW/H200s only;all scientific selected/fresh and original sources/logs/configs/guards/metrics/receipts remain protected.',
            root_name_warning='Original001 STOP is scorer-interface failure after control training;control was independently accepted through new002 adapter without repeated updates/copies. Eligibility follows actual acceptance,not failed-root naming.',
            future_consumers='Only immutable original receipts, integration record and this later manifest. No future constructor/pilot/scorer/quality stage reloads these four tensors. Never restore/rerun them.',free_before=before)
        publish(E/'manifest.json',manifest)
        require(set(map(str,inventory(E)))=={r['path'] for r in survivors}|set(paths),'Survivor inventory changed before deletion')
        for r in survivors:
            s=Path(r['path']).stat();require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(r['device'],r['inode'],r['size'],r['mtime_ns']),'Survivor changed before deletion')
        consumer_check(paths)
        with (E/'journal.jsonl').open('x') as j:
            def event(d):j.write(json.dumps(d)+'\n');j.flush();os.fsync(j.fileno())
            for r in delete:
                p=Path(r['path']);consumer_check([str(p)]);require(record(p)==r,'Exact payload changed before deletion')
                event(dict(event='intent_permanent_retirement',file=r));p.unlink();syncdir(p.parent);event(dict(event='removed',path=str(p)))
        require(set(map(str,inventory(E)))=={r['path'] for r in survivors},'Survivor inventory changed after deletion')
        for r in survivors:require(record(Path(r['path']))==r,'Retained survivor bytes changed')
        require(integration['sources']==sources() and integration['campaign_sources']==campaign_sources(),'Frozen source changed')
        for ref in [integration_ref,review_ref,*stopped]:require(reference(ref['path'])==ref,'Original closure/evidence changed')
        free=shutil.disk_usage(ROOT).free
        result=dict(status='completed',kind=manifest['kind'],manifest=reference(E/'manifest.json'),deleted_files=4,logical_bytes=sum(r['size'] for r in delete),allocated_bytes=sum(r['allocated'] for r in delete),survivors_verified=len(survivors),all_selected_fresh_original_evidence_unchanged=True,free_before=before,free_after=free,
            observed_free_delta=free-before,no_restore_or_closed_fixture_rerun=True)
        publish(E/'completed.json',result);print(json.dumps(result),flush=True)
