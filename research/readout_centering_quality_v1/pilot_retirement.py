"""One exact completed-pilot retirement after additive quality prerequisites."""
import json,shutil,time
from pathlib import Path
from . import campaign_contract as c,artifact_acceptance as a
from .support import idle,exclusive,cache_audit
from research.readout_centering_campaign_v2 import retention as tested,pilot as pilot
from research.readout_centering_worker_v1 import inference as inf
from research.block_norm_v1.fixture_dispatch import reference,read
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.protocol import file_digest,digest
B=pilot.B;DEST=B/'quality001-pilot-retirement'

def record(p):
    c.require(not p.is_symlink() and p.is_file(),'Regular retained file required');s=p.stat();h=file_digest(p);t=p.stat()
    c.require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns),'Evidence changed during inventory')
    return dict(path=str(p),size=t.st_size,sha256=h)
def all_paths():
    return sorted(p for name in ('research_runs','research_archives') for p in (c.ROOT/name).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='.lock')
def verify_retirement(ref):
    done=read(ref);c.require(done['status']=='completed' and done['deleted_bytes']==1118335248,'Exact pilot retirement required')
    manifest=read(done['manifest']);context=done['context'];root=pilot.ROOT/'evidence'
    tested.recheck_execution(root,manifest,done['execution_ref'],digest(context))
    return done

def apply():
    from .quality_execution import QualityGuard
    with session_lock(GLOBAL_LOCK.parent) as g,session_lock(pilot.ROOT) as p:
        idle();c.require(not DEST.exists(),'Pilot retirement attempt cannot retry');DEST.mkdir()
        with QualityGuard(DEST/'audit-guard.json',c.ROOT,600):
            checked=read(reference(B/'quality-integration001-review.json'));c.require(checked['status']=='passed','Additive real quality prerequisites required')
            sources=c.source_inventory();c.require(sources==read(checked['plan'])['sources'],'Reviewed source differs')
            report=read(reference(B/'pilot001-review.json'));c.require(report['status']=='both_resource_pilots_and_inference_passed','Completed inference required')
            rows=[read(reference(pilot.ROOT/f'evidence/pilots/{i:02d}/accepted.json')) for i in range(2)];pairs={};records=[];bindings={}
            for i,row in enumerate(rows):
                req=dict(row['acceptance_request'],sources=sources,scope=a.SCOPE);a.accept(req,row['validation']);cache_audit(req)
                pairs[i]={}
                for role in ('parent','h200'):
                    q=Path(req['files'][role]['path']);c.require(file_digest(q)==req['files'][role]['sha256'],'Pilot weight changed')
                    pairs[i]['raw' if role=='parent' else role]=dict(path=str(q.relative_to(pilot.ROOT/'evidence')),size=q.stat().st_size,sha256=file_digest(q))
                records.append(dict(slot=i,arm=row['arm'],lr=.00125,status='completed',h200_ce=row['validation']['ce']))
                bindings[row['arm']]={k:req['files'][k] for k in ('config','parent','h200')}
            inf.review(read(report['inference']),bindings,0)
            writer=MetadataWriter(pilot.ROOT,(g,p));writer.publish_json('evidence/meta/pilot-inference.json',dict(review=reference(B/'pilot001-review.json'),integration=reference(B/'quality-integration001-review.json')))
            e=pilot.ROOT/'evidence';inventory={str(q.relative_to(e)):dict(size=q.stat().st_size,sha256=file_digest(q)) for q in e.rglob('*') if q.is_file()}
            evidence={n:v['sha256'] for n,v in inventory.items() if not n.endswith('.pt')}
            args=dict(boundary='pilot',records=records,selection=None,pairs=pairs,evidence=evidence);manifest=tested.planned_manifest(**args)
            exclusive(DEST/'manifest.json',manifest)
            candidates={str(e/x['path']) for x in manifest['delete']};c.require(len(candidates)==4 and sum(x['size'] for x in manifest['delete'])==1118335248,'Exact eligible pilot pair sizes required')
            survivors=[record(q) for q in all_paths() if str(q) not in candidates];shards=[]
            for start in range(0,len(survivors),500):
                path=DEST/f'protected-{start//500:03d}.json';exclusive(path,survivors[start:start+500]);shards.append(reference(path))
            context=dict(integration=reference(B/'quality-integration001-review.json'),pilot_plan=reference(B/'pilot001-plan.json'),pilot_inference=reference(B/'pilot001-review.json'),sources_sha256=c.fingerprint(sources))
            def revalidate():
                idle();c.verify_sources(sources);c.require(not (pilot.ROOT/'STOP').exists(),'Pilot root stopped')
                for k in ('integration','pilot_plan','pilot_inference'):read(context[k])
                return context
            review=tested.execute_boundary(e,'pilot',(g,p),lambda:dict(inputs=args,inventory=inventory),revalidate)
            for row in survivors:c.require(record(Path(row['path']))==row,'Protected research evidence changed')
            added={str(q) for q in (e/'.retention/pilot').rglob('*') if q.is_file()}
            c.require({str(q) for q in all_paths()}=={r['path'] for r in survivors}|added,'Research inventory changed')
            done=dict(status='completed',manifest=reference(DEST/'manifest.json'),execution_ref=dict(path='.retention/pilot/execution-review.json',sha256=file_digest(e/'.retention/pilot/execution-review.json')),context=context,deleted_bytes=review['deleted_logical_bytes'],protected_files_verified=len(survivors),protected_inventory_shards=shards,free_bytes=shutil.disk_usage(c.ROOT).free,failed_selected_fresh_unchanged=True,limitation='These four successful pilot weights cannot be restored; original audits and metadata remain immutable.')
            exclusive(DEST/'completed.json',done);verify_retirement(reference(DEST/'completed.json'))
        print(json.dumps({k:done[k] for k in ['status','deleted_bytes','protected_files_verified','free_bytes']}),flush=True)
if __name__=='__main__':apply()
