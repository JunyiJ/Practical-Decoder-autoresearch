"""Exact retirement of fifteen dependency-closed generated prerequisite payloads."""
import argparse,hashlib,json,os,shutil,subprocess
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_worker_v1.execution import sources,checked
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
from research.body_allocation_campaign_v1.rules import FLOOR,TARGET
W=Path(__file__).resolve().parent;B=W/'generated-retirement001';POLICY=ROOT/'research/TOKENIZER_CHECKPOINT_RETENTION.md'
R1=ROOT/'research_runs/tokenizer-body-allocation-resume-v1-001';R2=ROOT/'research_runs/tokenizer-body-allocation-worker-v1-001'
p=argparse.ArgumentParser();p.add_argument('mode',choices=['audit','apply']);a=p.parse_args()
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def syncdir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def record(path):
    require(not path.is_symlink() and path.is_file(),'Regular nonsymlink evidence required');s=path.stat();h=sha(path);t=path.stat()
    require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns),'Evidence changed during hash')
    return dict(path=str(path),size=s.st_size,allocated=s.st_blocks*512,sha256=h,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,uid=s.st_uid,gid=s.st_gid,hardlinks=s.st_nlink)
def paths():
    result=[]
    for name in ('research_runs','research_archives','research/tokenizer-evidence'):
        for p in (ROOT/name).rglob('*'):
            if '__pycache__' in p.parts or p.is_relative_to(B):continue
            require(not p.is_symlink(),'Symlink in retained research inventory')
            if p.is_file():result.append(p)
    return sorted(set(result))
def consumers(targets):
    idle()
    proc=subprocess.run(['lsof','-nP',*targets],text=True,capture_output=True)
    require(proc.returncode==1 and not proc.stdout.strip(),'Payload consumer exists or lsof failed: '+proc.stderr)
def journal(row):
    with (B/'journal.jsonl').open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
with session_lock(GLOBAL_LOCK.parent):
    idle()
    if a.mode=='audit':require(not B.exists(),'Closed audit cannot repeat');B.mkdir()
    with CpuPrerequisiteGuard(B/f'{a.mode}-guard.json',ROOT,timeout_seconds=900,poll_seconds=.25):
        if a.mode=='audit':
            closure=checked(reference(W/'integration001.json'));require(closure['status']=='generated_model_and_worker_dependencies_closed' and closure['sources']==sources(),'Actual integration closure required')
            eligibility=[reference(W/'integration001.json')];delete=[];expected=set()
            for slot,arm in enumerate(('incumbent','shallow','deeper')):
                report=checked(reference(W/f'resume001-{arm}.json'));require(report['status']=='passed','Successful exact generated resume required');eligibility.append(reference(W/f'resume001-{arm}.json'))
                accepted=checked(reference(R2/f'evidence/pilots/{slot:02d}/accepted.json'));require(accepted['acceptance']['status']=='accepted','Worker parent acceptance required')
                worker=checked(accepted['worker']);eligibility.extend([reference(R2/f'evidence/pilots/{slot:02d}/accepted.json'),accepted['worker']])
                for root,refs,names in [(R1,report,('mid','parent','h200')),(R2,worker,('parent','h200'))]:
                    for name in names:
                        path=root/f'evidence/pilots/{slot:02d}/checkpoints/{name}.pt';expected.add(str(path));ref=refs[name]
                        require(ref['path']==str(path),'Exact generated basename/root differs');row=record(path)
                        require(row['size']==ref['size'] and row['sha256']==ref['sha256'] and row['uid']==os.getuid() and row['hardlinks']==1,'Exact owned generated payload identity differs')
                        delete.append(dict(**row,reason='Obsolete successful four-update correctness payload, downstream model/real-worker/finite-config and cache/full-validation dependencies closed; not selected or fresh science.'))
            require(len(delete)==15 and len(expected)==15,'Exact fifteen-payload allowlist required');consumers(sorted(expected))
            for root in (R1,R2):
                require(not (root/'STOP').exists(),'Generated fixture unexpectedly closed')
                publish(root/'STOP',dict(kind='completed_generated_fixture_payload_retirement',closure=reference(W/'integration001.json'),never_restore_or_rerun=True))
            survivors=[record(p) for p in paths() if str(p) not in expected];shards=[]
            for i in range(0,len(survivors),500):
                path=B/f'survivors-{i//500:03d}.json';publish(path,survivors[i:i+500]);shards.append(reference(path))
            manifest=dict(status='audited',source=reference(__file__),policy=reference(POLICY),closure=reference(W/'integration001.json'),eligibility=eligibility,
                delete=delete,survivor_shards=shards,deleted_logical_bytes=sum(r['size'] for r in delete),deleted_allocated_bytes=sum(r['allocated'] for r in delete),
                free_before=shutil.disk_usage(ROOT).free,required_free=FLOOR,cleanup_target=TARGET,
                authorization='Human authorizes retirement of proved obsolete successful generated payloads after actual dependency closure; all selected/fresh/original sources/logs/configs/metrics/receipts preserved.',
                limitation='Permanent exact retirement only. Immutable old receipts resolve absent payloads through this later manifest; never restore or rerun these closed fixtures.')
            publish(B/'manifest.json',manifest);print(json.dumps(dict(status='audited',files=15,logical=manifest['deleted_logical_bytes'],allocated=manifest['deleted_allocated_bytes'],survivors=len(survivors))),flush=True)
        else:
            require(json.loads((B/'audit-guard.json').read_text())['status']=='finished','Completed audit required');m=checked(reference(B/'manifest.json'))
            for ref in (m['source'],m['policy'],m['closure'],*m['eligibility']):require(reference(ref['path'])==ref,'Eligibility evidence changed')
            survivors=[r for s in m['survivor_shards'] for r in checked(s)];remove={r['path'] for r in m['delete']}
            require(set(map(str,paths()))=={r['path'] for r in survivors}|remove,'Retained inventory changed')
            for row in survivors:
                s=Path(row['path']).stat();require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(row['device'],row['inode'],row['size'],row['mtime_ns']),'Retained survivor changed')
            consumers(sorted(remove))
            for row in m['delete']:require(record(Path(row['path']))=={k:v for k,v in row.items() if k!='reason'},'Delete allowlist changed')
            for row in m['delete']:
                path=Path(row['path']);consumers([str(path)]);require(record(path)=={k:v for k,v in row.items() if k!='reason'},'Payload changed before retirement')
                journal(dict(event='intent_permanent_retirement',file=row));path.unlink();syncdir(path.parent);journal(dict(event='removed',path=str(path)))
            require(set(map(str,paths()))=={r['path'] for r in survivors},'Post-retirement inventory changed')
            for row in survivors:require(record(Path(row['path']))==row,'Retained evidence hash changed')
            for ref in (m['source'],m['policy'],m['closure'],*m['eligibility']):require(reference(ref['path'])==ref,'Original evidence changed')
            free=shutil.disk_usage(ROOT).free;result=dict(status='completed',manifest=reference(B/'manifest.json'),files_removed=15,
                deleted_logical_bytes=m['deleted_logical_bytes'],deleted_allocated_bytes=m['deleted_allocated_bytes'],survivors_verified=len(survivors),
                all_selected_fresh_original_evidence_unchanged=True,free_after=free,mandatory_deficit=max(0,FLOOR-free),target_deficit=max(0,TARGET-free),required_free=FLOOR,cleanup_target=TARGET)
            publish(B/'completed.json',result);print(json.dumps(result),flush=True)
