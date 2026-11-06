"""Exact dependency-closed pilot/losing-screen retirement; never fresh/selected."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.body_allocation_worker_v1.execution import checked

def publish(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
def record(path):
    s=path.stat();require(not path.is_symlink() and path.is_file() and s.st_nlink==1,'Unique regular retained file required')
    with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    t=path.stat();require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns),'File changed during audit')
    return dict(path=str(path),device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,sha256=h)
def inventory(excluded):
    out=[]
    for name in ('research_runs','research_archives','research/tokenizer-evidence'):
        for p in (ROOT/name).rglob('*'):
            if p.is_relative_to(excluded) or p==Path(str(excluded)+'-guard.json') or '__pycache__' in p.parts:continue
            require(not p.is_symlink(),'Retained research symlink refused')
            if p.is_file():out.append(p)
    return sorted(set(out))
def consumer_check(paths):
    r=subprocess.run(['lsof','-nP',*paths],text=True,capture_output=True)
    require(r.returncode==1 and not r.stdout.strip(),'Checkpoint has a consumer or consumer query failed')
def syncdir(p):
    fd=os.open(p,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def retire(out,bindings,eligibility,*,kind,selected=()):
    require(kind in ('pilots','losing_screens') and not out.exists(),'New exact stage retirement only')
    require(len(bindings)==(6 if kind=='pilots' else 12),'Complete exact eligible stage required')
    out.mkdir();paths=[b['path'] for b in bindings];require(len(set(paths))==len(paths),'Aliased deletion request')
    selected=set(selected);require(not selected.intersection(paths),'Selected/fresh payload retirement forbidden')
    allowed=ROOT/('research_runs/tokenizer-body-allocation-pilot-v1-001' if kind=='pilots' else 'research_runs/tokenizer-body-allocation-quality-v1-001')
    for ref in eligibility:require(reference(ref['path'])==ref,'Dependency evidence changed')
    for b in bindings:
        p=Path(b['path']);relative=p.relative_to(allowed);parts=relative.parts
        require(len(parts)==5 and parts[0]=='evidence' and parts[1]==('pilots' if kind=='pilots' else 'quality')
            and 0<=int(parts[2])<(3 if kind=='pilots' else 9) and parts[3]=='checkpoints' and parts[4] in ('parent.pt','h200.pt'),'Fresh/foreign path rejected')
    delete=[]
    for b in bindings:
        r=record(Path(b['path']));require(r['sha256']==b['sha256'] and r['size']==b['size'] and r['uid']==os.getuid(),'Exact eligible payload identity differs');delete.append(r)
    consumer_check(paths)
    survivors=[record(p) for p in inventory(out) if str(p) not in set(paths)];shards=[]
    for i in range(0,len(survivors),500):
        p=out/f'survivors-{i//500:03d}.json';publish(p,survivors[i:i+500]);shards.append(reference(p))
    manifest=dict(kind=kind,source=reference(__file__),eligibility=eligibility,delete=delete,survivors=shards,protected_selected_paths=sorted(selected),
        excluded_own_current_audit=[str(out),str(out)+'-guard.json'],
        authorization='Human authorizes obsolete dependency-closed pilots and saved-selection losing screens, with exact path/size/hash/inode/consumer audit. Every selected/fresh/original source/log/config/metric/receipt remains protected.',
        limitation='Permanent exact retirement; original receipts resolve absent tensors through this later manifest. Never restore/rerun closed numerical fixtures or cleanup.')
    publish(out/'manifest.json',manifest)
    require(set(map(str,inventory(out)))=={r['path'] for r in survivors}|set(paths),'Inventory changed before removal')
    for r in survivors:
        s=Path(r['path']).stat();require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(r['device'],r['inode'],r['size'],r['mtime_ns']),'Survivor changed')
    consumer_check(paths)
    with (out/'journal.jsonl').open('x') as j:
        for r in delete:
            p=Path(r['path']);consumer_check([str(p)]);require(record(p)==r,'Payload changed before deletion')
            j.write(json.dumps(dict(event='intent_permanent_retirement',file=r))+'\n');j.flush();os.fsync(j.fileno())
            p.unlink();syncdir(p.parent)
            j.write(json.dumps(dict(event='removed',path=str(p)))+'\n');j.flush();os.fsync(j.fileno())
    require(set(map(str,inventory(out)))=={r['path'] for r in survivors},'Inventory changed after retirement')
    for r in survivors:require(record(Path(r['path']))==r,'Retained survivor hash changed')
    for ref in eligibility:require(reference(ref['path'])==ref,'Original eligibility evidence changed')
    result=dict(status='completed',kind=kind,manifest=reference(out/'manifest.json'),deleted_files=len(delete),
        logical_bytes=sum(r['size'] for r in delete),allocated_bytes=sum(r['allocated'] for r in delete),survivors_verified=len(survivors),
        all_selected_fresh_original_evidence_unchanged=True,free_after=shutil.disk_usage(ROOT).free)
    publish(out/'completed.json',result);return result
