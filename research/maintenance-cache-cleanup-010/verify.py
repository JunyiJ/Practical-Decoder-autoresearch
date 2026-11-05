import os,json,stat,hashlib,fcntl,datetime
from pathlib import Path
root=Path('/Users/junyi/Documents/ChatGPT/LLM-pretraining-RSI');out=root/'research/maintenance-cache-cleanup-010'
lock=open(root/'research_state/external-cache-maintenance.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
m=json.load(open(out/'manifest.json'));assert not Path(m['target']).exists()
def entry(p):
 s=p.lstat();r=dict(path=str(p),dev=s.st_dev,ino=s.st_ino,uid=s.st_uid,gid=s.st_gid,mode=s.st_mode,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
 if stat.S_ISLNK(s.st_mode):r['link']=os.readlink(p)
 elif stat.S_ISREG(s.st_mode):
  h=hashlib.sha256()
  with open(p,'rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  r['sha256']=h.hexdigest()
 return r
count=0
for rows in m['retained'].values():
 for r in rows:assert entry(Path(r['path']))==r;count+=1
assert os.readlink(Path(m['target']).parent/'Current')=='156.0.8078.80'
journal=[json.loads(l) for l in open(out/'deletion-journal.jsonl')]
assert set(r['path'] for r in journal if r['event']=='removed')==set(r['path'] for r in m['entries'])
s=os.statvfs(root);free=s.f_bavail*s.f_frsize
result=dict(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='deletion_complete_independent_verification_pass',allocated_bytes_removed=sum(r['allocated'] for r in m['entries']),free_before=journal[0]['free'],free_after_verification=free,free_delta=free-journal[0]['free'],retained_entries_verified=count,target_absent=True,cleanup_target_bytes=14529069056,remaining_deficit=14529069056-free,original_final_record_error='TypeError duplicate at keyword after deletion and retained verification; deletion was not rerun',note='APFS shared extents and concurrent writes can make free-space delta differ from summed allocation.')
with open(out/'completed.json','x') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
print(json.dumps(result))
