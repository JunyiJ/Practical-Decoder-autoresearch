"""One-shot removal of unused component downloads verified against installed bytes."""
import fcntl,hashlib,json,os,re,shutil,stat,subprocess,sys,time,zipfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
BASE=Path('/Users/junyi/Library/Application Support')
APPS=[BASE/'Google/Chrome',BASE/'Codex']

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def inspect(p):
 for q in (p,*p.parents):assert not q.is_symlink(),str(q)
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid() and s.st_nlink==1,str(p)
 h=sha(p);t=p.lstat()
 keys=('st_dev','st_ino','st_mode','st_uid','st_nlink','st_size','st_mtime_ns','st_ctime_ns','st_blocks')
 assert all(getattr(s,k)==getattr(t,k) for k in keys)
 return dict(path=str(p),sha256=h,allocated_bytes=s.st_blocks*512,**{k:getattr(s,k) for k in keys})

def save(n,v):
 with (HERE/n).open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def opened():
 r=subprocess.run(['/usr/sbin/lsof','-nP','-Fn'],capture_output=True,text=True,timeout=30)
 assert r.returncode==0 and not r.stderr.strip(),r.stderr
 return {x[1:] for x in r.stdout.splitlines() if x.startswith('n')}

def installed(root):
 out=[]
 for p in root.glob('*/*/manifest.json'):
  if p.parts[-3] in ('Default','Profile 1','System Profile','Guest Profile'):continue
  try:m=json.loads(p.read_text())
  except (ValueError,OSError):continue
  out.append((m.get('name'),m.get('version'),p.parent))
 return out

def verify_copy(p,dest):
 refs=[]
 with zipfile.ZipFile(p) as z:
  for item in z.infolist():
   rel=Path(item.filename);assert not rel.is_absolute() and '..' not in rel.parts
   if item.is_dir():continue
   target=dest/rel;row=inspect(target)
   assert row['sha256']==hashlib.sha256(z.read(item)).hexdigest(),str(target)
   refs.append(dict(path=str(target),size=row['st_size'],sha256=row['sha256']))
 return refs

if sys.argv[1:]==['plan']:
 opened_paths=opened();rows=[];excluded=[]
 for root in APPS:
  candidates=installed(root)
  for p in sorted((root/'component_crx_cache').iterdir()):
   if not re.fullmatch('[a-f0-9]{64}',p.name):continue
   row=inspect(p);assert row['sha256']==p.name
   with zipfile.ZipFile(p) as z:m=json.loads(z.read('manifest.json'))
   matches=[d for name,version,d in candidates if (name,version)==(m.get('name'),m.get('version'))]
   if len(matches)!=1 or str(p) in opened_paths or time.time()-p.stat().st_mtime<3600:
    excluded.append(dict(path=str(p),reason='No unique installed-copy proof, open download, or recent updater activity'));continue
   refs=verify_copy(p,matches[0])
   rows.append(dict(file=row,installed_directory=str(matches[0]),installed_files=refs,name=m['name'],version=m['version'],reason='Completed component download; every archived file SHA256-identical to installed copy; no open handle; installed software retained'))
 rows.sort(key=lambda x:x['file']['allocated_bytes'],reverse=True)
 save('manifest.json',dict(source_sha256=sha(Path(__file__)),authorization='Standing human authorization to remove verified unused package/archive downloads and specific unused application caches; explicit immediate cleanup pass',files=rows,excluded=excluded,allocated_bytes=sum(r['file']['allocated_bytes'] for r in rows)))
 print(json.dumps(dict(files=len(rows),allocated_bytes=sum(r['file']['allocated_bytes'] for r in rows),excluded=len(excluded))))
elif sys.argv[1:]==['apply']:
 m=json.loads((HERE/'manifest.json').read_text());assert m['source_sha256']==sha(Path(__file__))
 lock=ROOT/'research_state/external-cache-maintenance.lock';assert not lock.is_symlink()
 with lock.open('a+') as held:
  fcntl.flock(held,fcntl.LOCK_EX|fcntl.LOCK_NB)
  before=shutil.disk_usage(ROOT).free;removed=[];errors=[]
  with (HERE/'deletion-journal.jsonl').open('x') as j:
   def event(v):j.write(json.dumps(dict(time=time.time(),**v))+'\n');j.flush();os.fsync(j.fileno())
   for r in m['files']:
    p=Path(r['file']['path']);assert any(p.parent==root/'component_crx_cache' for root in APPS)
    assert str(p) not in opened();assert inspect(p)==r['file']
    assert verify_copy(p,Path(r['installed_directory']))==r['installed_files']
    assert inspect(p)==r['file'];event(dict(event='intent',file=r['file'],reason=r['reason']))
    try:p.unlink()
    except OSError as e:errors.append(dict(path=str(p),error=repr(e)));event(dict(event='failed',**errors[-1]));continue
    assert not p.exists();removed.append(r);event(dict(event='removed',path=str(p),allocated_bytes=r['file']['allocated_bytes']))
  retained=0
  for r in removed:
   for ref in r['installed_files']:
    p=Path(ref['path']);assert p.stat().st_size==ref['size'] and sha(p)==ref['sha256'];retained+=1
  after=shutil.disk_usage(ROOT).free
  save('receipt.json',dict(status='completed' if not errors else 'partial',files_removed=len(removed),allocated_bytes_removed=sum(r['file']['allocated_bytes'] for r in removed),logical_bytes_removed=sum(r['file']['st_size'] for r in removed),retained_installed_files_verified=retained,free_before=before,free_after=after,observed_free_delta=after-before,errors=errors,research_payloads_modified=False,installed_software_modified=False,profiles_sessions_modified=False,limitation='Free delta may include concurrent system writes and APFS effects; cached downloads can be fetched again.',completed_unix=time.time()))
  print((HERE/'receipt.json').read_text())
else:raise SystemExit('plan or apply')
