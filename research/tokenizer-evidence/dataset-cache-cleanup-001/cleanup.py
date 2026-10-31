"""Explicit user-authorized disposable dataset caches only; no model/research deletion."""
import argparse,fcntl,hashlib,json,os,shutil,stat,subprocess
from pathlib import Path
from datetime import datetime,timezone
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,idle,session_lock
from src.research.protocol import ROOT
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
BASE=ROOT/'research/tokenizer-evidence/dataset-cache-cleanup-001'
HF=Path('/Users/junyi/.cache/huggingface')
TARGETS=[HF/'datasets']+[HF/'hub'/n for n in ['datasets--BAAI--TACO','datasets--argilla--distilabel-math-preference-dpo','datasets--codeparrot--apps','datasets--gsm8k','datasets--intel--orca_dpo_pairs','datasets--openai--gsm8k','datasets--xinlai--Math-Step-DPO-10K']]

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for chunk in iter(lambda:f.read(4*1024**2),b''):h.update(chunk)
 return h.hexdigest()

def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())

def entry(p):
 s=p.lstat();v=dict(bytes=s.st_size,allocated_bytes=s.st_blocks*512,device=s.st_dev,inode=s.st_ino,mtime_ns=s.st_mtime_ns,mode=stat.S_IMODE(s.st_mode))
 if p.is_symlink():
  target=p.resolve(strict=True);assert any(target.is_relative_to(r) and p.is_relative_to(r) for r in TARGETS)
  v.update(kind='symlink',target=os.readlink(p))
 else:
  assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
  v.update(kind='file',sha256=sha(p))
 return v

def inventory():
 files={};dirs=[]
 for root in TARGETS:
  for p in [root,*root.parents]:assert not p.is_symlink()
  assert root.is_dir();dirs.append(str(root))
  for base,ds,fs in os.walk(root,followlinks=False):
   for name in ds:
    p=Path(base)/name;assert not p.is_symlink();dirs.append(str(p))
   for name in fs:
    p=Path(base)/name;files[str(p)]=entry(p)
 return dict(files=files,directories=sorted(dirs))

def protected():
 # Byte fingerprints for all research weights and every immutable portable-evidence
 # entry; model cache gets an inode/size/mtime inventory without reading weights.
 state=json.loads((ROOT/'research/TOKENIZER_RESUME_STATE.json').read_text());result={}
 for name,h in state['evidence_sha256'].items():
  assert sha(ROOT/name)==h;result[name]=h
 for folder in ['research_runs','research_state','research_archives']:
  for p in (ROOT/folder).rglob('*'):
   if p.is_file() and not p.is_symlink() and (p.suffix in ['.pt','.pth','.zst','.safetensors'] or p.name.endswith('.pt.tmp')):result[str(p.relative_to(ROOT))]=sha(p)
 models={}
 for root in (HF/'hub').glob('models--*'):
  for p in root.rglob('*'):
   if p.is_file() or p.is_symlink():
    s=p.lstat();models[str(p)]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,os.readlink(p) if p.is_symlink() else None]
 return dict(research_sha256=result,model_cache_stat=models)

parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['plan','apply']);args=parser.parse_args()
with session_lock(GLOBAL_LOCK.parent),CpuPrerequisiteGuard(BASE/(args.mode+'-guard.json'),ROOT,timeout_seconds=600):
 idle()
 scan=subprocess.run(['lsof','-nP','+D',str(HF)],capture_output=True,text=True)
 assert scan.returncode==1 and not scan.stdout and not scan.stderr,'Hugging Face cache open or inspection failed'
 if args.mode=='plan':
  before=protected();inv=inventory()
  write(BASE/'plan.json',dict(status='prospective_exact_dataset_cache_deletion',observed_at=datetime.now(timezone.utc).isoformat(),script_sha256=sha(__file__),
   authorization='Human explicitly requested deletion of unused dataset caches on October31; current experiment reads local Shakespeare/generated fixtures, no Hugging Face datasets.',
   roots=[str(p) for p in TARGETS],inventory=inv,protected=before,free_before=shutil.disk_usage(ROOT).free,
   expected_allocated_bytes=sum(v['allocated_bytes'] for v in inv['files'].values()),
   excludes='All model caches, codex runtime caches, corpus, research checkpoints and archives. No unrelated program stopped.',
   reusability='These downloaded/processed dataset caches can be downloaded/rebuilt; source datasets are not deleted.',reserve_bytes=8*1024**3))
 else:
  plan=json.loads((BASE/'plan.json').read_text());assert sha(__file__)==plan['script_sha256']
  assert [str(p) for p in TARGETS]==plan['roots'] and inventory()==plan['inventory'] and protected()==plan['protected']
  lockpaths=[p for p in (HF/'datasets').rglob('*.lock') if p.is_file()]
  lockpaths += [p for p in (HF/'hub/.locks').glob('datasets--*/*.lock') if p.is_file()]
  locks=[]
  try:
   for path in sorted(lockpaths):
    assert not path.is_symlink();fd=os.open(path,os.O_RDWR|os.O_NOFOLLOW);locks.append(fd);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   with (BASE/'journal.jsonl').open('x') as journal:
    for name,saved in plan['inventory']['files'].items():
     p=Path(name);assert entry(p)==saved;p.unlink()
     journal.write(json.dumps(dict(path=name,deleted=True,kind=saved['kind'],bytes=saved['bytes']))+'\n');journal.flush();os.fsync(journal.fileno())
    for name in sorted(plan['inventory']['directories'],key=lambda n:len(Path(n).parts),reverse=True):Path(name).rmdir()
  finally:
   for fd in locks:os.close(fd)
  assert all(not p.exists() for p in TARGETS);assert protected()==plan['protected'];idle()
  free=shutil.disk_usage(ROOT).free
  write(BASE/'completed.json',dict(status='completed_verified',observed_at=datetime.now(timezone.utc).isoformat(),plan_sha256=sha(BASE/'plan.json'),
   files_removed=len(plan['inventory']['files']),directories_removed=len(plan['inventory']['directories']),allocated_bytes_removed=plan['expected_allocated_bytes'],
   free_before=plan['free_before'],free_after=free,observed_free_change_bytes=free-plan['free_before'],research_and_model_cache_unchanged=True,
   research_checkpoints_deleted=0,quality_training_launched=False,holdout_scored=False))
 print(json.dumps(dict(mode=args.mode,status='complete')))
