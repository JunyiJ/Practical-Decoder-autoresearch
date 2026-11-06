"""Human-authorized retirement of superseded historical model weights only.

New permission: 'clean up disk space as much as possible - ok to delete old
checkpoint directly'. This narrowly supersedes historical selected/fresh weight
retention for the exact 202 previously archived pre-body-allocation endpoints.
All original records, recent body/final-scale payloads and current incumbent stay.
"""
import hashlib,json,os,shutil,stat,subprocess
from pathlib import Path
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
R=Path.cwd();E=Path(__file__).resolve().parent
AUTH='clean up disk space as much as possible - ok to delete old checkpoint directly'
ALLOWED_TERMINAL={'completed_closed','completed','completed_not_adopted','recovery_closed_independently_reviewed','complete_independent_quality_cost_review','finite_campaign_closed','failed_startup_preserved','closed_not_adopted','completed_adopted','passed','completed_closed_adopted'}
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p.resolve()),sha256=sha(p))
def record(p):
 s=p.lstat();assert p.resolve()==p and stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_uid==os.getuid(),str(p)
 h=sha(p);t=p.stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(t.st_dev,t.st_ino,t.st_size,t.st_mtime_ns)
 return dict(path=str(p),device=s.st_dev,inode=s.st_ino,uid=s.st_uid,gid=s.st_gid,nlink=s.st_nlink,size=s.st_size,allocated=s.st_blocks*512,mtime_ns=s.st_mtime_ns,sha256=h)
def bound(v):
 p=Path(v['path']);assert p.is_file() and not p.is_symlink() and sha(p)==v['sha256'],str(p)
 return p

def publish(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
def syncdir(p):
 fd=os.open(p,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def consumers(paths):
 idle();cp=subprocess.run(['lsof','-nP',*paths],text=True,capture_output=True,timeout=30)
 assert cp.returncode==1 and not cp.stdout.strip() and not cp.stderr.strip(),'Consumer present or query failed'
def inventory():
 out=[]
 for base in ('research_runs','research_archives','research/tokenizer-evidence','src','data'):
  for p in (R/base).rglob('*'):
   if p.is_relative_to(E) or '__pycache__' in p.parts or p.name in ('.lock','.metadata.lock'):continue
   assert not p.is_symlink(),'Symlink in retained scope'
   if p.is_file():out.append(p)
 return sorted(set(out))
with session_lock(GLOBAL_LOCK.parent):
 idle();assert not (E/'manifest.json').exists(),'One exact new retirement only'
 with CpuPrerequisiteGuard(E/'guard.json',R,timeout_seconds=1200,poll_seconds=.25):
  failed=R/'research/tokenizer-evidence/human-old-checkpoint-retirement-001'
  assert json.loads((failed/'guard.json').read_text())['status']=='fixture_exception' and not (failed/'manifest.json').exists() and not (failed/'journal.jsonl').exists(),'Original audit must remain closed before deletion'
  before=shutil.disk_usage(R).free;rows=[];eligibility={};delete=[]
  for p in sorted((R/'research/tokenizer-evidence').glob('storage-recovery-*/raw-completed.json')):
   d=json.loads(p.read_text());assert d['status']=='completed_lossless_raw_storage' and d['all_raw_bytes_preserved'] and d['all_h200_unchanged']
   bound(d['plan']);eligibility[str(p)]=ref(p)
   for v in d['files']:
    ar=Path(v['archive']['path'])
    if not ar.exists():continue
    assert ar.parent.parent==R/'research_archives' and ar.parent.name.startswith('protected-raw-storage-')
    original=Path(v['original']['path']);assert not original.exists(),'Unexpected restored original'
    root=R/'research_runs'/original.relative_to(R/'research_runs').parts[0]
    assert not any(x in root.name for x in ('body-allocation','final-ln-scale','embedding-input-dropout'))
    terminal=bound(v['terminal']);acceptance=bound(v['acceptance']);td=json.loads(terminal.read_text())
    assert td.get('status') in ALLOWED_TERMINAL or td.get('reason')=='finite_panel_closed_not_adopted',str(terminal)
    if td.get('reason')=='finite_panel_closed_not_adopted':bound(td['review'])
    h=bound(v['h200']);assert h.parent==original.parent and h.name=='h200.pt'
    if root.name not in str(acceptance) and '/research/tokenizer-evidence/' not in str(acceptance):
     # Exactly one documented imported endpoint has a later-root acceptance.
     assert root.name=='tokenizer-gradient-clip-quality-v1-001' and str(acceptance.relative_to(R))=='research_runs/tokenizer-gradient-clip-quality-v2-001/evidence/quality/00/resource-events.jsonl'
     records=[json.loads(line) for line in acceptance.read_text().splitlines() if line.strip()]
     matching=[line['artifact_acceptance'] for line in records if isinstance(line.get('artifact_acceptance'),dict) and line['artifact_acceptance'].get('accepted') is True]
     assert any(a['files']['parent']['path']==str(original) and a['files']['parent']['sha256']==v['original']['sha256'] and a['files']['h200']==v['h200'] for a in matching),'Imported original identities differ'

    for name in ('archive','h200'):
     b=v[name];rp=bound(b);rec=record(rp);assert rec['sha256']==b['sha256']
     if 'size' in b:assert rec['size']==b['size']
     rec['reason']='Superseded historical endpoint: original accepted/final or closed-failure evidence retained; latest incumbent now retained in body-allocation panel; current input-dropout workers initialize fresh and do not consume historical weight files.'
     rec['kind']=name;rec['original_raw_path']=str(original);delete.append(rec)
    rows.append(dict(archive_storage_receipt=ref(p),original=v['original'],archive=v['archive'],h200=v['h200'],acceptance=v['acceptance'],terminal=v['terminal'],terminal_status=td.get('status',td.get('reason'))))
  assert len(rows)==202 and len(delete)==404 and len({r['path'] for r in delete})==404
  keys={(r['device'],r['inode']) for r in delete};assert len(keys)==404
  # Current and most recent comparative models remain in original paths.
  body=json.loads((R/'research/tokenizer-evidence/body-allocation-preparation-001/quality001-closure001.json').read_text())
  scale=json.loads((R/'research/tokenizer-evidence/final-ln-scale-preparation-001/quality002-closure.json').read_text())
  protected=[*body['protected_original_payloads'],*scale['weights']];assert len(protected)==36
  for v in protected:bound(v)
  for v in body['trajectories']:
   if v['arm']=='incumbent' and v['slot']>=9:bound(v['accepted'])
  closure=json.loads((R/'research/tokenizer-evidence/embedding-input-dropout-preparation-001/quality-contracts001.json').read_text())
  pins={**closure['worker_sources'],**closure['pilot_sources'],**closure['quality_sources']};assert len(pins)==972
  for n,h in pins.items():assert sha(R/n)==h
  assert "cfg.training.resume_from is None" in (R/'research/embedding_input_dropout_worker_v1/training.py').read_text()
  assert not (R/'research_runs/tokenizer-embedding-input-dropout-quality-v1-001').exists()
  paths=[r['path'] for r in delete];consumers(paths)
  doomed=set(paths);survivors=[record(p) for p in inventory() if str(p) not in doomed];shards=[]
  for i in range(0,len(survivors),500):
   p=E/f'survivors-{i//500:03d}.json';publish(p,survivors[i:i+500]);shards.append(ref(p))
  policy=dict(human_authorization=AUTH,scope='Only exact 202 historical archived RAW endpoints and their matching 202 H200s, before the latest body/final-scale panels.',
   explicit_exception='Earlier retain-all historical selected/fresh archive rules are superseded by the latest direct human old-checkpoint deletion request for these superseded models. Recent36body/final-scale payloads, all current incumbent models, upcoming work and original records remain protected.',
   closed_dependencies='All original acceptance and terminal hashes revalidated; actual archive storage receipts preserved. Upcoming input-dropout training requires fresh initialization and reads only source/data/prerequisite metadata, not historical models.',
   irreversibility='Permanent retirement, no backup copy or restore. Historical receipts resolve missing tensors through this later exact manifest. Never rerun closed studies.')
  manifest=dict(source=ref(Path(__file__)),prior_closed_no_deletion_audit=[ref(failed/n) for n in ['retire.py','retire.log','guard.json']],cross_root_exception='Only gradient-clip v1 slot00 original identities accepted and bound in v2 slot00; actual original RAW and H200 hashes verified in accepted record.',policy=policy,eligibility=list(eligibility.values()),endpoints=rows,delete=delete,survivors=shards,protected_recent_payloads=protected,frozen_source_bindings=pins,
   free_before=before,allocated_to_reclaim=sum(r['allocated'] for r in delete),logical_to_remove=sum(r['size'] for r in delete))
  publish(E/'manifest.json',manifest)
  assert set(map(str,inventory()))=={r['path'] for r in survivors}|doomed
  for r in survivors:
   s=Path(r['path']).stat();assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(r['device'],r['inode'],r['size'],r['mtime_ns'])
  with (E/'journal.jsonl').open('x') as journal:
   for i,r in enumerate(delete):
    if i%24==0:consumers([v['path'] for v in delete[i:i+24]])
    p=Path(r['path']);current=record(p);assert all(current[k]==r[k] for k in current),'Payload changed'
    journal.write(json.dumps(dict(event='intent_permanent_human_authorized_old_checkpoint_retirement',file=r))+'\n');journal.flush();os.fsync(journal.fileno())
    p.unlink();syncdir(p.parent)
    journal.write(json.dumps(dict(event='removed',path=r['path']))+'\n');journal.flush();os.fsync(journal.fileno())
  assert set(map(str,inventory()))=={r['path'] for r in survivors}
  for r in survivors:assert record(Path(r['path']))==r,'Retained evidence changed'
  for v in eligibility.values():bound(v)
  for v in protected:bound(v)
  for n,h in pins.items():assert sha(R/n)==h
  result=dict(status='completed_human_authorized_superseded_historical_checkpoint_retirement',manifest=ref(E/'manifest.json'),deleted_files=len(delete),historical_endpoints=len(rows),
   logical_removed=sum(r['size'] for r in delete),allocated_reclaimed=sum(r['allocated'] for r in delete),survivors_verified=len(survivors),protected_recent_weights_verified=36,
   frozen_source_bindings_verified=972,all_original_records_and_current_models_unchanged=True,free_before=before,free_after=shutil.disk_usage(R).free,
   no_apps_or_workers_stopped=True,no_active_or_upcoming_payload_removed=True,no_restoration_or_closed_fixture_replay=True)
  publish(E/'completed.json',result);print(json.dumps(result),flush=True)
