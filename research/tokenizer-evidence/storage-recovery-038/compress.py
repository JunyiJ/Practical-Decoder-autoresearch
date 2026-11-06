"""New prefix6 blocker: lossless compression of newly closed body-allocation metadata."""
import fcntl,importlib.util,json,os,shutil,stat,subprocess
from pathlib import Path
R=Path.cwd();E=Path(__file__).resolve().parent;B=R/'research/tokenizer-evidence/body-allocation-preparation-001'
s=importlib.util.spec_from_file_location('compression_primitives',R/'research/tokenizer-evidence/filesystem-compression-001/compress.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.body_allocation_worker_v1.execution import sources,checked
from research.body_allocation_quality_v1.contract import quality_sources
from research.body_allocation_campaign_v1.pilot import campaign_sources
from research.final_ln_scale_quality_v3.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
RUN=R/'research_runs/tokenizer-body-allocation-quality-v1-001'
REQUIRED=13327286044;TARGET=15474769692
def info(p):
    st=p.lstat();assert stat.S_ISREG(st.st_mode) and st.st_nlink==1 and st.st_uid==os.getuid() and not p.is_symlink()
    return dict(a.info(p),device=st.st_dev,inode=st.st_ino,nlink=st.st_nlink,reason='Newly closed accepted body-allocation prerequisite/pilot/retirement or completed screen metadata, lossless transparent compression only')
def check(p,v,physical=False):
    a.check(p,v,physical=False)
    if physical:assert info(p)==v
with session_lock(GLOBAL_LOCK.parent):
    idle();assert not (RUN/'STOP').exists() and not (RUN/'evidence/quality/06').exists()
    with CpuPrerequisiteGuard(E/'guard.json',R,timeout_seconds=300,poll_seconds=.25):
        plan=checked(dict(path=str(B/'quality001-plan.json'),sha256=a.info(B/'quality001-plan.json')['sha256']))
        assert plan['sources']==sources() and plan['quality_sources']==quality_sources() and plan['pilot_sources']==campaign_sources()
        closed=set()
        prefixes=('modelchecks001','resume001','worker001','integration001','census001','pilot001','generated-retirement001','quality-contracts001','quality001-pilot-retirement001','quality001-accepted-','quality001-slot00')
        for p in B.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.relative_to(B).parts[0].startswith(prefixes):closed.add(p)
        for slot in range(6):
            base=RUN/f'evidence/quality/{slot:02d}';row=json.loads((base/'accepted.json').read_text());assert row['acceptance']['status']=='accepted'
            closed.update(p for p in base.rglob('*') if p.is_file() and 'checkpoints' not in p.parts and not p.name.startswith('.'))
            for kind in ('worker','parent'):
                p=RUN/f'control/{kind}-quality-{slot:02d}-guard.json';assert json.loads(p.read_text())['status']=='finished';closed.add(p)
        attempted=set()
        # Exact previously attempted paths are excluded, including zero-saving probes.
        for pkg in (R/'research/tokenizer-evidence').iterdir():
            if not pkg.is_dir() or pkg==E or not pkg.name.startswith(('storage-recovery-','filesystem-compression-')):continue
            for path in pkg.glob('*plan*.json'):
                if path.stat().st_size>64*1024**2:continue
                d=json.loads(path.read_text());old=d.get('files',{}) if isinstance(d,dict) else {}
                if isinstance(old,dict):attempted.update(str((R/n).resolve()) if not Path(n).is_absolute() else n for n in old)
                elif isinstance(old,list):attempted.update(str((R/v['path']).resolve()) if not Path(v['path']).is_absolute() else v['path'] for v in old if isinstance(v,dict) and isinstance(v.get('path'),str))
        files={str(p):info(p) for p in closed if p.suffix in ('.json','.jsonl','.log','.txt') and p.stat().st_size>=4096 and not(p.stat().st_flags&32) and str(p) not in attempted}
        assert 0<len(files)<=4000 and sum(v['allocated_bytes'] for v in files.values())<=512*1024**2
        protected={}
        historical=json.loads((R/'research/tokenizer-evidence/storage-recovery-028/raw-completed.json').read_text())
        for row in historical['files']:
            for kind in ('archive','h200'):
                p=Path(row[kind]['path']);v=info(p);assert v['sha256']==row[kind]['sha256'];protected[str(p)]=v
        for slot in range(6):
            row=json.loads((RUN/f'evidence/quality/{slot:02d}/accepted.json').read_text());worker=checked(row['worker'])
            for kind in ('parent','h200'):
                ref=worker[kind];p=Path(ref['path']);v=info(p);assert v['sha256']==ref['sha256'] and v['size']==ref['size'];protected[str(p)]=v
        consumer=subprocess.run(['lsof','-nP','-Fpn','--',*files],capture_output=True,text=True,timeout=30)
        assert consumer.returncode==1 and not consumer.stdout.strip() and not consumer.stderr.strip(),'Closed metadata has a consumer'
        a.publish(E/'plan.json',dict(operation='lossless_APFS_closed_metadata_compression',files=files,source=info(Path(__file__)),protected_affected_models=protected,
            free_before=shutil.disk_usage(R).free,required=REQUIRED,target_with_extra_headroom=TARGET,authorization='New capacity block, standing human recovery authorization. Only new closed metadata; original paths/readable bytes preserved, no checkpoint modified.'))
        saved=0
        with (E/'journal.jsonl').open('x') as log:
            def event(d):log.write(json.dumps(d)+'\n');log.flush();os.fsync(log.fileno())
            for name,v in files.items():
                p=Path(name);check(p,v,True);tmp=p.with_name(p.name+'.storage038.tmp');assert not tmp.exists() and shutil.disk_usage(R).free>=8*1024**3+v['size']
                subprocess.run(['/usr/bin/ditto','--hfsCompression','--noclone',str(p),str(tmp)],check=True,timeout=30);w=a.check(tmp,v)
                with tmp.open('rb') as f:os.fsync(f.fileno())
                check(p,v,True)
                if w['allocated_bytes']>=v['allocated_bytes']:event(dict(event='verified_no_savings',path=name));tmp.unlink();a.syncdir(p.parent);continue
                assert w['flags']&32;event(dict(event='verified_staged_copy',path=name,sha256=v['sha256'],before=v['allocated_bytes'],after=w['allocated_bytes']))
                os.replace(tmp,p);a.syncdir(p.parent);check(p,v);saved+=v['allocated_bytes']-w['allocated_bytes'];event(dict(event='replacement_verified',path=name,sha256=v['sha256']))
        for name,v in files.items():check(Path(name),v)
        for name,v in protected.items():check(Path(name),v,True)
        assert plan['sources']==sources() and plan['quality_sources']==quality_sources() and plan['pilot_sources']==campaign_sources()
        free=shutil.disk_usage(R).free;a.publish(E/'completed.json',dict(status='completed_lossless_metadata_compression',files=len(files),allocated_bytes_reclaimed=saved,
            protected_affected_models_verified=len(protected),all_paths_and_readable_bytes_unchanged=True,no_weights_modified=True,free_after=free,required=REQUIRED,target=TARGET,
            mandatory_deficit=max(0,REQUIRED-free),target_deficit=max(0,TARGET-free)))
        print((E/'completed.json').read_text())
