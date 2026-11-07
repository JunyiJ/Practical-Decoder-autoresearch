"""New unused scientific-role reservation after complete gate proof; no RNG draws."""
import hashlib,json,os,re,subprocess,sys
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_quality_v1 import contract as C
from research.tokenizer_mla_gate_worker_v1.execution import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;R=E/'gate-quality-seed-census001'
def publish(p,v):
    with p.open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
with session_lock(GLOBAL_LOCK.parent):
    idle();seeds=json.loads(sys.argv[1]);C.seeds_valid(seeds)
    require(not R.exists() and not C.RUN.exists() and not C.FREEZE.exists(),'Fresh unused census before any production attempt required')
    review=json.loads((E/'gate-quality-review001.json').read_text());require(review['status']=='full_gate_quality_correctness_independently_reviewed' and review['source_bindings']==1084,'Actual complete independently reviewed correctness gate required')
    scope=['research','research_runs','research_state','src','tests'];globs=['*.py','*.json','*.jsonl','*.yaml','*.yml','*.md','*.log']
    flags=['--hidden','--no-ignore'];filters=sum((['-g',g] for g in globs),[])
    baseline=subprocess.check_output(['rg','--files',*flags,*scope,*filters],cwd=ROOT,text=True).splitlines();baseline=sorted(set(baseline))
    R.mkdir()
    try:
        with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120):
            values=[seeds['screen'],seeds['evaluation'],*seeds['fresh'],seeds['bootstrap']]
            pattern=r'(?<![0-9A-Za-z_])(?:'+'|'.join(map(str,values))+r')(?![0-9A-Za-z_])'
            found=subprocess.run(['rg',*flags,'-l','-P',pattern,*scope,*filters],cwd=ROOT,text=True,capture_output=True)
            require(found.returncode==1 and not found.stdout and not found.stderr,'Proposed scientific role already recorded or audit incomplete')
            inventory=[];total=0;chunks=[]
            for name in baseline:
                p=ROOT/name;before=p.stat();require(p.is_file() and not p.is_symlink(),'Regular prior seed-evidence file required')
                h=hashlib.sha256()
                with p.open('rb') as f:
                    for b in iter(lambda:f.read(1024**2),b''):h.update(b)
                after=p.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Recorded seed-evidence file changed during read')
                inventory.append(dict(path=name,bytes=after.st_size,sha256=h.hexdigest()));total+=after.st_size
                if len(inventory)==5000:
                    p=R/f'inventory-{len(chunks):03d}.json';publish(p,inventory);chunks.append(reference(p));inventory=[]
            if inventory:
                p=R/f'inventory-{len(chunks):03d}.json';publish(p,inventory);chunks.append(reference(p))
            require(sum(p.stat().st_size for p in R.iterdir() if p.is_file())<32*1024**2,'Bounded census metadata required')
            publish(R/'result.json',dict(status='reserved_unused',source=reference(__file__),seeds=seeds,scope=scope,globs=globs,pattern=pattern,rg_exit_code=1,prior_matches=[],prior_recorded_files=len(baseline),prior_recorded_bytes=total,inventory=chunks,correctness_review=reference(E/'gate-quality-review001.json'),scientific_training_started=False,rng_draws=0,model_loaded=False,holdout_encoded_or_scored=False))
        publish(R/'STOP',dict(status='complete_unused_scientific_roles_reserved',result=reference(R/'result.json'),never_rerun=True));print(json.dumps(dict(status='reserved_unused',files=len(baseline),bytes=total,seeds=seeds)),flush=True)
    except BaseException as exc:
        if not (R/'STOP').exists():publish(R/'STOP',dict(status='closed_failed_census',error=repr(exc),no_retry=True))
        raise
