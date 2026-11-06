"""Prospective pilot/screen/evaluation/bootstrap/fresh seed nonuse census."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.body_allocation_model_v1.contract import ROOT,require
from research.final_ln_scale_quality_v3.support import idle
B=Path(__file__).resolve().parent/'census001'
p=argparse.ArgumentParser();p.add_argument('--prefix',type=int,required=True);a=p.parse_args()
require(0<a.prefix<4294967,'Bounded seed namespace required')
seeds=dict(pilot=a.prefix*1000+1,screen=a.prefix*1000+2,evaluation=a.prefix*1000+3,bootstrap=a.prefix*1000+4,fresh=[a.prefix*1000+i for i in (11,12,13)])
values=[seeds[k] for k in ('pilot','screen','evaluation','bootstrap')]+seeds['fresh']
pattern=re.compile(rb'(?<![A-Za-z0-9_])(?:'+b'|'.join(str(v).encode() for v in values)+rb')(?![A-Za-z0-9_])')
suffix={'.json','.jsonl','.yaml','.yml','.py','.md','.log','.txt','.ipynb'}
roots=['research','research_runs','research_state','src','tests']
def files():
    out={p for p in ROOT.iterdir() if p.is_file() and p.suffix in suffix}
    for name in roots:
        for parent,dirs,names in os.walk(ROOT/name,followlinks=False):
            dirs[:]=[d for d in dirs if d!='__pycache__' and not (Path(parent)/d).is_relative_to(B)]
            out.update(Path(parent)/n for n in names if Path(n).suffix in suffix)
    return sorted(out)
def save(name,value):
    with (B/name).open('x') as f:json.dump(value,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not B.exists(),'Closed census cannot rerun');B.mkdir()
    with CpuPrerequisiteGuard(B/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        selected=files();inventory={};matches=[];require(len(selected)<100000,'Bounded census')
        for path in selected:
            before=path.stat();require(not path.is_symlink() and before.st_nlink==1 and before.st_size<512*1024**2,'Regular bounded recorded file required')
            h=hashlib.sha256();tail=b'';offset=0
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024**2),b''):
                    h.update(chunk);window=tail+chunk
                    for m in pattern.finditer(window):
                        if m.end()>len(tail):matches.append(dict(path=str(path.relative_to(ROOT)),seed=int(m.group()),offset=offset-len(tail)+m.start()))
                    tail=window[-64:];offset+=len(chunk)
            after=path.stat();require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Census input changed')
            inventory[str(path.relative_to(ROOT))]=dict(size=before.st_size,sha256=h.hexdigest())
        require(selected==files(),'Census inventory changed');save('inventory.json',inventory)
        result=dict(status='reserved_unused' if not matches else 'collision_no_reservation',seeds=seeds,matches=matches,files=len(inventory),bytes_scanned=sum(v['size'] for v in inventory.values()),
            roots=roots,suffixes=sorted(suffix),inventory_sha256=hashlib.sha256((B/'inventory.json').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            scope='Recorded repository text/config/logs only; excludes binary payloads/archives, raw corpus, external files and this new census directory. No training or holdout.',
            quality_frozen=False,launch_permitted=False)
        save('result.json',result);require(not matches,'Seed collision; no reservation')
    print(json.dumps(dict(status=result['status'],files=result['files'],seeds=seeds)))
