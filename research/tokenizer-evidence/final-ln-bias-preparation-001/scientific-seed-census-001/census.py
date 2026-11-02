"""One read-only census and prospective reservation; never a launch authorization."""
import argparse,hashlib,json,os,re,subprocess,shutil
from pathlib import Path
from datetime import datetime,timezone
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_bias_campaign_v1.pilot import idle,require_capacity
from research.final_ln_bias_campaign_v1.rules import validate_seeds
from research.ema_horizon_v1.contract import ROOT
HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--prefix',type=int,required=True);args=parser.parse_args()
assert 0<args.prefix<4294967
SEEDS=dict(pilot=args.prefix*1000+1,fresh=[args.prefix*1000+i for i in (11,12,13)],evaluation=args.prefix*1000+3,bootstrap=args.prefix*1000+4)
VALUES=[SEEDS['pilot'],*SEEDS['fresh'],SEEDS['evaluation'],SEEDS['bootstrap']]
SUFFIXES={'.json','.jsonl','.yaml','.yml','.py','.md','.log','.txt','.ipynb'}
ROOTS=['research','research_runs','research_state','src','tests']
PATTERN=re.compile(rb'(?<![A-Za-z0-9_])(?:'+b'|'.join(str(v).encode() for v in VALUES)+rb')(?![A-Za-z0-9_])')

def save(name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2)+'\n').encode();assert len(raw)<=16*1024**2
    with (HERE/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())

def paths():
    found=set(p for p in ROOT.iterdir() if p.is_file() and p.suffix in SUFFIXES)
    for name in ROOTS:
        for parent,dirs,files in os.walk(ROOT/name,followlinks=False):
            dirs[:]=[d for d in dirs if d!='__pycache__' and not (Path(parent)/d).is_relative_to(HERE)]
            for name in files:
                p=Path(parent)/name
                if p.suffix in SUFFIXES:found.add(p)
    return sorted(found)

validate_seeds(SEEDS)
with session_lock(GLOBAL_LOCK.parent):
    idle();require_capacity(shutil.disk_usage(ROOT).free)
    with CpuPrerequisiteGuard(HERE/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        selected=paths();assert len(selected)<100000
        inventory={};matches=[];total=0
        for path in selected:
            assert not path.is_symlink() and all(not x.is_symlink() for x in path.parents)
            before=path.stat();assert before.st_nlink==1 and before.st_size<512*1024**2
            sha=hashlib.sha256();tail=b'';offset=0
            with path.open('rb') as f:
                for chunk in iter(lambda:f.read(1024**2),b''):
                    sha.update(chunk);window=tail+chunk
                    for match in PATTERN.finditer(window):
                        if match.end()>len(tail):matches.append(dict(path=str(path.relative_to(ROOT)),seed=int(match.group()),offset=offset-len(tail)+match.start()))
                    offset+=len(chunk);tail=window[-64:]
            after=path.stat()
            assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
            inventory[str(path.relative_to(ROOT))]=dict(size=before.st_size,sha256=sha.hexdigest());total+=before.st_size
        assert selected==paths(),'Census inventory changed'
        save('inventory.json',inventory)
        result=dict(status='fresh_seeds_reserved' if not matches else 'collision_no_reservation',observed_at=datetime.now(timezone.utc).isoformat(),
            commit_before_census=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),seeds=SEEDS,matches=matches,
            file_count=len(inventory),bytes_scanned=total,roots=ROOTS,root_level_text_included=True,suffixes=sorted(SUFFIXES),
            excluded=[str(HERE.relative_to(ROOT)),'__pycache__','binary checkpoints/archives','raw corpus and untracked external files outside declared roots'],
            inventory_sha256=hashlib.sha256((HERE/'inventory.json').read_bytes()).hexdigest(),audit_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            no_screen_seed=True,quality_plan_frozen=False,launch_permitted=False,holdout_scored=False,
            scope='No candidate whole-word integer appears in recorded repository research/config/source/log text. Final LayerNorm learned/fixed-zero bias arms share pilot and three prospective paired fresh seeds at a fixed LR; evaluation/bootstrap separate. Not a claim about unrelated external experiments; recheck new recorded inputs before launch.')
        save('result.json',result)
        assert not matches,'Seed collision; do not reserve or launch'
    idle()
print(json.dumps(dict(status=result['status'],files=result['file_count'],bytes=result['bytes_scanned'],seeds=SEEDS)))
