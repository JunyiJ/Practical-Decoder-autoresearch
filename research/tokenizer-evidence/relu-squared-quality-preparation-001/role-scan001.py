"""NEW literal stream/chunk metadata oracle; no recorded-file census or RNG calls."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from research.recorded_role_scan_v1 import scan
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[3]
R=Path(__file__).with_suffix('')
NEEDLES={b'73110492851',b'9034682710856241957',b'51840392617'}


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def publish(name,value):
    raw=(json.dumps(value,indent=2,sort_keys=True)+'\n').encode()
    assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=2*1024**2
    with (R/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def oracle(raw):
    if b'\0' in raw:return []
    matches=[]
    for value in NEEDLES:
        matches.extend(dict(offset=m.start(),value=value.decode()) for m in re.finditer(rb'(?<![0-9])'+re.escape(value)+rb'(?![0-9])',raw))
    return sorted(matches,key=lambda v:v['offset'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);arg=ap.parse_args()
    assert sha(R/'plan.json')==arg.plan_sha and set(p.name for p in R.iterdir())=={'plan.json','source-review.json'}
    plan=json.loads((R/'plan.json').read_text())
    for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        publish('attempt.json',dict(pid=os.getpid(),plan_sha256=arg.plan_sha))
        try:
            with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
                a,b,c=sorted(NEEDLES)
                literals=[b'',a,b'xx'+a+b'yy',b'-'+b+b'.0',a+b','+b+b','+c,
                    b'0'+a,a+b'0',b'9'*65+a+b'8'*65,b'7'*131+b'!'+c+b'?',
                    b'8'*(1024**2)+b'|'+b,b'\xff'+a+b'\xfe',a+b'\0'+c,
                    b'11'+b'\0'+b'22',b'1234567890',b'73110492850',b'+'+a+b'\n'+a]
                rows=[]
                for ordinal,raw in enumerate(literals):
                    expected=oracle(raw)
                    for size in [1,2,7,10,11,18,19,20,31,64,8192,1024**2]:
                        chunks=[raw[i:i+size] for i in range(0,len(raw),size)]
                        got=scan(iter(chunks),NEEDLES)
                        assert got['binary']==(b'\0' in raw)
                        assert got['matches']==expected,(ordinal,size,got['matches'],expected)
                        if not got['binary']:
                            assert got['bytes']==len(raw) and got['sha256']==hashlib.sha256(raw).hexdigest()
                        rows.append(dict(literal=ordinal,chunk_bytes=size,input_bytes=len(raw),matches=len(expected),binary=got['binary']))
                # Every split location, plus empty chunks and a long continuation
                # crossing multiple chunks, independently checks pending offsets.
                raw=b'x'+a+b' y'+b+b'z '+b'4'*101+c+b'!'
                for cut in range(len(raw)+1):
                    got=scan(iter([raw[:cut],b'',raw[cut:]]),NEEDLES)
                    assert got['matches']==oracle(raw) and got['sha256']==hashlib.sha256(raw).hexdigest()
                invalids=0
                for needles,chunks in [(set(),[b'x']),({b''},[b'x']),({b'a'},[b'x']),({b'1'*20},[b'x']),({1},[b'x']),({b'1'},['text'])]:
                    try:scan(iter(chunks),needles)
                    except ValueError:invalids+=1
                    else:raise AssertionError('Invalid scanner contract accepted')
                assert 'torch' not in sys.modules and 'numpy' not in sys.modules
                for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
                publish('result.json',dict(status='passed_new_literal_bounded_role_stream_oracle',rows=rows,all_split_positions=len(raw)+1,invalid_cases=invalids,
                    no_recorded_files_census_or_torch_numpy_model_rng_seed_calls=True,
                    numeric_literals='Only metadata test strings,not assigned scientific/fixture RNG roles',source_pins_unchanged=True))
            publish('STOP',dict(status='complete_new_literal_role_scanner_gate',no_replay=True,no_retry=True))
        except BaseException as exc:
            if not (R/'STOP').exists():publish('STOP',dict(status='closed_failed',no_retry=True,error=repr(exc)))
            raise

if __name__=='__main__':main()
