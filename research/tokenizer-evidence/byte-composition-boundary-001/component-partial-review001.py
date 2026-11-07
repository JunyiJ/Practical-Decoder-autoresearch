"""FIRST saved failure evidence reconciliation; no candidate/oracle/Torch replay."""
from array import array
import base64
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zlib
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'component-partial-review001'
def need(value,reason):
    if not value:raise ValueError(reason)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def write(name,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    need(sum(p.stat().st_size for p in OUT.iterdir())+len(raw)<2097152,'Review output cap')
    with (OUT/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def unpack(r,kind,count):
    need(r['format']=='little_'+kind and r['count']==count,'Saved dtype/count')
    size=count*(8 if kind=='d' else 1);d=zlib.decompressobj()
    raw=d.decompress(base64.b64decode(r['zlib_base64'],validate=True),size+1)
    need(len(raw)==size and d.eof and not d.unused_data and not d.unconsumed_tail and sha(raw)==r['sha256'],'Exact complete bounded vector')
    a=array(kind);a.frombytes(raw)
    if sys.byteorder!='little':a.byteswap()
    return a
def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        need({p.name for p in OUT.iterdir()}=={'plan.json'},'One original saved-failure review')
        write('attempt.json',dict(pid=os.getpid(),plan_sha256=expected,no_retry=True,no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                raw=(OUT/'plan.json').read_bytes();need(sha(raw)==expected,'Independent review plan')
                p=json.loads(raw)
                for pin in p['pins']:
                    data=Path(pin['path']).read_bytes();need(len(data)==pin['size'] and sha(data)==pin['sha256'],'Own review source pin')
                root=BASE/'component-math001';original=json.loads((root/'plan.json').read_text())
                need(json.loads((root/'STOP').read_text())['status']=='closed_failed_original_component_math','Original failed STOP preserved')
                report=json.loads((root/'cpu-short_nul.json').read_text())
                need(set(report['comparisons'])==set(original['shapes'])|{'output'},'All first-CPU forward/gradient arrays present')
                coordinates=0;maximum=0.
                for name,c in report['comparisons'].items():
                    need(c['shape']==([1,256] if name=='output' else original['shapes'][name]),'Exact saved coordinate shape')
                    need(c['atol']==4e-6 and c['rtol']==4e-5,'Original fixed gates')
                    n=math.prod(c['shape']);a=unpack(c['actual'],'d',n);e=unpack(c['expected'],'d',n);mask=unpack(c['mask'],'B',n)
                    flags=[math.isfinite(x) and math.isfinite(y) and abs(x-y)<=4e-6+4e-5*abs(y) for x,y in zip(a,e)]
                    delta=max(abs(x-y) for x,y in zip(a,e));need(list(mask)==flags and all(flags) and c['all_pass'] is True and c['max_abs']==delta,'ALL saved original numerical predicates reconciled')
                    coordinates+=n;maximum=max(maximum,delta)
                before=report['before'];after=report['after']
                need(before!=after and report['preserved'] is False,'Original preservation predicate really failed')
                differences={k for k in before if before[k]!=after[k]};need(differences=={'rng'},'Only RNG record differs')
                details={}
                for backend in ['cpu','mps']:
                    a=before['rng'][backend];b=after['rng'][backend]
                    changed=[k for k in a if a[k]!=b[k]];details[backend]=changed
                    need({k:v for k,v in a.items() if k!='storage_ptr'}=={k:v for k,v in b.items() if k!='storage_ptr'},'Exact RNG payload/shape/dtype/device/layout hashes equal')
                need(details=={'cpu':[],'mps':['storage_ptr']},'Saved RNG-copy address-only discrepancy')
                sources=original['execution_sources']
                for pin in sources:
                    raw=Path(pin['path']).read_bytes();need(len(raw)==pin['size'] and sha(raw)==pin['sha256'],'FIRST final original source-pin preservation')
                manifest=[]
                for path in sorted(root.iterdir()):
                    need(path.is_file() and not path.is_symlink(),'Flat regular original evidence')
                    data=path.read_bytes();manifest.append(dict(path=str(path),size=len(data),sha256=sha(data)))
                need(not (root/'result.json').exists(),'Full original acceptance unreached')
                write('result.json',dict(status='complete_FIRST_saved_component_failure_evidence_only',accepted_component_or_complete_fixture=False,comparisons=19,coordinates_reconciled=coordinates,all_saved_first_CPU_numerical_coordinates_pass=True,max_abs=maximum,original_snapshot_false=True,only_discrepancy='Returned MPS RNG-state copy storage address; exact saved state bytes/shape/dtype/device/layout hashes equal',RNG_record_changed_fields=details,source_pins_unchanged=len(sources),original_payloads=manifest,no_component_oracle_Torch_model_RNG_replay=True,unreached='Other5CPUcases,all6MPScases,all113invalidtransactions,CPU/MPScomparisonsandfinalRNG/sourcecountacceptance',no_quality_or_model_adoption=True))
            write('STOP',dict(status='complete_FIRST_original_saved_failure_review',no_retry=True,no_replay=True))
        except BaseException as exc:write('STOP',dict(status='closed_failed_original_partial_review',error=repr(exc),no_retry=True,no_replay=True));raise
if __name__=='__main__':main(sys.argv[1])
