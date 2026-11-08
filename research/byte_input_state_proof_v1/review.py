"""Independent saved-receipt review; no Torch or candidate numerical calls."""
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from .cases import BACKENDS,VARIANTS,REJECTIONS,IO_REJECTIONS
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'state-review001';SOURCE=BASE/'state-math001'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while x:=f.read(1048576):h.update(x)
    return h.hexdigest()

def read(name):return json.loads((SOURCE/name).read_text())
def save(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())

def tensor(value):
    assert value['dtype']=='torch.float32' and value['layout']=='torch.strided'
    assert len(value['sha256'])==64
    return math.prod(value['shape'])


def main():
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.iterdir())
        save('attempt.json',dict(pid=os.getpid(),no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=240,reserve_bytes=8589934592):
                assert read('STOP')['status']=='complete_original_numerical_state'
                assert read('guard.json')['status']=='finished'
                assert read('result.json')['accepted'] and read('result.json')['caller_rng_restored']
                verified=0;archives=[];checks=[]
                for backend in BACKENDS:
                    for variant in VARIANTS:
                        prefix=backend+'-'+variant
                        expected_count=88 if variant=='native' else 105
                        expected_elements=28492416 if variant=='native' else 16351859
                        complete=read(prefix+'-complete.json')
                        assert complete['parameters']==expected_count and complete['shape']==[8,128,50257]
                        assert complete['payload_rejections']==33 and complete['io_rejections']==5 and complete['nonpristine_rejections']==1
                        zero=read(prefix+'-step0.json');assert zero['before']==zero['after']
                        restore=read(prefix+'-restore.json');end=read(prefix+'-next-state.json')
                        for record in (restore,end):
                            assert record['expected']==record['actual']
                            parameters=record['actual']['parameters']
                            assert len(parameters)==expected_count
                            assert sum(tensor(v) for v in parameters.values())==expected_elements
                            verified+=expected_elements
                            for child in record['actual']['children']:
                                assert set(child['state'])=={g['name'] for g in child['groups']}
                                for fields in child['state'].values():
                                    assert set(fields)==({'momentum_buffer'} if child['kind']=='Muon' else {'step','exp_avg','exp_avg_sq'})
                                    verified+=sum(tensor(v) for v in fields.values())
                        for stage in ('forward','gradients'):
                            control=read(prefix+'-control-next-'+stage+'.json');resumed=read(prefix+'-restored-next-'+stage+'.json')
                            assert control==resumed
                            if stage=='gradients':assert len(control)==expected_count and sum(tensor(v) for v in control.values())==expected_elements
                            else:assert control['logits']['shape']==[8,128,50257] and control['loss']['shape']==[]
                            verified+=sum(tensor(v) for v in control.values())
                        rng=read(prefix+'-rng.json');assert rng['expected']==rng['actual']
                        for name in REJECTIONS:
                            r=read(prefix+'-reject-'+name+'.json');assert r['error'] and r['before']==r['after'] and r['payload_before']==r['payload_after']
                        for name in IO_REJECTIONS:
                            r=read(prefix+'-io-'+name+'.json');assert r['error'] and r['before']==r['after'] and r['original_sha256']==r['after_sha256']
                        r=read(prefix+'-nonpristine.json');assert r['error'] and r['before']==r['after']
                        for step in (1,2):
                            receipt=read(prefix+'-archive-step'+str(step)+'.json');path=Path(receipt['path'])
                            assert path.is_file() and not path.is_symlink() and path.stat().st_nlink==1 and path.stat().st_size==receipt['size']<=536870912
                            assert sha(path)==receipt['sha256'];archives.append(receipt)
                        checks.append(dict(backend=backend,variant=variant,exact=True))
                save('review.json',dict(accepted=True,cases=checks,invalid_transactions=234,
                    exact_coordinate_digests_verified=verified,archives=archives,
                    scope='Complete bitwise tensor digest/shape reconciliation and archive byte hashes. No model rerun or independent deserialization; runtime restore/next-update evidence is from admitted driver. Numerical checkpoint only, no quality or training-worker qualification.'))
            save('STOP',dict(status='complete_original_state_saved_review',no_replay=True))
        except BaseException as exc:
            save('STOP',dict(status='closed_failed_state_saved_review',error=repr(exc),no_replay=True));raise

if __name__=='__main__':main()
