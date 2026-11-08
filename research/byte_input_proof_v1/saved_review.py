"""Independent saved-coordinate reconciliation; never imports model or Torch."""
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import numpy as np
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
SOURCE=BASE/'model-math002'
OUT=BASE/'model-saved-review001'


def save(name,value):
    payload=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    assert sum(p.stat().st_size for p in OUT.iterdir())+len(payload)<67108864
    with (OUT/name).open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())


def read(name):return json.loads((SOURCE/name).read_text())


def coordinates(path):
    record=json.loads(path.read_text())
    assert record['data_file']==path.stem+'.coordinates.gz'
    exact=any(x in path.stem for x in ('-initial-', '-code-', '-causality')) or path.stem.startswith(('codebook-','feistel'))
    assert (record['atol'],record['rtol'])==((0.,0.) if exact else (4e-6,4e-5))
    assert record['dtype']=='interleaved_le_float32_float32_uint8'
    assert math.prod(record['shape'])==record['count']
    dtype=np.dtype([('a','<f4'),('b','<f4'),('m','u1')])
    h=hashlib.sha256();count=0;all_pass=True;maximum=0.
    with gzip.open(SOURCE/record['data_file'],'rb') as stream:
        while block:=stream.read(9*65536):
            assert len(block)%9==0
            h.update(block);data=np.frombuffer(block,dtype=dtype)
            a,b=data['a'].astype(np.float64),data['b'].astype(np.float64)
            delta=np.abs(a-b)
            mask=np.isfinite(a)&np.isfinite(b)&(delta<=record['atol']+record['rtol']*np.abs(b))
            assert np.array_equal(data['m'],mask.astype(np.uint8))
            maximum=max(maximum,float(delta[np.isfinite(delta)].max(initial=0.)))
            all_pass=all_pass and bool(mask.all());count+=len(data)
    assert count==record['count'] and count*9==record['raw_bytes']
    assert h.hexdigest()==record['raw_sha256'] and all_pass==record['passed']
    assert maximum==record['max_finite_abs']
    return dict(file=path.name,coordinates=count,passed=all_pass,max_finite_abs=maximum)


def required_files():
    names={'plan.json','attempt.json','guard.json','STOP','result.json','bijection.json',
           'codebook-raw_bytes.json','codebook-nonlexical_codes.json'}
    comparisons=[]
    for variant in ('raw_bytes','nonlexical_codes'):
        comparisons += ['codebook-'+variant+'-rows','codebook-'+variant+'-lengths']
    for length in (1,2):
        comparisons += ['feistel'+str(length)+'-'+suffix for suffix in ('forward','inverse','reference-inverse')]
    # The case-name arrays are read as syntax, not imported or executed.
    import ast
    tree=ast.parse((ROOT/'research/byte_input_proof_v1/cases.py').read_text())
    negative={n.targets[0].id:ast.literal_eval(n.value) for n in tree.body
              if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name)
              and n.targets[0].id.endswith('REJECTIONS')}
    invalid_count=0
    for backend in ('cpu','mps'):
        for variant in ('native','raw_bytes','nonlexical_codes'):
            prefix=backend+'-'+variant
            initial=read(prefix+'-initialization.json')
            inventory=initial['inventory'];native=variant=='native'
            assert len(inventory)==(88 if native else 105)
            assert len({n for n,s in inventory})==len(inventory)
            assert sum(math.prod(s) for n,s in inventory)==(28492416 if native else 16351859)
            assert all(initial['exact_values'].values()) and initial['native_first_rng_equal']
            complete=read(prefix+'-complete.json')
            assert complete['updates']==2 and complete['forward_shape']==[8,128,50257]
            groups=read(prefix+'-optimizer-groups.json')
            assert groups['actual']==groups['reference']
            muon,adam=groups['actual']
            assert muon['type']=='Muon' and adam['type']=='AdamW'
            assert [len(muon['groups']),len(adam['groups'])]==([44,44] if native else [53,52])
            state_fields={}
            for child in groups['actual']:
                for group in child['groups']:
                    assert len(group['names'])==1
                    n=group['names'][0];assert n not in state_fields
                    options=group['options'];assert options['weight_decay']==0.
                    if child['type']=='Muon':
                        assert options['lr']==.000625 and options['adjust_lr_fn']=='match_rms_adamw'
                        state_fields[n]=('momentum_buffer',)
                    else:
                        assert options['lr']==.00125 and options['eps']==1e-6 and options['betas']==[.9,.999]
                        state_fields[n]=('step','exp_avg','exp_avg_sq')
            assert set(state_fields)=={n for n,s in inventory}
            for n,shape in inventory:
                comparisons += [prefix+'-'+kind+'-'+n for kind in ('initial','gradient','update1','update2')]
                for step in (1,2):
                    comparisons += [prefix+'-state'+str(step)+'-'+n+'-'+field for field in state_fields[n]]
            comparisons += [prefix+'-'+kind for kind in ('logits','loss','causality')]
            if not native:comparisons += [prefix+'-code-ids',prefix+'-code-lengths']
            invalids=negative['INPUT_REJECTIONS']+negative['OPTIMIZER_REJECTIONS']
            if backend=='cpu':invalids+=negative['CPU_ONLY_REJECTIONS']
            assert complete['rejected']==len(invalids)
            for n in invalids:
                name=prefix+'-reject-'+n+'.json';names.add(name);record=read(name)
                assert record['case']==n and record['rejected'] and record['preserved']
                assert record['before']==record['after'] and record['error']
                invalid_count+=1
            names.update(prefix+suffix for suffix in ('-initialization.json','-optimizer-groups.json','-complete.json'))
    for name in comparisons:names.update((name+'.json',name+'.coordinates.gz'))
    assert invalid_count==357
    return names,comparisons


def main():
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not list(OUT.iterdir()),'One original saved review'
        assert (SOURCE/'STOP').is_file(),'Require terminal numerical evidence'
        save('attempt.json',dict(pid=os.getpid(),source=str(SOURCE),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=600,
                                     rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
                terminal=read('STOP');complete=terminal['status']=='complete_original_actual_model_math'
                records=[]
                for path in sorted(SOURCE.glob('*.json')):
                    # Every complete record is independently checked, even after
                    # a failed/incomplete run. A partial gzip is preserved, not accepted.
                    record=json.loads(path.read_text())
                    if 'data_file' in record:records.append(coordinates(path))
                expected_count=None
                if complete:
                    names,comparisons=required_files()
                    assert names=={p.name for p in SOURCE.iterdir()}
                    assert {r['file'] for r in records}=={n+'.json' for n in comparisons}
                    assert read('guard.json')['status']=='finished'
                    assert read('result.json')['caller_rng_restored'] is True
                    expected_count=len(comparisons)
                accepted=complete and all(r['passed'] for r in records)
                save('coordinates.json',records)
                save('review.json',dict(accepted=accepted,numerical_complete=complete,
                    complete_records=len(records),expected_records=expected_count,
                    all_coordinates=sum(r['coordinates'] for r in records),
                    failed_records=[r['file'] for r in records if not r['passed']],
                    source_terminal=terminal,scope='saved finite model implementation evidence only'))
            save('STOP',dict(status='complete_original_saved_review',no_replay=True))
        except BaseException as exc:
            save('STOP',dict(status='closed_failed_saved_review',error=repr(exc),no_replay=True));raise


if __name__=='__main__':main()
