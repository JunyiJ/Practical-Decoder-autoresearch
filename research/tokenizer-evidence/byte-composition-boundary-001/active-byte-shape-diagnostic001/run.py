"""Separate shape/stride diagnostic; does not rerun or qualify a model candidate."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import traceback

ROOT=Path(__file__).resolve().parents[4]
OUT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def publish(name,value):
    with (OUT/name).open('x') as f:
        json.dump(value,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())


def run():
    assert not (OUT/'guard.json').exists()
    plan=read(OUT/'plan.json');assert sha(Path(__file__))==plan['runner_sha256']
    for r in plan['bindings']:
        p=ROOT/r['path'];assert p.stat().st_size==r['size'] and sha(p)==r['sha256']
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        from research.byte_input_quality_recovery_v1.guard import Guard
        with Guard(OUT/'guard.json',ROOT,timeout_seconds=180,rss_limit_bytes=4*1024**3,reserve_bytes=8*1024**3,
                   weight_roots=[ROOT/'research_runs/byte-input-quality-v1-001',ROOT/'research_runs/byte-input-quality-recovery-v1-001']) as guard:
            import torch
            assert torch.__version__==plan['torch_version'] and torch.version.git_version==plan['torch_commit']
            assert torch.backends.mps.is_available();torch.set_num_threads(4);guard.torch=torch
            assert 'PYTORCH_MPS_PREFER_METAL' not in os.environ
            assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in ('1','true','True')
            cpu_rng=torch.get_rng_state().clone();mps_rng=torch.mps.get_rng_state().clone()
            def digest(t):return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()
            def compare(reference,candidate):
                diff=(reference.double()-candidate.double()).abs()
                return dict(exact=torch.equal(reference,candidate) and digest(reference)==digest(candidate),
                            unequal_coordinates=int((reference!=candidate).sum()),max_abs=float(diff.max()),
                            reference_sha256=digest(reference),candidate_sha256=digest(candidate))
            records=[]
            full_vocabulary=torch.arange(259,device='mps')
            for K in plan['reduction_lengths']:
                for count in range(1,260):
                    indices=(torch.arange(K,dtype=torch.int64)%count).to('mps')
                    full=(indices[:,None]==full_vocabulary[None,:]).float()
                    compact=(indices[:,None]==torch.arange(count,device='mps')[None,:]).float()
                    selected=full[:,:count]
                    assert torch.equal(compact,selected)
                    for pattern in plan['patterns']:
                        q=torch.arange(K*15,dtype=torch.int64)
                        values=((q*29+7)%1021-510).float()/1009
                        if pattern=='amplified':values=values*torch.where(q%101==0,16.,1.)
                        upstream=values.reshape(K,15).to('mps')
                        reference=(full.T@upstream).cpu()
                        compact_result=(compact.T@upstream).cpu()
                        strided_result=(selected.T@upstream).cpu()
                        zeros=torch.zeros_like(reference[count:])
                        assert torch.equal(reference[count:],zeros) and digest(reference[count:])==digest(zeros)
                        row=dict(K=K,output_rows=count,pattern=pattern,logical_inputs_equal=True,
                                 actual_input_rows=min(count,K),full_stride=list(full.T.stride()),
                                 compact_stride=list(compact.T.stride()),selected_stride=list(selected.T.stride()),
                                 full_shape=list(full.T.shape),compact_shape=list(compact.T.shape),
                                 inactive_reference_rows_exact_zero=True,
                                 compact=compare(reference[:count],compact_result),
                                 same_stride_view=compare(reference[:count],strided_result))
                        records.append(row)
                    del full,compact,selected,upstream,reference,compact_result,strided_result
                publish(f'length{K:04}.json',[r for r in records if r['K']==K])
                torch.mps.empty_cache()
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(mps_rng,torch.mps.get_rng_state())
            groups=[]
            for K in plan['reduction_lengths']:
                for pattern in plan['patterns']:
                    subset=[r for r in records if r['K']==K and r['pattern']==pattern]
                    groups.append(dict(K=K,pattern=pattern,cases=len(subset),
                        compact_mismatch_counts=[r['output_rows'] for r in subset if not r['compact']['exact']],
                        same_stride_mismatch_counts=[r['output_rows'] for r in subset if not r['same_stride_view']['exact']],
                        compact_max_abs=max(r['compact']['max_abs'] for r in subset),
                        same_stride_max_abs=max(r['same_stride_view']['max_abs'] for r in subset)))
            result=dict(completed=True,groups=groups,case_count=len(records),comparison_count=2*len(records),
                        torch_version=torch.__version__,torch_commit=torch.version.git_version,macos=platform.mac_ver()[0],
                        no_candidate_executed=True,no_candidate_adopted=True,training_updates=0,quality_evaluations=0,
                        rng_unchanged=True,plan_sha256=sha(OUT/'plan.json'),runner_sha256=sha(Path(__file__)),
                        limits=plan['interpretation_limits'])
        assert read(OUT/'guard.json')['status']=='finished'
        result['guard_sha256']=sha(OUT/'guard.json');publish('result.json',result)
        publish('STOP',dict(status='complete_shape_stride_diagnostic_no_adoption',no_replay=True))
    print(json.dumps(result))


if __name__=='__main__':
    try:run()
    except Exception:
        publish('failure.json',dict(traceback=traceback.format_exc(),no_automatic_replay=True));raise
