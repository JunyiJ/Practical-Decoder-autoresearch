"""Independent read-only terminal review; no dispatch, retries or weight cleanup."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from src.research.protocol import digest,file_digest
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_wide_hybrid_retention_audit import numerical_review,NUMERICAL_CHECKS
from src.research.tokenizer_wide_hybrid_driver import CORPUS_SHA256
from src.research.tokenizer_wide_hybrid_metadata_writer import json_encoder

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('numerical_procedure',HERE/'run_rank_numerical_v2.py')
p=importlib.util.module_from_spec(spec);spec.loader.exec_module(p)


def review(root,sha,output):
    root=Path(root).resolve();output=Path(output).absolute()
    assert output==output.resolve()
    plan=p.verify(root,sha)
    assert json.loads((root/'completed.json').read_text())==dict(plan_sha256=sha,status='all27_phases_complete_pending_independent_review')
    assert not (root/'failure.json').exists() and not any(root.rglob('*.tmp'))
    # Validate every original file, including the retained partial checkpoints.
    files={str(f):file_digest(f) for f in root.rglob('*') if f.is_file()}
    cases=[]
    for i,case in enumerate(p.CASES):
        phases=[]
        partial_sha=files[str(root/f'{i:02d}/partial.pt')]
        for phase in p.PHASES:
            directory=root/f'{i:02d}'/phase
            assert {f.name for f in directory.iterdir()}=={'process.json','worker.log','guard.json','receipt.json'}
            process=json.loads((directory/'process.json').read_text())
            guard=json.loads((directory/'guard.json').read_text())
            receipt=json.loads((directory/'receipt.json').read_text())
            assert process['status']=='completed' and process['returncode']==0 and process['error'] is None
            assert 0<process['elapsed_seconds']<600 and process['timeout_seconds']==600
            assert process['required_free_bytes']==(8+2*(9-i))*p.GiB and process['reserve_bytes']==8*p.GiB
            assert process['output_limit_bytes']==2*p.GiB and process['log_limit_bytes']==4*1024**2
            obs=process['observation'];assert obs['free_bytes']>=8*p.GiB and obs['output_bytes']<=2*p.GiB
            assert obs['log_bytes']==(directory/'worker.log').stat().st_size<=4*1024**2
            argv=process['argv'];assert argv[:2]==[sys.executable,str(HERE/'run_rank_numerical_v2.py')]
            assert argv[2:11]==['worker','--root',str(root),'--plan-sha256',sha,'--index',str(i),'--phase',phase]
            assert argv[11]=='--global-fd' and argv[13]=='--local-fd' and len(argv)==15
            assert int(argv[12])>2 and int(argv[14])>2 and argv[12]!=argv[14]
            assert guard['status']=='finished' and guard['reason'] is None and guard['pid']==process['pid']
            assert guard['limits']==dict(rss_limit_bytes=4*p.GiB,reserve_bytes=8*p.GiB,timeout_seconds=600)
            assert 0<guard['elapsed_seconds']<600
            assert 0<guard['observation']['peak_rss_bytes']<4*p.GiB and guard['observation']['free_bytes']>=8*p.GiB
            assert receipt['case']==case and receipt['phase']==phase and receipt['plan_sha256']==sha
            assert receipt['config_sha256']==plan['configs'][f'{i:02d}/{phase}']
            assert receipt['completed_steps']==(1 if phase=='partial' else 4)
            steps={'baseline':[1,2,3,4],'partial':[1],'resumed':[2,3,4]}[phase]
            assert receipt['ema_observations']==[dict(step=s,snapshots=s) for s in steps]
            assert receipt['exact_resume_equal']==(phase=='resumed')
            assert receipt['ema_load_and_resume_rejection_checked']==(phase=='resumed')
            assert receipt['test_loss_computed'] is False and receipt['quality_evidence'] is False
            assert receipt['partial_checkpoint_sha256']==(None if phase=='baseline' else partial_sha)
            assert set(receipt['hashes'])=={'model','optimizer','ema','rng','loader_rng'}
            phases.append(dict(phase=phase,receipt=receipt,guard=guard,process_sha256=files[str(directory/'process.json')]))
        assert phases[0]['receipt']['hashes']==phases[2]['receipt']['hashes']
        assert phases[0]['receipt']['hashes']['model']!=phases[1]['receipt']['hashes']['model']
        cases.append(dict(width=512,**case,phases=phases))
    assert len(list(root.glob('*/*/process.json')))==27
    assert p.verify(root,sha)==plan
    assert all(file_digest(Path(n))==s for n,s in files.items())
    output.mkdir(exist_ok=False);meta=output/'meta';meta.mkdir()
    trace=dict(kind='nine_case_rank_numerical_trace_v2',run_root=str(root),plan_sha256=sha,
        reviewer_sha256=file_digest(Path(__file__)),original_files=files,cases=cases,
        geometry_cache_evidence=str(HERE/'rank-review.json'),geometry_cache_evidence_sha256=file_digest(HERE/'rank-review.json'),
        qualifications=['Four CPU updates only; no MPS or long-trajectory stability claim.',
            'Complete state hashes check exact resume; independent scalar EMA recurrence samples first/middle/last per parameter.',
            'Geometry/cache reuse hash-verified historical rank evidence; disk H200 export remains a pilot check.',
            'Original raw phase records and all partial checkpoints are retained; no cleanup performed.'])
    p.publish(meta/'numerical-trace.json',trace)
    context=dict(source_sha256=digest(plan['source']),runtime_sha256=digest(plan['runtime']),corpus_sha256=CORPUS_SHA256)
    numeric=dict(kind='numerical',status='passed',**context,
        checks=dict.fromkeys(('all_declared_ranks',)+NUMERICAL_CHECKS,True),
        cases=[dict(width=512,**case,checks=dict.fromkeys(NUMERICAL_CHECKS,True)) for case in p.CASES],
        artifacts={'meta/numerical-trace.json':file_digest(meta/'numerical-trace.json')})
    p.publish(meta/'numerical.json',numeric)
    ref=dict(path='meta/numerical.json',sha256=file_digest(meta/'numerical.json'))
    assert len(numerical_review(output,ref,context))==2
    print(json.dumps(dict(numerical_ref=ref,metadata_bytes=sum(f.stat().st_size for f in meta.iterdir()),
        cases=9,phases=27,peak_rss_bytes=max(v['guard']['observation']['peak_rss_bytes'] for c in cases for v in c['phases']))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',required=True)
    parser.add_argument('--plan-sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args()
    with CpuPrerequisiteGuard(HERE/'rank-numerical-v2-review-guard.json',ROOT,timeout_seconds=600):
        review(args.root,args.plan_sha256,args.output)
