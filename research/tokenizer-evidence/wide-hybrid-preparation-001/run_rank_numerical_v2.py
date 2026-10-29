"""Fresh finite CPU prerequisite panel, separate from the closed combined fixture.

Three isolated processes per family/rank; no BytesIO checkpoint, model-math
patch or change to production resume lifetimes. Numerical evidence only.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.research.protocol import digest,file_digest
from src.research.runner import source_fingerprint
from src.research.tokenizer_study import runtime_identity
from src.research.tokenizer_wide_hybrid_config import rank_config
from src.research.tokenizer_wide_hybrid_cleanup import GLOBAL_LOCK,_locks
from src.research.tokenizer_wide_hybrid_driver import CORPUS_SHA256
from src.research.tokenizer_serial_process import run_process
from src.research import cpu_prerequisite_guard as guard

EVIDENCE=Path(__file__).resolve().parent
CASES=[dict(family=f,divisor=d) for f in ('hybrid','gated_mla','kda') for d in (8,4,2)]
PHASES=('baseline','partial','resumed')
GiB=1024**3


def publish(path,value):
    data=json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n'
    if len(data.encode())>128*1024:raise ValueError('Numerical document cap')
    with path.open('x') as stream:
        stream.write(data);stream.flush();os.fsync(stream.fileno())


def configuration(root,index,phase):
    case=CASES[index]
    cfg=rank_config(root/f'{index:02d}',case['family'],512,3490,.00125,4,'cpu',
                    bottleneck_divisor=case['divisor'])
    t=cfg['training']
    t.update(eval_every=2,eval_iters=1,checkpoint_every=0,keep_best_checkpoint=False,
             metrics_file=None,summary_file=None,max_train_seconds=1e-9 if phase=='partial' else None,
             resume_from=str(root/f'{index:02d}/partial.pt') if phase=='resumed' else None)
    t['weight_averaging'].update(start_step=1,interval=1)
    return cfg


def verify(root,expected):
    assert file_digest(root/'plan.json')==expected
    plan=json.loads((root/'plan.json').read_text())
    assert plan['status']=='frozen_cpu_prerequisite_only' and plan['cases']==CASES
    assert plan['phases']==list(PHASES) and not (root/'STOP').exists()
    assert plan['source']==source_fingerprint() and plan['runtime']==runtime_identity()
    assert plan['script_sha256']==file_digest(Path(__file__))
    assert all(file_digest(Path(p))==s for p,s in plan['files'].items())
    for i in range(9):
        for phase in PHASES:assert digest(configuration(root,i,phase))==plan['configs'][f'{i:02d}/{phase}']
    return plan


def freeze(root):
    root=Path(root).absolute();assert root==root.resolve()
    root.mkdir(exist_ok=False)
    cache=ROOT/'research_state/tiktoken-cache'
    assets=json.loads((EVIDENCE/'training-input-cache-restoration.json').read_text())['files']
    files={row['cache_path']:row['sha256'] for row in assets}
    files[str(Path(configuration(root,0,'baseline')['data']['path']).resolve())]=CORPUS_SHA256
    historical=json.loads((EVIDENCE/'rank-review.json').read_text())
    files.update({str(ROOT/n):s for n,s in historical['hashes'].items()})
    for name in ('rank-review.json','cpu-resource-failure-001.json','cpu-failure-review-001.md',
                 'rank-numerical-v2-procedure.md'):
        files[str(EVIDENCE/name)]=file_digest(EVIDENCE/name)
    assert all(file_digest(Path(n))==s for n,s in files.items())
    plan=dict(kind='wide_hybrid_rank_numerical_v2',status='frozen_cpu_prerequisite_only',
        output=str(root),cases=CASES,phases=list(PHASES),source=source_fingerprint(),runtime=runtime_identity(),
        script_sha256=file_digest(Path(__file__)),files=files,cache=str(cache),
        configs={f'{i:02d}/{p}':digest(configuration(root,i,p)) for i in range(9) for p in PHASES},
        timeout_per_phase_seconds=600,rss_limit_bytes=4*GiB,reserve_bytes=8*GiB,
        per_case_output_cap=2*GiB,initial_required_free_bytes=26*GiB,
        max_phases=27,cpu_threads=4,stop_on_first_failure=True,retry=False,
        closed_combined_fixture_called=False,research_training_authorized=False)
    publish(root/'plan.json',plan)
    print(file_digest(root/'plan.json'))


def tree_hash(value):
    """Stream tensor bytes in bounded chunks; reject every nonfinite float."""
    import torch
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):
            assert v.device.type=='cpu'
            flat=v.detach().contiguous().view(-1)
            h.update(str((v.dtype,tuple(v.shape))).encode())
            for part in flat.split(262144):
                if part.is_floating_point():assert bool(torch.isfinite(part).all())
                h.update(memoryview(part.numpy()).cast('B'))
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(repr(k).encode());visit(v[k])
        elif isinstance(v,(tuple,list)):
            for x in v:visit(x)
        else:h.update(repr(v).encode())
    visit(value);return h.hexdigest()


def recheck_prefix(files):
    assert all(file_digest(Path(n))==s for n,s in files.items()),'Earlier numerical evidence changed'


def worker(root,sha,index,phase,fds):
    root=Path(root);plan=verify(root,sha);_locks(root/'evidence',fds)
    os.environ['TIKTOKEN_CACHE_DIR']=plan['cache']
    out=root/f'{index:02d}'/phase
    assert out.is_dir() and not (out/'receipt.json').exists()
    observation=guard.observation
    def observed(path):
        if (root/'STOP').exists():raise RuntimeError('Numerical panel STOP')
        return observation(path)
    guard.observation=observed
    with guard.CpuPrerequisiteGuard(out/'guard.json',ROOT,timeout_seconds=600):
        import torch
        from omegaconf import OmegaConf
        from unittest.mock import patch
        import src.train.train as training
        from src.train.weight_averaging import TailWeightAverages
        from src.utils.checkpoint import load_model_weights
        torch.set_num_threads(4)
        cfg=OmegaConf.create(configuration(root,index,phase));saved=[];observations=[]
        scalar={};max_error=0.;original_update=TailWeightAverages.update
        original_load=TailWeightAverages.load_state_dict
        original_save=training.save_checkpoint
        original_read=training.read_checkpoint
        def read(path,map_location='cpu'):
            receipt=json.loads((root/f'{index:02d}/partial/receipt.json').read_text())
            expected=receipt['partial_checkpoint_sha256']
            assert Path(path)==root/f'{index:02d}/partial.pt' and file_digest(Path(path))==expected
            result=original_read(path,map_location=map_location)
            assert file_digest(Path(path))==expected
            return result
        def samples(panel):
            return {name:[float(p.detach().view(-1)[i]) for i in (0,p.numel()//2,p.numel()-1)]
                    for name,p in panel.items()}
        def loaded(self,state,steps):
            original_load(self,state,steps)
            scalar.update(samples(state['averages']['200']))
        def update(self,steps):
            nonlocal max_error
            values=samples(dict(self.parameters))
            original_update(self,steps)
            beta=math.exp(-math.log(2)/200)
            for name,vals in values.items():
                expected=vals if steps==1 else [beta*a+(1-beta)*b for a,b in zip(scalar[name],vals)]
                actual=samples({name:self.averages['200'][name]})[name]
                for a,b in zip(actual,expected):
                    max_error=max(max_error,abs(a-b));assert math.isclose(a,b,rel_tol=3e-6,abs_tol=1e-7)
                scalar[name]=expected
            observations.append(dict(step=steps,snapshots=self.snapshots))
            print('completed_optimizer_ema_step',steps,flush=True)
        def save(path,model,optimizer,step,cfg,training_state):
            assert not saved and step==(1 if phase=='partial' else 4)
            ema=training_state['weight_averaging_state'];assert ema['snapshots']==step
            hashes={name:tree_hash(value) for name,value in dict(model=model.state_dict(),
                optimizer=optimizer.state_dict(),ema=ema,rng=training_state['torch_rng'],
                loader_rng=training_state['loader_rng']).items()}
            if phase=='partial':
                target=root/f'{index:02d}/partial.pt';assert not target.exists() and not target.with_suffix('.pt.tmp').exists()
                original_save(target,model,optimizer=optimizer,step=step,cfg=cfg,training_state=training_state)
            if phase=='resumed':
                baseline=json.loads((root/f'{index:02d}/baseline/receipt.json').read_text())
                assert hashes==baseline['hashes'],'Exact resumed state differs from baseline'
                # Exercise the existing inference-only lineage/load rule with
                # the complete stored EMA, without constructing a second model.
                buffers={n:tree_hash(v) for n,v in model.state_dict().items() if n not in ema['averages']['200']}
                derived=dict(cfg=OmegaConf.to_container(cfg),model_state=dict(model.state_dict()),
                    averaged_weights={'half_life':200},tokenizer_contract=training_state['contract']['tokenizer'])
                derived['model_state'].update(ema['averages']['200'])
                load_model_weights(model,derived)
                assert all(torch.equal(dict(model.named_parameters())[n],v) for n,v in ema['averages']['200'].items())
                assert all(tree_hash(model.state_dict()[n])==s for n,s in buffers.items())
                try:load_model_weights(model,derived,for_resume=True)
                except ValueError:pass
                else:raise AssertionError('EMA export permitted optimizer resume')
            saved.append(hashes)
        print('starting_phase',index,phase,flush=True)
        with patch.object(training,'save_checkpoint',save),patch.object(training,'read_checkpoint',read),patch.object(TailWeightAverages,'update',update),patch.object(TailWeightAverages,'load_state_dict',loaded):
            result=training.run_training(cfg)
        assert result['completed_steps']==(1 if phase=='partial' else 4) and len(saved)==1
        assert [r['step'] for r in observations]==({'baseline':[1,2,3,4],'partial':[1],'resumed':[2,3,4]}[phase])
        verify(root,sha);_locks(root/'evidence',fds)
        publish(out/'receipt.json',dict(kind='rank_numerical_phase_v2',case=CASES[index],phase=phase,
            plan_sha256=sha,config_sha256=digest(configuration(root,index,phase)),hashes=saved[0],
            completed_steps=result['completed_steps'],ema_observations=observations,
            ema_sample_max_absolute_error=max_error,parameter_sample_positions='first_middle_last_each_parameter',
            partial_checkpoint_sha256=file_digest(root/f'{index:02d}/partial.pt') if phase in ('partial','resumed') else None,
            exact_resume_equal=phase=='resumed',ema_load_and_resume_rejection_checked=phase=='resumed',
            test_loss_computed=False,quality_evidence=False))


def run(root,sha):
    root=Path(root);verify(root,sha)
    GLOBAL_LOCK.parent.mkdir(exist_ok=True,parents=True)
    with GLOBAL_LOCK.open('a+b') as global_lock,(root/'.lock').open('a+b') as local_lock:
        for stream in (global_lock,local_lock):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fds=(global_lock.fileno(),local_lock.fileno())
        publish(root/'attempt.json',dict(plan_sha256=sha,parent_pid=os.getpid()))
        old={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
        def interrupt(*_):raise KeyboardInterrupt('Numerical controller interrupted')
        for sig in old:signal.signal(sig,interrupt)
        prefix={}
        try:
            for i in range(9):
                case=root/f'{i:02d}';case.mkdir(exist_ok=False)
                for phase in PHASES:
                    verify(root,sha)
                    recheck_prefix(prefix)
                    argv=[sys.executable,str(Path(__file__).resolve()),'worker','--root',str(root),
                        '--plan-sha256',sha,'--index',str(i),'--phase',phase,
                        '--global-fd',str(fds[0]),'--local-fd',str(fds[1])]
                    run_process(argv,ROOT,case/phase,fds,required_free_bytes=(8+2*(9-i))*GiB,
                        output_limit_bytes=2*GiB,timeout_seconds=600,watch_root=case)
                    assert json.loads((case/phase/'guard.json').read_text())['status']=='finished'
                    assert (case/phase/'receipt.json').is_file()
                    recheck_prefix(prefix)
                    prefix.update({str(p):file_digest(p) for p in case.rglob('*') if p.is_file()})
                    print('accepted_phase',i,phase,flush=True)
            publish(root/'completed.json',dict(plan_sha256=sha,status='all27_phases_complete_pending_independent_review'))
        except BaseException as exc:
            if not (root/'STOP').exists():publish(root/'STOP',dict(reason=repr(exc)))
            publish(root/'failure.json',dict(error=repr(exc),retry=False))
            raise
        finally:
            for sig,handler in old.items():signal.signal(sig,handler)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('freeze','run','worker'))
    parser.add_argument('--root',required=True);parser.add_argument('--plan-sha256')
    parser.add_argument('--index',type=int,choices=range(9));parser.add_argument('--phase',choices=PHASES)
    parser.add_argument('--global-fd',type=int);parser.add_argument('--local-fd',type=int)
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.root)
    elif args.action=='run':run(args.root,args.plan_sha256)
    else:worker(args.root,args.plan_sha256,args.index,args.phase,(args.global_fd,args.local_fd))
