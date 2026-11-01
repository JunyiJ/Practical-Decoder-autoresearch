"""Actual generated-text trainer in serial children with independent parent acceptance."""
import argparse, copy, gc, hashlib, json, os, shutil, sys, time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.ema_horizon_v1 import contract as c, execution as e
from research.block_norm_v1.fixture_dispatch import reference, read
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.ffn_structure_v1 import validation as v
from research.ffn_structure_v1.runtime.train.train import run_training, _make_loader, _training_contract
from research.ffn_structure_v1.runtime.train.weight_averaging import export_averaged_checkpoint
from research.ffn_structure_v1.runtime.utils.checkpoint import read_checkpoint, load_model_weights
from research.ffn_structure_v1.runtime.models.gpt import GPT
from src.research.tokenizer_activation_entry import GLOBAL_LOCK, session_lock
from src.research.tokenizer_serial_process import run_process
B=Path(__file__).resolve().parent;ROOT=c.ROOT/'research_runs/tokenizer-ffn-structure-trainer-prerequisite-001'
CAP=800*1024**2

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def sources():return {str(p.relative_to(c.ROOT)):sha(p) for p in (c.ROOT/'research/ffn_structure_v1').rglob('*.py')}
def clear():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
class CappedWriter:
    def __init__(self,f,limit):self.f,self.limit=f,limit
    def tell(self):return self.f.tell()
    def flush(self):return self.f.flush()
    def write(self,data):
        assert self.tell()+memoryview(data).nbytes<=self.limit
        return self.f.write(data)
class Sink:
    def __init__(self,root,cfg):self.root,self.cfg,self.raw,self.count=root,cfg,None,0
    def validate_config(self,cfg):assert OmegaConf.to_container(cfg,resolve=True)==self.cfg
    def checkpoint(self,path,payload):
        path=Path(path);assert path.parent==self.root and path.name.startswith('final_') and path.suffix=='.pt' and self.raw is None and payload['step']==2
        with path.open('xb') as f:torch.save(payload,CappedWriter(f,600*1024**2))
        self.raw=path
    def metric(self,path,value):
        assert Path(path)==self.root/'metrics.jsonl' and value['step']==self.count
        line=(json.dumps(value)+'\n').encode();assert len(line)<1024**2
        with Path(path).open('xb' if not self.count else 'ab') as f:f.write(line)
        self.count+=1
    def summary(self,path,value):assert Path(path)==self.root/'summary.json';e.exclusive(Path(path),value)

def worker(args):
    root=ROOT/args.arm;rq=read(dict(path=str(root/'request.json'),sha256=args.request_sha))
    assert rq['parent_pid']==os.getppid() and rq['arm']==args.arm and rq['source']==ref(__file__)
    assert rq['sources']==sources() and rq['frozen_sources']==c.source_inventory()
    for fd,path in [(args.global_fd,GLOBAL_LOCK),(args.run_fd,ROOT/'.lock')]:
        a=os.fstat(fd);b=path.stat();assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
    e.idle(allowed=(os.getppid(),))
    with StrictMpsGuard(root/'worker-guard.json',c.ROOT):
        cfg=OmegaConf.create(rq['config']);sink=Sink(root,rq['config']);e.exclusive(root/'config.json',rq['config'])
        summary=run_training(cfg,output_sink=sink)
        assert summary['device']=='mps' and summary['completed_steps']==2 and summary['stop_reason']=='max_iters'
        assert summary['weight_averaging']['snapshots']==2 and summary['final_metric']['tokens_processed']==2048
        assert sink.count==2 and sink.raw is not None;clear()
        derived=root/'h200.pt'
        def writer(path,payload):
            assert Path(path)==derived
            with derived.open('xb') as f:torch.save(payload,CappedWriter(f,128*1024**2))
        export=export_averaged_checkpoint(sink.raw,derived,200,payload_writer=writer);clear()
        binding=dict(config=ref(root/'config.json'),parent=ref(sink.raw),h200=ref(derived))
        report=v.evaluate(binding,'mps');e.exclusive(root/'validation.json',report)
        e.exclusive(root/'result.json',dict(binding=binding,summary=summary,export=export,validation=ref(root/'validation.json')))
        assert rq['sources']==sources() and rq['frozen_sources']==c.source_inventory()
    print('generated trainer and validation completed',args.arm,flush=True)

def parent():
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        e.idle();assert not ROOT.exists();ROOT.mkdir()
        prior=read(reference(B/'checkpoint-resume001.json'));assert prior['status']=='serialized_mps_repeat_resume_h200_passed'
        original=c.source_inventory();current=sources()
        generated=ROOT/'generated.txt';generated.write_bytes(Path(prior['generated_corpus']['path']).read_bytes());assert sha(generated)==prior['generated_corpus']['sha256']
        rows=[]
        with session_lock(ROOT) as run_fd:
            for arm in ('swiglu','ungated_silu'):
                root=ROOT/arm;root.mkdir();cfg=copy.deepcopy(prior['configs'][arm]);cfg['data']['path']=str(generated)
                t=cfg['training'];t.update(seed=377201,eval_seed=377203,max_iters=2,checkpoint_dir=str(root),metrics_file=str(root/'metrics.jsonl'),summary_file=str(root/'summary.json'),eval_iters=2,eval_every=2)
                t['lr_schedule'].update(total_steps=2,start_step=1);t['weight_averaging'].update(total_steps=2,start_step=1)
                rq=dict(arm=arm,parent_pid=os.getpid(),config=cfg,sources=current,frozen_sources=original,source=ref(__file__))
                e.exclusive(root/'request.json',rq)
                argv=[sys.executable,str(Path(__file__).resolve()),'--worker','--arm',arm,'--global-fd',str(global_fd),'--run-fd',str(run_fd),'--request-sha',sha(root/'request.json')]
                process=run_process(argv,c.ROOT,root/'process',(global_fd,run_fd),required_free_bytes=8*1024**3+2*CAP,
                    output_limit_bytes=CAP,timeout_seconds=600,watch_root=root)
                assert process['status']=='completed' and read(reference(root/'worker-guard.json'))['status']=='finished'
                with StrictMpsGuard(root/'parent-guard.json',c.ROOT):
                    result=read(reference(root/'result.json'));binding=result['binding'];reported=read(result['validation'])
                    before={p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in root.iterdir() if p.is_file() and p.name!='parent-guard.json'}
                    raw=read_checkpoint(binding['parent']['path']);avg=read_checkpoint(binding['h200']['path']);data=_make_loader(OmegaConf.create(cfg),'cpu')
                    assert set(data.splits)=={'train','val'} and raw['training_state']['contract']==_training_contract(OmegaConf.create(cfg),'mps',data)
                    assert raw['step']==2 and raw['training_state']['weight_averaging_state']['snapshots']==2
                    assert raw['training_state']['metrics']==result['summary']['metrics']
                    assert [json.loads(line) for line in (root/'metrics.jsonl').read_text().splitlines()]==result['summary']['metrics']
                    model=GPT(OmegaConf.create(cfg).model).to('mps');load_model_weights(model,raw,for_resume=True)
                    averages=raw['training_state']['weight_averaging_state']['averages']['200']
                    for n,value in avg['model_state'].items():assert torch.equal(value,averages[n] if n in averages else raw['model_state'][n])
                    del raw,avg,model,averages;clear()
                    independently=v.evaluate(binding,'mps');assert independently==reported
                    cpu_report=v.evaluate(binding,'cpu')
                    assert cpu_report['target']==reported['target'] and abs(cpu_report['ce']-reported['ce'])<1e-5
                    assert len(reported['window_loss_sums'])==len(reported['target']['window_counts'])
                    bad=copy.deepcopy(reported);bad['target']['targets']+=1
                    try:v.validate_report(bad,binding,data)
                    except ValueError:pass
                    else:raise AssertionError('Incorrect target coverage accepted')
                    assert before=={p.name:dict(size=p.stat().st_size,sha256=sha(p)) for p in root.iterdir() if p.is_file() and p.name!='parent-guard.json'}
                    assert current==sources() and original==c.source_inventory()
                    row=dict(arm=arm,child_pid=process['pid'],process=ref(root/'process/process.json'),binding=binding,worker_result=ref(root/'result.json'),
                        independent_parent_mps_report_exact=True,cpu_mps_ce_abs=abs(cpu_report['ce']-reported['ce']),validation_targets=reported['target']['targets'],
                        validation_windows=len(reported['window_loss_sums']),tail_targets=reported['target']['window_counts'][-1],raw_resume_and_exact_export=True,
                        generated_updates=2,generated_targets_processed=2048,inventory=before)
                row['parent_guard']=ref(root/'parent-guard.json');rows.append(row);clear();e.idle()
        outcome=dict(status='actual_serial_trainer_and_parent_acceptance_passed',runner=ref(__file__),root=str(ROOT),sources=current,frozen_sources=original,arms=rows,
            checkpoint_prerequisite=reference(B/'checkpoint-resume001.json'),free_bytes_after=shutil.disk_usage(c.ROOT).free,
            scientific_training_launched=False,research_corpus_read=False,holdout_read=False,
            scope='Two generated-text updates per arm in separate serial children; bounded actual trainer output sinks, exact H200 export, raw resume preflight, full-validation target accounting and independent parent MPS/CPU scoring.',
            remaining='Fresh trained-pair timing/acceptance integration, source/seed census, actual high-water audit and finite resource-pilot/quality freeze.')
        e.exclusive(B/'trainer-worker001.json',outcome);print(json.dumps(dict(status=outcome['status'],arms=[dict(arm=r['arm'],cpu_mps_ce_abs=r['cpu_mps_ce_abs'],validation_targets=r['validation_targets']) for r in rows])),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--arm',choices=['swiglu','ungated_silu']);p.add_argument('--global-fd',type=int);p.add_argument('--run-fd',type=int);p.add_argument('--request-sha');args=p.parse_args()
    try:worker(args) if args.worker else parent()
    except Exception as exc:
        if not args.worker:e.exclusive(B/'trainer-worker001-failure.json',dict(error=repr(exc),runner=ref(__file__),root=str(ROOT)))
        raise
