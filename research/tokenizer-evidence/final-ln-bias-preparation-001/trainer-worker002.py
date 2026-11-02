"""Two real generated-text trainer children, bounded publication and parent acceptance."""
import argparse,copy,datetime,gc,hashlib,json,os,shutil,sys,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.final_ln_bias_worker_v1 import evidence_contract as c,artifact_acceptance as a
from research.final_ln_bias_worker_v1 import validation as v
from research.local_mix_v1.output import Sink,TRAJECTORY_CAP,RAW_CAP,EXPORT_CAP
from research.final_ln_bias_worker_v1.training import run_training
from research.final_ln_bias_model_v1.runtime.train.lr_schedule import schedule_contract
from research.final_ln_bias_model_v1.runtime.train.weight_averaging import averaging_contract,export_averaged_checkpoint
from research.final_ln_bias_worker_v1.validation_data import load as validation_data
from research.gradient_clip_quality_v2.quality_execution import idle
from research.block_norm_v1.fixture_dispatch import read,reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.block_norm_v1.pilot_execution import exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot,budgets
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
B=Path(__file__).resolve().parent;ROOT=c.ROOT/'research_runs/tokenizer-final-ln-bias-trainer-prerequisite-002'
ARMS=('learned','zero')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def clear():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
class Guard(StrictMpsGuard):
    def __init__(self,path,seconds=600):
        assert 0<seconds<=600
        super().__init__(path,c.ROOT);self.limits['timeout_seconds']=seconds

def worker(args):
    assert args.slot==1, 'Continuation forbids training imported slot00'
    writer=MetadataWriter(ROOT,(args.global_fd,args.run_fd));base=f'evidence/pilots/{args.slot:02d}';root=ROOT/base
    rq=read(dict(path=str(root/'request.json'),sha256=args.request_sha));plan=read(rq['plan'])
    assert rq['parent_pid']==os.getppid() and rq['slot']==args.slot and plan['runner']==ref(__file__)
    c.verify_sources(plan['sources']);idle((os.getppid(),))
    assert ref(plan['corpus']['path'])==plan['corpus']
    with Guard(ROOT/f'control/worker-{args.slot:02d}-guard.json'):
        plain=plan['configs'][args.slot];cfg=OmegaConf.create(plain)
        from research.final_ln_bias_model_v1.integration import identity
        identity(cfg)
        writer.publish_json(base+'/config.json',plain);sink=Sink(writer,'pilots',args.slot,plain)
        try:summary=run_training(cfg,output_sink=sink)
        finally:sink.close()
        assert summary['completed_steps']==4 and summary['stop_reason']=='max_iters' and summary['device']=='mps'
        assert summary['weight_averaging']['snapshots']==3 and summary['final_metric']['tokens_processed']==4096
        clear();derived=root/'checkpoints/h200.pt'
        exported=export_averaged_checkpoint(sink.raw,derived,200,payload_writer=sink.averaged)
        writer.publish_json(base+'/export.json',exported);clear()
        files={k:ref(p) for k,p in dict(config=root/'config.json',parent=sink.raw,h200=derived,summary=root/'summary.json',metrics=root/'metrics.jsonl',export=root/'export.json').items()}
        request=dict(scope=a.SCOPE,sources=plan['sources'],config=plain,files=files)
        a.verify_request(request);binding={k:files[k] for k in ('config','parent','h200')}
        report=v.evaluate(binding,'mps');clear()
        writer.publish_json(base+'/worker.json',dict(request=ref(root/'request.json'),acceptance_request=request,validation=report,pid=os.getpid(),parent_pid=os.getppid()))
        c.verify_sources(plan['sources']);assert ref(plan['corpus']['path'])==plan['corpus']
    writer.publish_json(base+'/worker-guard.json',json.loads((ROOT/f'control/worker-{args.slot:02d}-guard.json').read_text()))
    print('bounded generated trainer completed',ARMS[args.slot],flush=True)

def parent():
    binding=read(ref(B/'trainer-worker002-source-plan.json'))
    assert binding['runner']==ref(__file__)
    c.verify_sources(binding['sources'])
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle();required=8*1024**3+TRAJECTORY_CAP+16*1024**2
        assert shutil.disk_usage(c.ROOT).free>=required
        closure=read(binding['closure']);assert closure['status']=='CLOSED_CAPACITY_PRELAUNCH_FAILURE'
        prior_process=read(closure['process']);assert prior_process['pid'] is None and prior_process['status']=='failed'
        oldplan=read(binding['original_plan']);imported=read(closure['accepted'])
        assert imported['slot']==0 and imported['arm']=='learned' and imported['acceptance']['accepted']
        assert not ROOT.exists();ROOT.mkdir()
        with session_lock(ROOT) as run_fd:
            writer=MetadataWriter(ROOT,(global_fd,run_fd));current=binding['sources']
            # Independently re-audit accepted endpoint without another optimizer update.
            with Guard(ROOT/'control/import-00-guard.json'):
                oldrequest=read(imported['worker'])['acceptance_request']
                repeated=a.accept(oldrequest,imported['acceptance']['validation']);clear()
                assert repeated==imported['acceptance']
            imported['accepted']=closure['accepted']
            imported['parent_guard']=ref(Path(closure['accepted']['path']).parent/'parent-guard.json')
            writer.publish_json('evidence/meta/imported-slot00.json',dict(accepted=closure['accepted'],audit=repeated,repeat_updates=0,copied_weight_bytes=0))
            slot=1;arm='zero';base='evidence/pilots/01';root=ROOT/base
            configs=copy.deepcopy(oldplan['configs']);cfg=configs[1];t=cfg['training']
            t.update(checkpoint_dir=str(root/'checkpoints'),metrics_file=str(root/'metrics.jsonl'),summary_file=str(root/'summary.json'))
            check=copy.deepcopy(cfg)
            for key in ('checkpoint_dir','metrics_file','summary_file'):check['training'][key]=oldplan['configs'][1]['training'][key]
            assert check==oldplan['configs'][1]
            plan=dict(oldplan,runner=ref(__file__),configs=configs,original_plan=binding['original_plan'],closure=binding['closure'],required_free=required,continuation_only_slot=1)
            assert ref(plan['corpus']['path'])==plan['corpus']
            writer.publish_json('evidence/meta/prerequisite-plan.json',plan);plan_ref=ref(ROOT/'evidence/meta/prerequisite-plan.json')
            started=time.monotonic();writer.publish_json(base+'/request.json',dict(plan=plan_ref,parent_pid=os.getpid(),slot=slot))
            argv=[sys.executable,str(Path(__file__).resolve()),'--worker','--slot','1','--global-fd',str(global_fd),'--run-fd',str(run_fd),'--request-sha',sha(root/'request.json')]
            process=run_process(writer,'pilots',slot,argv,c.ROOT,required_free_bytes=required,output_limit_bytes=TRAJECTORY_CAP,timeout_seconds=600)
            assert process['status']=='completed' and read(ref(root/'worker-guard.json'))['status']=='finished'
            with Guard(ROOT/'control/parent-01-guard.json',600-(time.monotonic()-started)):
                result=read(ref(root/'worker.json'));assert result['pid']==process['pid'] and result['parent_pid']==os.getpid()
                request=result['acceptance_request'];assert request['config']==configs[1] and request['sources']==current
                before={str(p.relative_to(root)):ref(p) for p in root.rglob('*') if p.is_file()}
                accepted=a.accept(request,result['validation']);clear()
                files={k:request['files'][k] for k in ('config','parent','h200')};cpu=v.evaluate(files,'cpu')
                assert cpu['target']==result['validation']['target'] and abs(cpu['ce']-result['validation']['ce'])<1e-5
                assert before=={str(p.relative_to(root)):ref(p) for p in root.rglob('*') if p.is_file()}
                c.verify_sources(current);assert ref(plan['corpus']['path'])==plan['corpus']
                snap=snapshot(ROOT);weight_bytes=sum(r['size'] for n,r in snap['files'].items() if n.startswith(base+'/checkpoints/'))
                metadata=sum(r['size'] for n,r in snap['files'].items() if n.startswith(base+'/') and '/checkpoints/' not in n)
                assert weight_bytes<=RAW_CAP+EXPORT_CAP and metadata<=16*1024**2 and not any(n.endswith('.tmp') for n in snap['files'])
                row=dict(arm=arm,slot=slot,binding=files,worker=ref(root/'worker.json'),process=ref(root/'process/process.json'),acceptance=accepted,cpu_mps_ce_abs=abs(cpu['ce']-result['validation']['ce']),weight_bytes=weight_bytes,observed_trial_metadata_bytes=metadata,independent_parent_validation_exact=True,all_sources_unchanged=True)
            writer.publish_json(base+'/accepted.json',row);writer.publish_json(base+'/parent-guard.json',json.loads((ROOT/'control/parent-01-guard.json').read_text()))
            row['accepted']=ref(root/'accepted.json');row['parent_guard']=ref(root/'parent-guard.json');clear();idle()
            outcome=dict(status='real_bounded_worker_parent_acceptance_passed',runner=ref(__file__),plan=plan_ref,root=str(ROOT),sources=current,arms=[imported,row],imported_slot00=closure['accepted'],imported_repeat_updates=0,copied_weight_bytes=0,generated_updates=4,total_pair_updates=8,scientific_training=False,research_corpus_read=False,holdout_read=False,free_bytes_after=shutil.disk_usage(ROOT).free,remaining='Paired trained inference then measured finite pilot preparation.')
            exclusive(B/'trainer-worker002.json',outcome)
            print(json.dumps(dict(status=outcome['status'],imported_updates=0,arms=[dict(arm=r['arm'],audit=r['acceptance']['audit']) for r in outcome['arms']])),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--slot',type=int,choices=[1]);p.add_argument('--global-fd',type=int);p.add_argument('--run-fd',type=int);p.add_argument('--request-sha');args=p.parse_args()
    try:worker(args) if args.worker else parent()
    except Exception as exc:
        if not args.worker:exclusive(B/'trainer-worker002-failure.json',dict(error=repr(exc),runner=ref(__file__),root=str(ROOT)))
        raise
