"""Two real generated-text trainer children, bounded publication and parent acceptance."""
import argparse,copy,datetime,gc,hashlib,json,os,shutil,sys,time
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.final_ln_scale_worker_v1 import evidence_contract as c,artifact_acceptance as a
from research.final_ln_scale_worker_v1 import validation as v
from research.local_mix_v1.output import Sink,TRAJECTORY_CAP,RAW_CAP,EXPORT_CAP
from research.final_ln_scale_worker_v1.training import run_training
from research.final_ln_scale_model_v1.runtime.train.lr_schedule import schedule_contract
from research.final_ln_scale_model_v1.runtime.train.weight_averaging import averaging_contract,export_averaged_checkpoint
from research.final_ln_scale_worker_v1.validation_data import load as validation_data
from research.gradient_clip_quality_v2.quality_execution import idle
from research.block_norm_v1.fixture_dispatch import read,reference
from research.block_norm_v1.mps_guard import StrictMpsGuard
from research.block_norm_v1.pilot_execution import exclusive
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.tokenizer_wide_hybrid_metadata_writer import MetadataWriter
from src.research.tokenizer_wide_hybrid_metadata import snapshot,budgets
from src.research.tokenizer_wide_hybrid_process_v2 import run_process
B=Path(__file__).resolve().parent;ROOT=c.ROOT/'research_runs/tokenizer-final-ln-scale-trainer-prerequisite-001'
ARMS=('learned','unit')

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(Path(p).resolve()),sha256=sha(p))
def clear():gc.collect();torch.mps.synchronize();torch.mps.empty_cache()
class Guard(StrictMpsGuard):
    def __init__(self,path,seconds=600):
        assert 0<seconds<=600
        super().__init__(path,c.ROOT);self.limits['timeout_seconds']=seconds

def worker(args):
    writer=MetadataWriter(ROOT,(args.global_fd,args.run_fd));base=f'evidence/pilots/{args.slot:02d}';root=ROOT/base
    rq=read(dict(path=str(root/'request.json'),sha256=args.request_sha));plan=read(rq['plan'])
    assert rq['parent_pid']==os.getppid() and rq['slot']==args.slot and plan['runner']==ref(__file__)
    c.verify_sources(plan['sources']);idle((os.getppid(),))
    assert ref(plan['corpus']['path'])==plan['corpus']
    with Guard(ROOT/f'control/worker-{args.slot:02d}-guard.json'):
        plain=plan['configs'][args.slot];cfg=OmegaConf.create(plain)
        from research.final_ln_scale_model_v1.integration import identity
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
    binding=read(ref(B/'trainer-worker001-source-plan.json'))
    assert binding['runner']==ref(__file__) and binding['prerequisite']==ref(B/'model-resume003.json')
    c.verify_sources(binding['sources'])
    with session_lock(GLOBAL_LOCK.parent) as global_fd:
        idle();assert shutil.disk_usage(c.ROOT).free>=8*1024**3+2*TRAJECTORY_CAP+16*1024**2
        assert not ROOT.exists();ROOT.mkdir()
        with session_lock(ROOT) as run_fd:
            writer=MetadataWriter(ROOT,(global_fd,run_fd))
            prior=read(ref(B/'model-resume003.json'));assert prior['status']=='final_ln_scale_both_arms_serialized_resume_final_h200_passed'
            # New final-horizon workers; never load the successful prerequisite tensors.
            corpus_path=B/'worker001-generated.txt'
            assert not corpus_path.exists()
            corpus_path.write_text(''.join(f'Worker example {i}: amber birds cross a quiet river. Count {i%17}.\n' for i in range(800)))
            corpus=ref(corpus_path)
            current=c.source_inventory();assert current==binding['sources'];configs=[]
            from research.ffn_structure_v1 import pilot_contract as pilot
            from research.final_ln_scale_model_v1.runtime.models.block_norm import READOUT_POLICY
            from research.final_ln_scale_model_v1.integration import identity
            from research.final_ln_scale_worker_v1.training import _make_loader
            from research.final_ln_scale_model_v1.runtime.models.output_bias import tokenizer_no_bias_prior
            base=OmegaConf.create(pilot.config(0,dict(pilot=92711,screen=92712,fresh=[92721,92722,92723],evaluation=92713,bootstrap=92714)))
            base.model.tokenizer_architecture_policy=READOUT_POLICY
            base.model.residual_gain_width_policy='mla4_fixed_complete_branch_gain_v1'
            base.model.residual_gain_kind='unit'
            base.model.readout_norm_kind='layer_affine';base.model.readout_norm_width_policy='mla4_final_ln_scale_v1';base.model.readout_norm_epsilon=1e-5
            base.model.final_ln_bias_policy='mla4_final_ln_bias_v1';base.model.final_ln_bias_kind='learned'
            base.model.final_ln_scale_policy='mla4_final_ln_scale_v1';base.model.final_ln_scale_kind='learned'
            base.training.adamw_epsilon_policy='mla4_adamw_epsilon_fixed_v1';base.training.adamw_epsilon=1e-6
            base.data.path=str(corpus_path);base.data.train_fraction=.8;base.data.val_fraction=.2
            base.training.max_iters=4;base.training.learning_rate=.00125
            base.training.adamw_beta2_policy='mla4_adamw_beta2_fixed_v1';base.training.adamw_beta2=.999
            base.training.weight_averaging=dict(kind='spaced_tail_ema',total_steps=4,start_step=2,interval=1,half_lives=[200])
            base.training.lr_schedule=dict(kind='linear_cooldown',total_steps=4,start_step=2,min_ratio=.1)
            loader=_make_loader(base,'cpu');base.model.output_bias_prior=tokenizer_no_bias_prior(loader,str(corpus_path));del loader
            for slot,arm in enumerate(ARMS):
                cfg=OmegaConf.to_container(base,resolve=True);cfg['training']['weight_decay']=.00006
                root=ROOT/f'evidence/pilots/{slot:02d}';t=cfg['training']
                t.update(seed=92711,eval_seed=92713,max_iters=4,checkpoint_dir=str(root/'checkpoints'),metrics_file=str(root/'metrics.jsonl'),summary_file=str(root/'summary.json'),eval_iters=2,eval_every=2)
                t['grad_clip']=1.;t['adamw_epsilon']=1e-6;cfg['model']['final_ln_scale_kind']=dict(learned='learned',unit='fixed_unit')[arm]
                t['lr_schedule'].update(total_steps=4,start_step=2);t['weight_averaging'].update(total_steps=4,start_step=2)
                identity(OmegaConf.create(cfg));schedule_contract(OmegaConf.create(cfg).training);averaging_contract(OmegaConf.create(cfg).training);configs.append(cfg)
            plan=dict(kind='generated_final_ln_scale_worker_parent_prerequisite',runner=ref(__file__),sources=current,configs=configs,corpus=corpus,
                raw_cap=RAW_CAP,export_cap=EXPORT_CAP,trajectory_cap=TRAJECTORY_CAP,round_metadata=16*1024**2,
                required_free=8*1024**3+2*TRAJECTORY_CAP+16*1024**2,scientific_training=False)
            assert shutil.disk_usage(ROOT).free>=plan['required_free']
            writer.publish_json('evidence/meta/prerequisite-plan.json',plan);plan_ref=ref(ROOT/'evidence/meta/prerequisite-plan.json')
            rows=[]
            for slot,arm in enumerate(ARMS):
                started=time.monotonic();base=f'evidence/pilots/{slot:02d}';root=ROOT/base
                writer.publish_json(base+'/request.json',dict(plan=plan_ref,parent_pid=os.getpid(),slot=slot))
                argv=[sys.executable,str(Path(__file__).resolve()),'--worker','--slot',str(slot),'--global-fd',str(global_fd),'--run-fd',str(run_fd),'--request-sha',sha(root/'request.json')]
                process=run_process(writer,'pilots',slot,argv,c.ROOT,required_free_bytes=8*1024**3+(2-slot)*TRAJECTORY_CAP+16*1024**2,
                    output_limit_bytes=TRAJECTORY_CAP,timeout_seconds=600)
                assert process['status']=='completed' and read(ref(root/'worker-guard.json'))['status']=='finished'
                with Guard(ROOT/f'control/parent-{slot:02d}-guard.json',600-(time.monotonic()-started)):
                    result=read(ref(root/'worker.json'));assert result['pid']==process['pid'] and result['parent_pid']==os.getpid()
                    request=result['acceptance_request'];assert request['config']==configs[slot] and request['sources']==current
                    before={str(p.relative_to(root)):ref(p) for p in root.rglob('*') if p.is_file()}
                    accepted=a.accept(request,result['validation']);clear()
                    binding={k:request['files'][k] for k in ('config','parent','h200')};cpu_report=v.evaluate(binding,'cpu')
                    assert cpu_report['target']==result['validation']['target'] and abs(cpu_report['ce']-result['validation']['ce'])<1e-5
                    data=validation_data(OmegaConf.create(configs[slot]));bad=copy.deepcopy(result['validation']);bad['target']['targets']+=1
                    try:v.validate_report(bad,binding,data)
                    except ValueError:pass
                    else:raise AssertionError('Incorrect validation coverage accepted')
                    assert before=={str(p.relative_to(root)):ref(p) for p in root.rglob('*') if p.is_file()}
                    c.verify_sources(current);assert ref(corpus['path'])==corpus
                    snap=snapshot(ROOT);weight_bytes=sum(r['size'] for n,r in snap['files'].items() if n.startswith(base+'/checkpoints/'))
                    metadata=sum(r['size'] for n,r in snap['files'].items() if n.startswith(base+'/') and '/checkpoints/' not in n)
                    assert weight_bytes<=RAW_CAP+EXPORT_CAP and metadata<=16*1024**2
                    assert not any(n.endswith('.tmp') for n in snap['files'])
                    row=dict(arm=arm,slot=slot,binding=binding,worker=ref(root/'worker.json'),process=ref(root/'process/process.json'),
                        acceptance=accepted,cpu_mps_ce_abs=abs(cpu_report['ce']-result['validation']['ce']),weight_bytes=weight_bytes,observed_trial_metadata_bytes=metadata,
                        publication='Bounded CheckpointWriter streaming temporary and same-inode hardlink/unlink; no duplicate whole-file payload',
                        independent_parent_validation_exact=True,all_sources_unchanged=True)
                writer.publish_json(base+'/accepted.json',row);writer.publish_json(base+'/parent-guard.json',json.loads((ROOT/f'control/parent-{slot:02d}-guard.json').read_text()))
                row['accepted']=ref(root/'accepted.json');row['parent_guard']=ref(root/'parent-guard.json');rows.append(row);clear();idle()
            outcome=dict(status='real_bounded_worker_parent_acceptance_passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),runner=ref(__file__),plan=plan_ref,
                root=str(ROOT),sources=current,arms=rows,metadata_limits=budgets(),observed_output=snapshot(ROOT),free_bytes_after=shutil.disk_usage(ROOT).free,
                generated_updates=8,scientific_training=False,research_corpus_read=False,holdout_read=False,
                remaining='Trained paired inference, prospectively tested retention boundary, actual full high-water and source/seed census before finite pilot/quality freeze.')
            exclusive(B/'trainer-worker001.json',outcome)
            print(json.dumps(dict(status=outcome['status'],arms=[dict(arm=r['arm'],audit=r['acceptance']['audit'],cpu_mps_ce_abs=r['cpu_mps_ce_abs']) for r in rows])),flush=True)

if __name__=='__main__':
    torch.set_num_threads(4)
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--slot',type=int,choices=[0,1]);p.add_argument('--global-fd',type=int);p.add_argument('--run-fd',type=int);p.add_argument('--request-sha');args=p.parse_args()
    try:worker(args) if args.worker else parent()
    except Exception as exc:
        if not args.worker:exclusive(B/'trainer-worker001-failure.json',dict(error=repr(exc),runner=ref(__file__),root=str(ROOT)))
        raise
