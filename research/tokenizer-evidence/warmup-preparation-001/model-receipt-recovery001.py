"""Recover a receipt from sealed successful numerical traces; never rerun training."""
import datetime,hashlib,json,shutil
from pathlib import Path
from research.ffn_norm_quality_v1.support import idle
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.block_norm_v1.pilot_execution import exclusive
B=Path(__file__).resolve().parent;R=B.parents[2]
ROOT=R/'research_runs/tokenizer-warmup-model-prerequisite-001'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(Path(p).relative_to(R)),sha256=sha(p))
def read(p):return json.loads(Path(p).read_text())
with session_lock(GLOBAL_LOCK.parent):
    idle();plan=read(B/'model-receipt-recovery001-plan.json')
    for n,h in plan['sources'].items():assert sha(R/n)==h
    failed=read(B/'model-resume001-failure.json')
    assert failed['status']=='closed_receipt_publication_failure' and failed['exit_code']==1
    for k in ('source','plan','trace','guard'):assert sha(R/failed[k]['path'])==failed[k]['sha256']
    original=read(B/'model-resume001-plan.json')
    with CpuPrerequisiteGuard(B/'model-receipt-recovery001-guard.json',R,timeout_seconds=120):
        for n,h in original['sources'].items():assert sha(R/n)==h
        guard=read(B/'model-resume001-guard.json');assert guard['status']=='finished' and guard['reason'] is None
        trace=[json.loads(line) for line in (B/'model-resume001-trace.jsonl').read_text().splitlines()]
        assert [x['event'] for x in trace]==['begin_arm','saved','saved','fresh_repeat_passed','resume_passed','arm_completed']*2
        rows=[];configs={};total=0
        for i,arm in enumerate(('none','warmup100')):
            events=trace[6*i:6*i+6];row=events[-1];assert row['arm']==arm
            assert row['repeat_max_abs']==row['resume_max_abs']==0
            assert row['scientific_full_schedule_rates_passed'] and row['restored_state_and_rng_exact']
            assert all(x['reset_kernel']['passed'] for x in row['inference']) and [x['batch'] for x in row['inference']]==[1,8]
            assert set(row['rejections'])=={'other_arm_training_contract','missing_optimizer','group_order','missing_mps_rng','schedule_contract','scheduled_rate','h200_resume'}
            directory=ROOT/arm
            assert {p.name for p in directory.iterdir()}==set(row['files'])=={'config.json','step1.pt','final.pt','h200.pt'}
            for n,v in row['files'].items():
                p=directory/n;assert not p.is_symlink() and p.stat().st_size==v['size'] and sha(p)==v['sha256'];total+=v['size']
            for event,name in zip(events[1:3],('step1.pt','final.pt')):
                assert Path(event['path'])==directory/name and event['sha256']==row['files'][name]['sha256']
            assert row['export']['parent_checkpoint']==dict(path=str(directory/'final.pt'),sha256=row['files']['final.pt']['sha256'])
            assert row['export']['checkpoint_sha256']==row['files']['h200.pt']['sha256']
            assert row['export']['averaged_weights']['snapshots']==3 and row['export']['averaged_weights']['completed_steps']==4
            configs[arm]=read(directory/'config.json');rows.append({k:v for k,v in row.items() if k!='event'})
        assert rows[0]['initial_state_sha256']==rows[1]['initial_state_sha256']
        assert rows[0]['optimizer_groups']==rows[1]['optimizer_groups']
        assert configs['none']['model']==configs['warmup100']['model']
        assert configs['none']['training']['max_iters']==configs['warmup100']['training']['max_iters']==4
        assert configs['warmup100']['training']['lr_schedule']['warmup_steps']==2
        assert configs['none']['training']['lr_schedule']['kind']=='linear_cooldown'
        for n,h in plan['sources'].items():assert sha(R/n)==h
    assert read(B/'model-receipt-recovery001-guard.json')['status']=='finished'
    result=dict(status='recovered_full_model_rates_short_repeat_resume_h200_passed',at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        original_attempt=ref(B/'model-resume001-failure.json'),original_plan=ref(B/'model-resume001-plan.json'),trace=ref(B/'model-resume001-trace.jsonl'),
        numerical_guard=ref(B/'model-resume001-guard.json'),recovery_plan=ref(B/'model-receipt-recovery001-plan.json'),recovery_guard=ref(B/'model-receipt-recovery001-guard.json'),
        configs=configs,arms=rows,root=str(ROOT),generated_corpus=ref(ROOT/'generated.txt'),verified_output_bytes=total,free_bytes=shutil.disk_usage(R).free,
        scope=original['scope'],numerical_rerun=False,scientific_training_launched=False,holdout_read=False,
        limitation='Actual full-model replay uses four updates and warmup2; all1200 warmup100 group rates were checked without scientific training. This establishes integration, not quality.',
        remaining='Actual bounded run_training worker and independent parent full validation/inference. Then fresh seed census, complete measured capacity, finite pilots and final quality freeze.')
    exclusive(B/'model-receipt-recovery001.json',result)
    exclusive(ROOT/'STOP',dict(status='closed_prerequisite',original_failure=ref(B/'model-resume001-failure.json'),recovery=ref(B/'model-receipt-recovery001.json'),no_retry=True))
    print(json.dumps({k:result[k] for k in ('status','verified_output_bytes','free_bytes','numerical_rerun')}))
