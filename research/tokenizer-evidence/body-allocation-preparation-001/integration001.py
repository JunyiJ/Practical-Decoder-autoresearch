"""Read-only actual worker transport/finite pilot and quality config integration."""
import copy,gc,json,shutil
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,config,plain,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_model_v1.model import GPT
from research.body_allocation_model_v1.optimizer import build_optimizer
from research.body_allocation_model_v1.checkpoint import validate_resume
from research.block_norm_v1.runtime.train.weight_averaging import TailWeightAverages,averaging_contract
from research.block_norm_v1.runtime.train.lr_schedule import schedule_contract
from research.body_allocation_worker_v1.execution import sources,checked
from research.body_allocation_worker_v1.validation import verify_artifact
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from research.final_ln_scale_quality_v3.support import idle
B=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-body-allocation-worker-v1-001';OUT=B/'integration001.json'
torch.set_num_threads(4)
def rejected(fn):
    try:fn()
    except (ValueError,KeyError,TypeError):return
    raise AssertionError('Corrupt integration accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not OUT.exists(),'New integration attempt required')
    with CpuPrerequisiteGuard(B/'integration001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25),torch.random.fork_rng(devices=[]):
        plan=checked(reference(B/'worker001-plan.json'));require(plan['sources']==sources(),'Frozen worker source changed')
        rows=[]
        for slot,arm in enumerate(('incumbent','shallow','deeper')):
            base=RUN/f'evidence/pilots/{slot:02d}';accepted=checked(reference(base/'accepted.json'));worker=checked(accepted['worker'])
            require(accepted['acceptance']['status']=='accepted' and accepted['acceptance']['step']==4,'Actual four-update acceptance required')
            require(accepted['acceptance']['cpu_mps_ce_difference']<3e-5,'CPU/MPS validation agreement required')
            cfg=OmegaConf.create(plan['specs'][slot]['config']);m=GPT(cfg.model);o=build_optimizer(m,cfg);e=TailWeightAverages(m,cfg.training)
            raw=torch.load(worker['parent']['path'],map_location='cpu',weights_only=True);require(validate_resume(m,o,e,raw,cfg)==4,'Raw resume integration failed')
            variants=[dict(raw,kind='h200_body_allocation_v1'),dict(raw,architecture={}),dict(raw,clipping_history=[]),dict(raw,training_identity={}),dict(raw,sampler_state={})]
            bad=copy.copy(raw);em=copy.deepcopy(raw['ema']);em['parameter_names']=em['parameter_names'][:-1];bad['ema']=em;variants.append(bad)
            for bad in variants:rejected(lambda:validate_resume(m,o,e,bad,cfg))
            del variants,bad,em,raw,m,o,e;gc.collect()
            finite=[]
            for steps in (20,1200):
                for lr in (.000625,.00125,.0025):
                    target=config(arm,steps,91261+slot,lr);spec=averaging_contract(target.training);sched=schedule_contract(target.training)
                    require(plain(target.model)==plain(cfg.model),'Pilot/science geometry changed')
                    require(spec['start_step']==(13 if steps==20 else 800) and spec['interval']==(1 if steps==20 else 10)
                        and spec['total_steps']==steps and sched['total_steps']==steps,'Finite tail/cooldown contract differs')
                    finite.append(dict(steps=steps,lr=lr,ema_snapshots=8 if steps==20 else 41))
            mirror=B/f'worker001-{arm}-evidence001';require(not mirror.exists(),'Evidence mirror already exists');mirror.mkdir()
            mirrored=[]
            for p in sorted(base.rglob('*')):
                if p.is_file() and 'checkpoints' not in p.parts and not p.name.startswith('.'):
                    to=mirror/p.relative_to(base);to.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(p,to);require(reference(p)['sha256']==reference(to)['sha256'],'Evidence mirror differs');mirrored.append(reference(to))
            rows.append(dict(arm=arm,accepted=reference(base/'accepted.json'),worker=accepted['worker'],binding=accepted['acceptance']['binding'],
                generated_weights=[worker['parent'],worker['h200']],read_only_raw_reaudit=True,forgeries_rejected=6,finite_configs=finite,mirrored_metadata=mirrored,weight_copies=0))
        result=dict(status='generated_model_and_worker_dependencies_closed',sources=sources(),rows=rows,
            eligible_payloads='Only the nine resume001 and six worker001 generated numerical prerequisite payloads, after new exact audit. No pilot, screen, selected or fresh scientific weights.',
            closure='All three real-data workers independently accepted optimizer/EMA/RNG/samplers/clipping/RAW/H200/full-validation/cache. Actual raw transport and finite20/1200 configs rechecked; future workers initialize fresh, bind metadata and never load prerequisite payloads.',
            scientific_quality=False,holdout_scored=False,retirement_executed=False)
        with OUT.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],arms=3,forgeries_rejected=18,finite_configs=18,weight_copies=0)))
