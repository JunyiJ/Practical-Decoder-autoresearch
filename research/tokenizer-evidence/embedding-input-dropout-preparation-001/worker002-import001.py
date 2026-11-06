"""New parent-only acceptance of the completed cross-root zero-input-dropout endpoint."""
import ast,copy,json
from pathlib import Path
import torch
from omegaconf import OmegaConf
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_worker_v1.execution import sources as old_sources
from research.embedding_input_dropout_worker_v2.execution import sources,checked
from research.embedding_input_dropout_worker_v2.validation import accept
from research.final_ln_scale_worker_v1.validation import validate_report
from research.embedding_input_dropout_worker_v1.data import load
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;RUN=ROOT/'research_runs/tokenizer-embedding-input-dropout-worker-v1-001';torch.set_num_threads(4)
def function(path,name):return next(n for n in ast.parse(Path(path).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
def rejected(fn):
    try:fn()
    except (ValueError,TypeError):return
    raise AssertionError('Forged scorer/acceptance metadata accepted')
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();out=E/'worker002-import001.json';guard=E/'worker002-import001-guard.json';require(not out.exists() and not guard.exists(),'Closed import proof cannot retry')
    stop=reference(RUN/'STOP');require('AttributeError' in checked(stop)['error'] and not(RUN/'evidence/pilots/01').exists(),'Original failed scope changed')
    planref=reference(E/'worker001-plan.json');plan=checked(planref);require(plan['sources']==old_sources(),'Original958sources changed')
    pins=sources();require(len(pins)==963 and all(pins[k]==v for k,v in plan['sources'].items()),'Explicit additive963-source bridge required')
    original=function(ROOT/'research/final_ln_scale_worker_v1/validation.py','score');new=function(ROOT/'research/embedding_input_dropout_worker_v2/scoring.py','score')
    require(ast.dump(ast.Module(body=original.body[2:],type_ignores=[]))==ast.dump(ast.Module(body=new.body[3:],type_ignores=[])),'Scoring numerical arithmetic changed')
    oldaccept=function(ROOT/'research/embedding_input_dropout_worker_v1/validation.py','accept');newaccept=function(ROOT/'research/embedding_input_dropout_worker_v2/validation.py','accept')
    require(ast.dump(oldaccept)==ast.dump(newaccept),'Acceptance numerical implementation changed beyond explicit imported scorer')
    base=RUN/'evidence/pilots/00';worker_ref=reference(base/'worker.json');w=checked(worker_ref);rq=checked(reference(base/'request.json'))
    require(w['request']==reference(base/'request.json') and rq['plan']==planref and rq['spec']==plan['specs'][0],'Original endpoint request/plan differs')
    worker_guard=checked(reference(RUN/'control/worker-pilots-00-guard.json'));process=checked(reference(base/'process/process.json'))
    require(worker_guard['status']=='finished' and worker_guard['reason'] is None and process['status']=='completed' and process['returncode']==0,'Original numerical worker not complete')
    cfg=OmegaConf.create(w['config']);binding={n:dict(path=w[n]['path'],sha256=w[n]['sha256']) for n in ('parent','h200')};binding['config']=reference(base/'config.json')
    with StrictMpsGuard(guard,ROOT),torch.random.fork_rng(devices=[]):
        rejected(lambda:accept(dict(w,targets=4095),cfg,binding));rejected(lambda:accept(dict(w,kind='body_allocation_worker_v1'),cfg,binding))
        result=accept(w,cfg,binding,compare_cpu=True);require(result['status']=='accepted' and result['step']==4 and result['targets']==4096 and result['ema_snapshots']==3 and result['parameter_tensors']==88,'Complete generated endpoint acceptance required')
        data=load(cfg);report=result['full_validation']
        bad=copy.deepcopy(report);bad['ce']+=.01;rejected(lambda:validate_report(bad,binding,data))
        bad=copy.deepcopy(report);bad['block_loss_sums']=bad['block_loss_sums'][:-1];rejected(lambda:validate_report(bad,binding,data))
        bad=copy.deepcopy(report);bad['target']['targets']-=1;rejected(lambda:validate_report(bad,binding,data))
        bad=copy.deepcopy(report);bad['total_loss']=float('nan');rejected(lambda:validate_report(bad,binding,data))
        bad=copy.deepcopy(report);bad['binding']['config']['sha256']='0'*64;rejected(lambda:validate_report(bad,binding,data))
        require(sources()==pins and reference(RUN/'STOP')==stop and checked(worker_ref)==w,'Original evidence/source changed')
    row=dict(slot=0,arm='input_zero',request=reference(base/'request.json'),worker=worker_ref,process=reference(base/'process/process.json'),worker_guard=reference(RUN/'control/worker-pilots-00-guard.json'),parent_guard=reference(guard),acceptance=result)
    receipt=dict(status='passed_cross_root_generated_endpoint_reaudit_and_scoring_adapter',source=reference(__file__),sources=pins,original_plan=planref,original_failed_stop=stop,
        original_failed_parent_guard=reference(RUN/'control/parent-pilots-00-guard.json'),numerical_scoring_ast_identical=True,acceptance_function_ast_identical=True,
        source_bridge='958immutable predecessors plus5new scoring/execution/dispatch sources. Original numeric training implementation reused unchanged;new scorer validates genuine input-policy dictionary,never fakes model.cfg.',
        imported_endpoint=row,new_optimizer_updates=0,weight_copies=0,seven_metadata_forgeries_rejected=True,scientific_seed=False,quality_result=False,guard=reference(guard))
    raw=(json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode();require(len(raw)<=512*1024,'Bounded import receipt')
    with out.open('xb') as f:f.write(raw)
    print(json.dumps(dict(status=receipt['status'],steps=4,ce=result['full_validation']['ce'],cpu_mps_difference=result['cpu_mps_ce_difference'],sources=len(pins),new_updates=0,weight_copies=0)))
