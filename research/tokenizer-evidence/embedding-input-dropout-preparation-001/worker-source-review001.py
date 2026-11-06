"""New source/route and real mask-only RNG oracle; zero optimizer updates/outputs."""
import ast,gc,json
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require,plain
from research.body_allocation_model_v1.evidence import reference,environment
from research.embedding_input_dropout_resume_v1.contract import config
from research.embedding_input_dropout_v1.model import GPT
from research.embedding_input_dropout_worker_v1.execution import sources
from research.embedding_input_dropout_worker_v1.validation import replay_dropout_rng
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;torch.set_num_threads(4)
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();require(not(E/'worker-source-review001.json').exists() and not(E/'worker-source-review001-guard.json').exists(),'Closed source/route proof cannot retry');pins=sources()
    builder=E/'worker001-build-run001.py';tree=ast.parse(builder.read_text());calls=[n.func.id for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
    require('main' in calls and 'launch' in calls,'Finite builder launch route required')
    text=(ROOT/'research/embedding_input_dropout_worker_v1/dispatch.py').read_text();require("'research.embedding_input_dropout_worker_v1.execution'" in text,'New actual worker module route required')
    for prob in (0.,.1):
        for steps in (4,20,1200):
            for lr in (.000625,.00125,.0025):
                c=config(prob,steps=steps,lr=lr);require(c.model.dim==256 and c.model.n_layers==4 and c.training.max_iters==steps and c.input_embedding_regularization.probability==prob,'Finite config route differs')
    with StrictMpsGuard(E/'worker-source-review001-guard.json',ROOT),torch.random.fork_rng(devices=[]):
        rows=[]
        for prob in (0.,.1):
            cfg=config(prob);torch.manual_seed(600408);torch.mps.manual_seed(600408);m=GPT(plain(cfg.input_embedding_regularization)).to('mps');initial=torch.mps.get_rng_state().clone();cpu=torch.get_rng_state().clone()
            with torch.no_grad():
                x=(torch.arange(1024,device='mps').reshape(8,128)%400+1).long();outputs=m(x)[0];require(bool(torch.isfinite(outputs).all()),'Finite full training-layout forward required');del outputs
                actual=torch.mps.get_rng_state().clone();expected=replay_dropout_rng(cfg,initial,1)
                require(torch.equal(actual,expected) and torch.equal(torch.mps.get_rng_state(),actual) and torch.equal(cpu,torch.get_rng_state()),'Parent mask-only RNG oracle or transaction differs')
                m.eval();outputs=m(x)[0];require(torch.equal(torch.mps.get_rng_state(),actual),'Eval advanced mask RNG');del outputs
            rows.append(dict(probability=prob,full_batch8_context128_forward_rng_equals_parent_mask_replay=True,zero_optimizer_updates=True,eval_rng_unchanged=True));del m,x;gc.collect();torch.mps.empty_cache()
        require(sources()==pins,'New worker source changed')
    result=dict(status='passed_finite_worker_source_routes_and_actual_mask_rng_oracle',source=reference(__file__),sources=pins,prepared_builder=reference(builder),rows=rows,
        finite_configurations_checked=18,worker_process_launched=False,weight_files_written=0,new_scientific_seed=False,guard=reference(E/'worker-source-review001-guard.json'))
    with (E/'worker-source-review001.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','source','prepared_builder','guard')}))
