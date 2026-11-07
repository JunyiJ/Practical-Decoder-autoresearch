"""Read-only completeness/source/resource review,without replaying generated steps."""
import itertools,json
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_worker_v1.execution import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;R=E/'optimizer-response001'
def checked(r):
    p=Path(r['path']);require(reference(p)=={k:r[k] for k in ('path','sha256')},'Immutable source/evidence changed');return json.loads(p.read_text())
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not(E/'optimizer-response-review001.json').exists() and not(E/'optimizer-response-review001-guard.json').exists(),'Fresh terminal review only;never retry')
    with CpuPrerequisiteGuard(E/'optimizer-response-review001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        terminal=reference(R/'STOP');stop=checked(terminal);result=checked(stop['result']);plan=checked(result['source_plan']);guard=checked(reference(R/'guard.json'))
        require(stop['status']=='complete_generated_optimizer_response_proof' and stop['never_rerun'] and result['status']=='generated_fixed_state_clipping_muon_adam_response_independent_oracle_passed','Complete original generated proof required')
        require(guard['status']=='finished' and guard['reason'] is None and guard['limits']['timeout_seconds']==120 and guard['limits']['rss_limit_bytes']==4*1024**3 and guard['limits']['reserve_bytes']==8*1024**3,'Original complete resource guard required')
        wanted=set(itertools.product([tuple(x) for x in plan['case_shapes']],plan['case_updates'],plan['case_kinds'],plan['case_scales']))
        actual=[(tuple(c['shape']),c['updates'],c['kind'],c['scale']) for c in result['cases']]
        require(len(actual)==len(set(actual))==len(wanted)==54 and set(actual)==wanted,'Exact complete generated Cartesian product required')
        require(result['none_gradient_cases']==2 and result['rejections_before_mutation']==23 and result['linear_reference_error']==0 and result['global_cpu_rng_unchanged'],'Actual missing-gradient/adversarial/linear/RNG tests required')
        for c in result['cases']:
            require(0<c['clip_coefficient']<=1 and c['parameter_maxerror']['matrix']<=plan['tolerances']['muon_parameter_atol'] and c['parameter_maxerror']['vector']<=plan['tolerances']['adam_parameter_atol'],'Generated case outside original tolerance')
        require(sum(c['clip_coefficient']<1 for c in result['cases'])==result['clipped_cases']==18 and result['unclipped_cases']==36,'Both actual clipping branches required')
        require(result['max_moment_error']<plan['tolerances']['moment_atol'] and result['max_clip_coefficient_error']<plan['tolerances']['clip_atol'] and result['max_muon_direction_error']<plan['tolerances']['muon_direction_atol'],'Original independent scalar arithmetic agreement required')
        require(max(result['noncommutation_max_abs'].values())>1e-7 and max(result['sequential_vs_fixed_state_max_abs'].values())>1e-7,'Actual discriminating nonlinear and state-evolution controls required')
        require(result['actual_model_optimizer_steps']==result['models_or_raw_loaded']==result['weight_files_written']==0 and result['generated_optimizer_steps_only'] and result['mps_api_called'] is False and result['corpus_loaded'] is False and result['validation_encoded'] is False and result['holdout_encoded'] is False,'Generated-only CPU scope required')
        require(len(plan['sources'])==result['source_bindings']==1050,'All1050 source bindings required')
        for p,h in plan['sources'].items():require(reference(ROOT/p)['sha256']==h,'Immutable source changed '+p)
        for r in [*plan['libraries'],plan['actual_inherited_decay_source']]:require(reference(r['path'])==r,'Exact actual installed/inherited optimizer source required')
        require(reference(R/'STOP')==terminal and not list(R.glob('*.pt')) and sum(p.stat().st_size for p in R.glob('*'))<plan['metadata_cap'],'Original terminal and metadata-only output required')
    out=dict(status='generated_optimizer_response_proof_independently_reviewed',source=reference(Path(__file__)),terminal=terminal,result=stop['result'],plan=result['source_plan'],original_guard=reference(R/'guard.json'),review_guard=reference(E/'optimizer-response-review001-guard.json'),source_bindings=1050,cases=54,rejections=23,optimizer_steps_replayed=0,models_or_raw_loaded=0,mps_api_called=False,quality_result=False,adopted=False,next_action='Review only a NEW additive tiny generated CPU/MPS response-transport proof before any actual checkpoint-state consumer or endpoint measurement. Preserve this CPU-only component/proof and all predecessors.')
    with (E/'optimizer-response-review001.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status=out['status'],cases=54,source_bindings=1050)),flush=True)
