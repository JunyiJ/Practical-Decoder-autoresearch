"""Independent terminal/source/count/resource review,no model or proof replay."""
import json
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_moe_worker_v1.execution import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent
def checked(r):
    p=Path(r['path']);require(reference(p)=={k:r[k] for k in ('path','sha256')},'Immutable evidence changed');return json.loads(p.read_text())
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not(E/'review001.json').exists() and not(E/'review001-guard.json').exists(),'Fresh terminal review only')
    with CpuPrerequisiteGuard(E/'review001-guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
        terminals={k:reference(E/k/'STOP') for k in ('gate-component001','gate-model001')};stops={k:checked(v) for k,v in terminals.items()};results={k:checked(v['result']) for k,v in stops.items()};plans={k:checked(v['source_plan']) for k,v in results.items()};guards={k:checked(reference(E/k/'guard.json')) for k in terminals}
        require(all(s['never_rerun'] for s in stops.values()) and all(g['status']=='finished' and g['reason'] is None for g in guards.values()),'Original proofs must be complete and terminal')
        c=results['gate-component001'];m=results['gate-model001'];cp=plans['gate-component001'];mp=plans['gate-model001'];design=checked(mp['design']);roles=checked(mp['roles'])
        require(c['status']=='elementwise_gate_scalar_output_gradient_and_native_cpu_mps_contracts_passed' and len(c['small_oracle_cases'])==9 and len(c['parameter_free_control_cases'])==6 and len(c['native_shape_cases'])==4 and c['rejections_before_mutation']==22,'Complete component oracle/control/adversary panel required')
        require({(r['device'],r['dtype'],r['weight_case']) for r in c['small_oracle_cases']}=={('cpu',dtype,kind) for dtype in ('torch.float32','torch.float64') for kind in ('zero','ordinary','saturated')}|{('mps','torch.float32',kind) for kind in ('zero','ordinary','saturated')},'Exact independent oracle cases required')
        for r in c['small_oracle_cases']:
            limit=cp['tolerances']['cpu64_atol' if r['dtype']=='torch.float64' else 'fp32_atol'];require(max(r['output_x_y_w_maxerrors'])<=limit and r['query_local_partition'] and r['inputs_and_parameter_values_unchanged'],'Original scalar oracle tolerance or immutability failed')
        require(all(r['gate_parameters']==65536 and r['four_layer_added_parameters']==262144 for r in c['native_shape_cases']),'Elementwise native parameter count differs')
        require(m['status']=='own_native_mla_gate_model_optimizer_gradient_and_cache_integration_passed' and all(m['native_control_exact'].values()) and m['static_learned_zero_common_gradients_exact'] and m['common_initial_weights_and_rng_exact'],'Exact native control and initial amplitude comparison required')
        expected={'none':(28492416,44,44,88),'static_half':(28492416,44,44,88),'sigmoid_elementwise_zero':(28754560,48,44,92)}
        require([r['arm'] for r in m['rows']]==list(expected),'All three complete model arms required')
        for r in m['rows']:
            require((r['parameters'],r['muon_groups'],r['adam_groups'],r['learned_gradients'])==expected[r['arm']] and r['one_generated_mps_step'] and r['all_perparameter_exposures_one'],'Own model/optimizer inventory or exposure differs')
            require(r['cpu_mps_ce_error']<=mp['tolerances']['ce_atol'] and r['cpu_mps_gradient_maxerror']<=mp['tolerances']['gradient_atol'] and r['cpu_mps_logit_maxerror']<=mp['tolerances']['logits_atol'],'Original generated CPU/MPS model tolerance failed')
            require([v['batch'] for v in r['cache']]==[1,8] and all(v['bytes_per_item']==196608 and v['maxerror']<=mp['tolerances']['logits_atol'] for v in r['cache']) and r['transactional_weights_optimizer_rejections']==4,'All complete generated cache/reset/transactional controls required')
        require(all(v['global_cpu_mps_rng_unchanged'] for v in results.values()) and c['optimizer_steps']==0 and c['tensor_files']==m['tensor_files_written']==0 and not c['corpus_loaded'] and not m['corpus_loaded'] and not m['quality_result'],'No language-data/quality/tensor output permitted')
        require(roles['no_prior_matches'] and roles['generated_model_initializer']==m['initializer_role_used']==600511 and not roles['scientific_seed'],'New initializer fixture role required')
        require(len(cp['sources'])==1057 and len(mp['sources'])==1062 and all(mp['sources'][p]==h for p,h in cp['sources'].items()),'All immutable predecessor source pins required')
        for p,h in mp['sources'].items():require(reference(ROOT/p)['sha256']==h,'Immutable source changed '+p)
        budget=design['complete_retained_budget'];require(budget['all_new_pairs_retained']==24 and budget['new_weight_caps']==8*sum(budget['pair_caps'].values()) and budget['full_floor']==sum(budget[k] for k in ('new_weight_caps','metadata_allowance','sidecar_allowance','reserve','external_allowance','entry')) and budget['target']==budget['full_floor']+2*1024**3 and budget['free_bytes']>=budget['target'],'Complete prospective retained-output accounting differs')
        require(all(reference(E/k/'STOP')==v for k,v in terminals.items()),'Original terminal changed')
    out=dict(status='gate_direction_component_and_generated_model_independently_reviewed',source=reference(Path(__file__)),component=stops['gate-component001']['result'],component_plan=reference(E/'gate-component001/plan.json'),component_terminal=terminals['gate-component001'],model=stops['gate-model001']['result'],model_plan=reference(E/'gate-model001/plan.json'),model_terminal=terminals['gate-model001'],guards={k:reference(E/k/'guard.json') for k in terminals},source_bindings=1062,model_loaded_by_review=False,proofs_replayed=False,quality_panel_frozen=False,scientific_seeds_assigned=False,adopted=False)
    with (E/'review001.json').open('x') as f:json.dump(out,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(dict(status=out['status'],source_bindings=1062)),flush=True)
