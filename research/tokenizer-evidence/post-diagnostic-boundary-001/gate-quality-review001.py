"""Read-only full-quality gate review; no contract/model/statistics replay."""
import json,math,os,shutil
from pathlib import Path
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference
from research.tokenizer_mla_gate_quality_v1 import contract as C
from research.tokenizer_mla_gate_worker_v1.execution import idle
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
E=Path(__file__).resolve().parent;OUT=E/'gate-quality-review001.json';GUARD=E/'gate-quality-review001-guard.json'
def read(p):return json.loads(Path(p).read_text())
def publish(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
with session_lock(GLOBAL_LOCK.parent):
    idle();require(not OUT.exists() and not GUARD.exists(),'Fresh independent full-quality gate review only')
    with CpuPrerequisiteGuard(GUARD,ROOT,timeout_seconds=120):
        result=read(E/'gate-quality-contracts001.json');plan=read(E/'gate-quality-contracts001-plan.json');guard=read(E/'gate-quality-contracts001-guard.json')
        require(result['status']=='passed_full_quality_bind_state_sources_and_final_contracts' and result['source']==plan['source']==reference(E/'gate-quality-contracts001.py') and result['sources']==plan['sources']==C.sources() and result['design']==plan['design']==C.DESIGN,'Exact complete own-runtime source gate required')
        require(len(result['sources'])==1084 and all(result['sources'].get(p)==h for p,h in read(E/'gate-pilot001-plan.json')['sources'].items()),'All1074 predecessor and1084 current bindings required')
        require(guard['status']=='finished' and guard['reason'] is None and guard['elapsed_seconds']<600 and guard['observation']['peak_rss_bytes']<4*1024**3 and guard['observation']['peak_observed_mps_driver_bytes']<6*1024**3 and guard['observation']['free_bytes']>=8*1024**3,'Original proof resource gates required')
        rows=result['real_pilot_state_and_full_layout_checks'];require([r['arm'] for r in rows]==list(C.ARMS),'Complete three-arm layout proof required')
        for r in rows:
            require(r['actual_pilot_state_checked'] and r['full_fixture_schema_validated'] and r['fixture_values_NOT_trained_full_horizon'] and r['raw_counted_bytes']+r['h200_counted_bytes']==r['full_history_layout_counted_bytes'] and r['cap']==C.CAPS[r['arm']] and r['margin_bytes']==r['cap']-r['full_history_layout_counted_bytes'] and r['additional_ancestry_path_headroom_bytes']==4096 and r['margin_bytes']>=8*1024**2+4096,'Complete conservative full-state layout and cap margin required')
        require(len(result['rejected_cases'])==35 and result['positive_actual_source_fixture_binding'] and result['fixture_path_locators_only'] and result['synthetic_tiny_payloads_NOT_numerical_or_quality_acceptance'] and result['independent_nonconstant_final_review_agreed'] and result['global_cpu_mps_rng_preserved'] and result['weight_files_written']==0 and not result['training'] and not result['model_scoring'] and not result['scientific_seed_used'] and not result['holdout_encoded_or_scored'],'All generated correctness cases and limited scope required')
        fixture=E/'gate-quality-contracts001-fixtures';tiny=list(fixture.rglob('*.pt'))+list(fixture.rglob('*.pt.tmp'))
        require(len(tiny)==82 and all(p.stat().st_size<128 and (p.read_bytes().startswith(b'SYNTHETIC_LAYOUT_FIXTURE_ONLY_') or p.read_bytes()==b'EXTRA_SYNTHETIC_ONLY') for p in tiny),'Synthetic text fixtures are not checkpoints')
        for name in ('valid_zero','valid_one','nine_and_immutable_selection','complete_eighteen_and_fresh_contracts'):require((fixture/name).is_dir(),'Required positive prefix fixture absent')
        retained=0;weights=[]
        for f in ('gate-worker-review001.json','gate-pilot-review001.json'):
            old=read(E/f)
            for r in old['weights']:
                p=Path(r['path']);s=p.stat();require(not p.is_symlink() and s.st_nlink==1 and s.st_ino==r['inode'] and s.st_dev==r['device'] and s.st_size==r['size'] and reference(p)['sha256']==r['sha256'],'Original accepted prerequisite payload changed');retained+=s.st_size;weights.append({k:r[k] for k in ('path','sha256','size')})
        require(len(weights)==12 and retained==3363688308 and not C.RUN.exists() and not C.PLAN.exists() and not C.FREEZE.exists(),'No production attempt;all12prerequisites protected')
        out=dict(status='full_gate_quality_correctness_independently_reviewed',source=reference(__file__),proof=reference(E/'gate-quality-contracts001.json'),proof_plan=reference(E/'gate-quality-contracts001-plan.json'),proof_guard=reference(E/'gate-quality-contracts001-guard.json'),source_bindings=1084,design=C.DESIGN,layout_rows=rows,source_invariant=True,prerequisite_weights=weights,prerequisite_weight_bytes=retained,tiny_text_pt_files=len(tiny),tiny_text_pt_bytes=sum(p.stat().st_size for p in tiny),models_or_contracts_or_statistics_replayed=False,scientific_seeds_assigned=False,production_freeze_exists=False,quality_result=False,adopted=False,quality_floor=C.FLOOR,quality_target=C.TARGET,free_bytes=shutil.disk_usage(ROOT).free)
        require(out['free_bytes']>=C.TARGET,'Complete18trajectory retained target required');publish(OUT,out);publish(E/'STOP-gate-quality-contracts001',dict(status='complete_full_quality_correctness_proof_and_review',proof=out['proof'],review=reference(OUT),never_rerun=True))
        print(json.dumps(dict(status=out['status'],source_bindings=1084,tiny_text_pt_files=len(tiny))),flush=True)
