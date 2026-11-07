"""Independent original-metadata reconciliation; no tensor/oracle/backend replay."""
import argparse,fcntl,hashlib,json,math,os
from pathlib import Path
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path(__file__).resolve().parents[4]
R=Path(__file__).resolve().parent
F=R.parent/'literal-vocabulary-ce001'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def put(n,v):
 raw=(json.dumps(v,indent=2,sort_keys=True,allow_nan=False)+'\n').encode()
 assert sum(p.stat().st_size for p in R.iterdir() if p.is_file())+len(raw)<=2*1024**2
 with (R/n).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--plan-sha',required=True);args=ap.parse_args()
 assert sha(R/'plan.json')==args.plan_sha and not (R/'attempt.json').exists();plan=read(R/'plan.json')
 with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  put('attempt.json',dict(pid=os.getpid(),plan_sha256=args.plan_sha))
  with CpuPrerequisiteGuard(R/'guard.json',ROOT,timeout_seconds=120,poll_seconds=.25):
   for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
   for p,h in plan['original_metadata'].items():assert sha(ROOT/p)==h,p
   result=read(F/'result.json');assert result['status']=='fixed_numerical_predicate_failed'
   assert read(F/'STOP')['status']=='closed_failed' and read(F/'STOP')['no_retry'] is True
   false={k for k,v in result['predicates'].items() if not v};assert false=={'CPU_MPS_gradient','cpu32_oracle'}
   assert result['caller_CPU_MPS_RNG_unchanged'] and result['source_pins_unchanged']
   assert result['negative_counts']==dict(cpu64=24,cpu32=25,mps32=24)
   whole=read(F/'mps-guard.json');assert whole['status']=='fixture_exception' and whole['elapsed_seconds']<600
   guards=[read(F/n) for n in ['cpu-oracle-guard.json','cpu64-guard.json','cpu32-guard.json']]
   assert all(g['status']=='finished' and g['reason'] is None and g['elapsed_seconds']<120 for g in guards)
   failures=[];maximum={}
   for backend in ['cpu64','cpu32','mps32']:
    rows=[]
    for part in ['six','perturb','slice']:
     doc=read(F/(backend+'-'+part+'.json'));assert doc['caller_input_targets_populated_gradslots_unchanged']
     for row in doc['rows']:
      assert all(type(row[k]) is float and math.isfinite(row[k]) and row[k]>=0 for k in ['ce','oracle_ce','forward_abs_error','gradient_abs_max'])
      assert row['forward_abs_error']==abs(row['ce']-row['oracle_ce'])
      assert row['forward_pass']==(row['forward_abs_error']<=doc['tolerances']['forward_atol'])
      if not row['gradient_elementwise_pass']:failures.append(dict(backend=backend,part=part,**row))
     rows+=doc['rows']
    maximum[backend]=dict(forward=max(x['forward_abs_error'] for x in rows),gradient=max(x['gradient_abs_max'] for x in rows))
    assert len(rows)==8 and maximum[backend]['forward']==result['forward_oracle_max'][backend]
    negatives=read(F/(backend+'-invalids.json'));assert len(negatives['rejected'])==result['negative_counts'][backend] and len(set(negatives['rejected']))==len(negatives['rejected'])
   assert len(failures)==1 and failures[0]['backend']=='cpu32' and failures[0]['row']==2
   pairs=read(F/'paired-comparison.json');assert pairs==result['pairs']
   diffs=[abs(a-b) for a,b in zip(pairs['cpu32_CE'],pairs['mps32_CE'])]
   assert diffs==pairs['per_row_abs_ce_error'] and max(diffs)<=3e-5 and abs(pairs['perturb_cpu32_CE']-pairs['perturb_mps32_CE'])<=3e-5
   assert read(F/'oracle.json')['common_shift_reference_exact'] is True
   for p,h in plan['original_metadata'].items():assert sha(ROOT/p)==h,p
   for p,h in plan['sources'].items():assert sha(ROOT/p)==h,p
   put('review.json',dict(status='closed_failed_literal_CE_probe_independent_metadata_reconciled',source_bindings=len(plan['sources']),original_metadata_files=len(plan['original_metadata']),false_predicates=sorted(false),maximum_errors=maximum,failed_gradient_rows=failures,same_input_CPU_MPS_CE_max=max(diffs),rejection_count=sum(result['negative_counts'].values()),
    input_and_gradslot_and_global_RNG_preservation_original_assertions_recorded=True,all3CPUguards_finished=True,MPSguard=whole,
    no_tensor_model_optimizer_corpus_checkpoint_oracle_backend_rng_scoring_replay=True,no_weights_or_deletions=True,
    limits='Reconciles reported elementwise gradient flags and max errors; does not recompute unavailable full gradient tensors/coordinates or locate trained model failure. CPU32 singleton and six-row periodic both agree within fixed forward tolerance; no kernel identity inferred.'))
  put('STOP',dict(status='complete_original_literal_probe_metadata_review',no_replay=True,no_retry=True))
if __name__=='__main__':
 try:main()
 except BaseException as exc:
  if (R/'attempt.json').exists() and not (R/'STOP').exists():put('STOP',dict(status='closed_failed',error=repr(exc),no_retry=True))
  raise
