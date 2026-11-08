"""Saved-data-only diagnosis; no Torch, model, reference or optimizer calls."""
import fcntl,gzip,hashlib,json,os
from pathlib import Path
import numpy as np
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path.cwd();BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'model-update-diagnosis001';SOURCE=BASE/'model-math002'
def load(name):
 p=SOURCE/(name+'.json');m=json.loads(p.read_text());raw=gzip.decompress((SOURCE/m['data_file']).read_bytes())
 assert len(raw)==m['raw_bytes'] and hashlib.sha256(raw).hexdigest()==m['raw_sha256']
 a=np.frombuffer(raw,dtype=[('a','<f4'),('b','<f4'),('m','u1')]);return a,m

def bf16(x):
 bits=np.asarray(x,dtype=np.float32).view(np.uint32)
 return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
  gradient,gm=load('cpu-native-gradient-transformer.h.0.attn.dkv.weight')
  update,um=load('cpu-native-update1-transformer.h.0.attn.dkv.weight')
  initial,im=load('cpu-native-initial-transformer.h.0.attn.dkv.weight')
  qa,qb=bf16(gradient['a']),bf16(gradient['b'])
  # A diagnostic rounded first-step Nesterov surrogate, explicitly not a
  # reproduction of the installed optimizer's fused lerp/addmm kernel.
  na,nb=bf16(gradient['a']*np.float32(.0975)),bf16(gradient['b']*np.float32(.0975))
  changed=np.flatnonzero(qa!=qb);nchanged=np.flatnonzero(na!=nb)
  failed=np.flatnonzero(update['m']==0)
  record=dict(classification='unequal-gradient optimizer differential; precision sensitivity is a test-harness confound',
   no_candidate_or_optimizer_execution=True,shape=gm['shape'],
   gradient_max_abs=gm['max_finite_abs'],gradient_exact_equal_coordinates=int(np.sum(gradient['a']==gradient['b'])),
   gradient_total_coordinates=len(gradient),gradient_norm_actual=float(np.linalg.norm(gradient['a'].astype(np.float64))),
   bfloat16_gradient_disagreements=len(changed),bfloat16_nesterov_surrogate_disagreements=len(nchanged),
   gradient_rounding_examples=[dict(index=int(i),actual=float(gradient['a'][i]),reference=float(gradient['b'][i]),actual_bf16=float(qa[i]),reference_bf16=float(qb[i])) for i in changed[:16]],
   update_max_abs=um['max_finite_abs'],failed_update_coordinates=len(failed),
   update_failure_examples=[dict(index=int(i),actual=float(update['a'][i]),reference=float(update['b'][i]),initial_actual=float(initial['a'][i]),initial_reference=float(initial['b'][i]),threshold=float(4e-6+4e-5*abs(float(update['b'][i])))) for i in failed[:16]],
   supported_fact='Both optimizer arms received different gradients; installed Muon source casts its input to BF16 before Newton-Schulz. Saved gradients cross BF16 rounding boundaries.',
   inference='The original test does not isolate wrapper update correctness from sensitivity to independently computed CE gradients. No local model-quality inference.',
   recovery='Keep independent forward/loss/gradient comparisons unchanged. In a separately identified fresh proof, copy the actual already-checked gradients to reference leaves immediately before update tests. Compare identical-input updates and states under unchanged tolerances; preserve this failed original.',
   limitations='No claim to have reproduced the failed update or proven its entire cause; no old model, gradients, optimizer or failure was rerun.',
   inputs=[dict(path=str(SOURCE/(n+'.json')),sha256=hashlib.sha256((SOURCE/(n+'.json')).read_bytes()).hexdigest()) for n in ['cpu-native-gradient-transformer.h.0.attn.dkv.weight','cpu-native-update1-transformer.h.0.attn.dkv.weight','cpu-native-initial-transformer.h.0.attn.dkv.weight']])
  with (OUT/'result.json').open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
