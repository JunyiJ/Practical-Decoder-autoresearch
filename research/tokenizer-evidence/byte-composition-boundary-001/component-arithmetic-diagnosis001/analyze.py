"""New existing-data FP32 primitive diagnostic, not a model/gradient proof replay.

No candidate, historical oracle, full native body, optimizer, backward, private
initializer or corpus call. Compare two algebraic component arithmetic paths
from saved initial arrays to explain the unresolved derivative discrepancy.
"""
import fcntl,gzip,hashlib,json,os
from pathlib import Path
import numpy as np
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path.cwd();BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
SOURCE=BASE/'model-math003';OUT=BASE/'component-arithmetic-diagnosis001';PINS=[]
def load(name):
 p=SOURCE/(name+'.json');m=json.loads(p.read_text());compressed=(SOURCE/m['data_file']).read_bytes();raw=gzip.decompress(compressed)
 assert len(raw)==m['raw_bytes'] and hashlib.sha256(raw).hexdigest()==m['raw_sha256']
 a=np.frombuffer(raw,dtype=[('a','<f4'),('b','<f4'),('m','u1')]);assert np.array_equal(a['a'],a['b'])
 PINS.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),payload_sha256=hashlib.sha256(compressed).hexdigest()))
 return a['a'].reshape(m['shape']).copy()
with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
  import torch
  from torch.nn import functional as F
  assert torch.__version__=='2.14.1'
  torch.set_num_threads(4);before=torch.random.get_rng_state().clone()
  codes=load('codebook-raw_bytes-rows').astype(np.int64);lengths=load('codebook-raw_bytes-lengths').astype(np.int64)
  ids=np.arange(1024)*49%50256;nul=int(np.flatnonzero((lengths==3)&(codes[:,1]==0))[0]);last=int(np.flatnonzero((lengths==3)&(codes[:,1]==255))[0])
  ids[:8]=[50256,int(np.argmax(lengths[:50256])),nul,last,0,0,1,2]
  params={n:torch.from_numpy(load('cpu-raw_bytes-initial-composer.'+n)) for n in ['byte_weight','highway_weight','highway_bias']+[f'filters.{i}' for i in range(6)]+[f'filter_biases.{i}' for i in range(6)]}
  seq=torch.from_numpy(codes[ids]);lens=torch.from_numpy(lengths[ids]);position=torch.arange(130)[None,:]
  with torch.inference_mode():
   embedded=F.embedding(seq-(seq>256).long(),params['byte_weight']).masked_fill(~(position<lens[:,None])[:,:,None],0.)
   a=[];b=[];c=[]
   for i in range(6):
    w=i+1;count=131-w;weight=params[f'filters.{i}'];bias=params[f'filter_biases.{i}']
    allowed=position[:,:count]+w<=lens[:,None]
    unfold=embedded.unfold(1,w,1).contiguous().reshape(1024,count,15*w)
    # Separately derive each window using explicit slices and stacking. This
    # uses the declared fused FP32 primitive while avoiding candidate unfold.
    stacked=torch.stack([embedded[:,j:j+count] for j in range(w)],dim=-1).reshape(1024,count,15*w)
    va=torch.tanh(F.linear(unfold,weight,bias))
    vb=torch.tanh(F.conv1d(embedded.transpose(1,2),weight.reshape(weight.shape[0],15,w),bias)).transpose(1,2)
    vc=torch.tanh(F.linear(stacked,weight,bias))
    for value,dest in [(va,a),(vb,b),(vc,c)]:
     maximum=value.masked_fill(~allowed[:,:,None],-2.).amax(1)
     dest.append(torch.where((lens>=w)[:,None],maximum,torch.zeros_like(maximum)))
   ya,yb,yc=[torch.cat(x,1) for x in (a,b,c)]
   pa=F.linear(ya,params['highway_weight'],params['highway_bias'])
   pb=yb@params['highway_weight'].T+params['highway_bias']
   pc=F.linear(yc,params['highway_weight'],params['highway_bias'])
   a,b,c=[x.numpy().copy() for x in (pa,pb,pc)]
   mask=(a>0)!=(b>0);rr,cc=np.where(mask)
   report=dict(scope='New CPU component primitive analysis on saved arrays; no old model/reference function, backward, optimizer or initializer replay.',
    changed_relu_decisions=int(mask.sum()),old_forward_arithmetic_max_abs=float(np.abs(a.astype(np.float64)-b).max()),
    explicit_window_same_fused_primitive_exact=bool(np.array_equal(a,c)),
    explicit_window_relu_decisions_equal=bool(np.array_equal(a>0,c>0)),
    decision_changes=[dict(row=int(r),token=int(ids[r]),unit=int(col),unfold_fused=float(a[r,col]),conv_separate=float(b[r,col]),explicit_window_fused=float(c[r,col])) for r,col in zip(rr,cc)],
    matches_failed_bias_unit_382=bool(mask[:,382].any()),inputs=PINS,
    rng_preserved=bool(torch.equal(before,torch.random.get_rng_state())),
    no_numerical_acceptance_or_quality_claim=True)
   with (OUT/'preactivations.npz').open('xb') as f:
    np.savez_compressed(f,unfold_fused=a,conv_separate=b,explicit_window_fused=c,decision_disagreement=mask,token_ids=ids);f.flush();os.fsync(f.fileno())
   report['saved_arrays_sha256']=hashlib.sha256((OUT/'preactivations.npz').read_bytes()).hexdigest()
   with (OUT/'result.json').open('x') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
   assert report['rng_preserved']
