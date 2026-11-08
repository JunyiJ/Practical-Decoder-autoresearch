"""New NumPy analysis of saved arrays only; no replay of model/reference calls."""
import fcntl,gzip,hashlib,json,os
from pathlib import Path
import numpy as np
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT=Path.cwd();BASE=ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT=BASE/'model-gradient-diagnosis001';SOURCE=BASE/'model-math003';PINS=[]
def load(name):
 p=SOURCE/(name+'.json');m=json.loads(p.read_text());data=(SOURCE/m['data_file']).read_bytes();raw=gzip.decompress(data)
 assert len(raw)==m['raw_bytes'] and hashlib.sha256(raw).hexdigest()==m['raw_sha256']
 PINS.append(dict(metadata=str(p),metadata_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),data_sha256=hashlib.sha256(data).hexdigest()))
 array=np.frombuffer(raw,dtype=[('a','<f4'),('b','<f4'),('m','u1')])
 return array,m

def initial(name):
 a,m=load('cpu-raw_bytes-initial-composer.'+name);assert np.array_equal(a['a'],a['b'])
 return a['a'].reshape(m['shape']).copy().astype(np.float64)
with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.25):
  gradient,gm=load('cpu-raw_bytes-gradient-composer.highway_bias');failed=np.flatnonzero(gradient['m']==0)
  codes,cm=load('codebook-raw_bytes-rows');codes=codes['a'].reshape(cm['shape']).astype(np.int64)
  lengths,lm=load('codebook-raw_bytes-lengths');lengths=lengths['a'].astype(np.int64)
  ids=np.arange(1024)*49%50256
  nul=int(np.flatnonzero((lengths==3)&(codes[:,1]==0))[0]);last=int(np.flatnonzero((lengths==3)&(codes[:,1]==255))[0])
  ids[:8]=[50256,int(np.argmax(lengths[:50256])),nul,last,0,0,1,2]
  byte=initial('byte_weight');weights=[initial('filters.'+str(i)) for i in range(6)];biases=[initial('filter_biases.'+str(i)) for i in range(6)]
  highway=initial('highway_weight');bias=initial('highway_bias');distance=np.full(525,np.inf);locations=np.zeros(525,dtype=int)
  close=[];sign_surrogate=[]
  for start in range(0,1024,16):
   seq=codes[ids[start:start+16]];lens=lengths[ids[start:start+16]];positions=np.arange(130)[None,:]
   embedded=byte[seq-(seq>256)]*(positions<lens[:,None])[:,:,None];pooled=[]
   for i in range(6):
    width=i+1;windows=np.lib.stride_tricks.sliding_window_view(embedded,width,axis=1).reshape(len(seq),131-width,15*width)
    values=np.tanh(windows@weights[i].T+biases[i]);allowed=np.arange(131-width)[None,:]+width<=lens[:,None]
    maximum=np.where(allowed[:,:,None],values,-2.).max(1)
    pooled.append(np.where((lens>=width)[:,None],maximum,0.))
   features=np.concatenate(pooled,axis=1);pre=features@highway.T+bias
   idx=np.abs(pre).argmin(axis=0);mins=np.abs(pre)[idx,np.arange(525)];improved=mins<distance
   distance[improved]=mins[improved];locations[improved]=start+idx[improved]
   r,c=np.where(np.abs(pre)<1e-7)
   close.extend(dict(row=int(start+x),token=int(ids[start+x]),unit=int(y),preactivation64=float(pre[x,y])) for x,y in zip(r,c))
   separate=(features.astype(np.float32)@highway.astype(np.float32).T)+bias.astype(np.float32)
   fused_surrogate=(features.astype(np.float32).astype(np.float64)@highway.astype(np.float32).astype(np.float64).T+bias.astype(np.float32).astype(np.float64)).astype(np.float32)
   r,c=np.where((separate>0)!=(fused_surrogate>0))
   sign_surrogate.extend(dict(row=int(start+x),token=int(ids[start+x]),unit=int(y),separate32=float(separate[x,y]),rounded_dot_plus_bias=float(fused_surrogate[x,y]),preactivation64=float(pre[x,y])) for x,y in zip(r,c))
  record=dict(classification='saved-coordinate gradient failure with independent component branch-distance diagnostic',
   failed_gradient_coordinates=len(failed),gradient_max_abs=gm['max_finite_abs'],
   failed_units=[dict(unit=int(i),actual=float(gradient['a'][i]),reference=float(gradient['b'][i]),nearest_preactivation64_distance=float(distance[i]),nearest_row=int(locations[i]),nearest_token=int(ids[locations[i]])) for i in failed],
   near_zero_preactivations=close,fp32_accumulation_surrogate_sign_disagreements=sign_surrogate,
   min_absolute_preactivation64=float(distance.min()),inputs=PINS,
   scope='Existing saved initial weights/codes and gradient coordinates only. NumPy FP64 component arithmetic and two reporting surrogates; not installed Torch fused kernels, no full model/reference/optimizer call, no original output recapture.',
   no_quality_conclusion=True,unchanged_gradient_gate_failed=True)
  with (OUT/'result.json').open('x') as f:json.dump(record,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
