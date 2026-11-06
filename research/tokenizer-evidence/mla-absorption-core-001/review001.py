"""Bounded retained-weight affine core oracle/profile; no trained-model change."""
import argparse,gc,json,statistics,time
from pathlib import Path
import torch
from research.body_allocation_model_v1.contract import ROOT,require
from research.body_allocation_model_v1.evidence import reference,environment
from research.body_allocation_worker_v1.execution import checked,sources
from research.body_allocation_worker_v1.validation import verify_artifact
from research.mla_context_extension_v1.model import GPT,config,import_original_h200
from research.mla_absorption_diagnostic_v1.core import original,Absorbed,matmul_macs
from research.block_norm_v1.mps_guard import StrictMpsGuard
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
from src.research.tokenizer_activation_entry import GLOBAL_LOCK,session_lock
from research.final_ln_scale_quality_v3.support import idle
E=Path(__file__).resolve().parent;B=ROOT/'research/tokenizer-evidence/body-allocation-preparation-001'
p=argparse.ArgumentParser();p.add_argument('device',choices=['cpu','mps']);a=p.parse_args();torch.set_num_threads(4)
def inventory():
    result=sources()
    for ns in ('mla_context_extension_v1','mla_absorption_diagnostic_v1'):
        for f in sorted((ROOT/'research'/ns).glob('*.py')):result[str(f.relative_to(ROOT))]=reference(f)['sha256']
    return result
def publish(p,d):
    payload=(json.dumps(d,indent=2,allow_nan=False)+'\n').encode();require(len(payload)<=512*1024,'Bounded metadata')
    with p.open('xb') as f:f.write(payload)
def array(shape,offset):
    return torch.sin((torch.arange(__import__('math').prod(shape),device=a.device).float()+offset)/97).reshape(shape)*.2
with session_lock(GLOBAL_LOCK.parent):
    idle();environment();out=E/f'review001-{a.device}.json';guardpath=E/f'review001-{a.device}-guard.json';planpath=E/f'plan001-{a.device}.json'
    require(not out.exists() and not guardpath.exists() and not planpath.exists(),'Closed core proof/profile cannot retry')
    closure=checked(reference(B/'quality001-closure001.json'));rows=[r for r in closure['trajectories'] if r['slot']>=9 and r['arm']=='incumbent'];require(len(rows)==3,'All fixed models required')
    pinned=inventory()
    if a.device=='mps':require(checked(reference(E/'review001-cpu.json'))['sources']==pinned,'Exact passed CPU prerequisite required')
    plan=dict(kind='affine_mla_attention_core_oracle_profile',device=a.device,sources=pinned,source=reference(__file__),closure=reference(B/'quality001-closure001.json'),
        cases=[dict(batch=b,queries=q,keys=k) for b in (1,8) for q,k in ((1,32),(1,128),(128,128))],
        resources='One retained model resident,4threads,CPU120s or strictMPS600s/RSS4GiB/MPS6GiB/reserve8GiB;512KiB metadata per record;no weight outputs.',
        prediction='Avoid cached KV expansion may help query1; latent64 exceeds head32,so prefill and output work can grow. No promised speedup.',
        equations='Q_h=Uq_h(cq); scores=.5/sqrt64*((Q_h Wuk_h) C^T + Q_h buk_h + Rq_h Rk^T); output=sum_h((A_h C)(O_h Wuv_h)^T + rowsum(A_h) O_h buv_h)+bout.',
        control='Original concatenated content/rotary attention core with identical inputs/weights/mask; all4layers of all3retained models. No gradient/train/model/serving adoption.',
        timing='MPS only,2warmups+5timed repeats per core. Synchronize each call; rotate first core by layer. Includes candidate weight-stamp validation; excludes precompute,loading,allocation,down projections,RoPE,cache updates,norms,MLP. Not end-to-end latency.',
        new_scientific_seed=False,training=False,quality_claim=False,holdout_scored=False)
    publish(planpath,plan);reports=[]
    guard=StrictMpsGuard(guardpath,ROOT) if a.device=='mps' else CpuPrerequisiteGuard(guardpath,ROOT,timeout_seconds=120,poll_seconds=.25)
    try:
        with guard,torch.random.fork_rng(devices=[]),torch.no_grad():
            for row in rows:
                worker=checked(row['worker']);ref=worker['h200'];verify_artifact(ref)
                m=GPT(config()).to(a.device);saved=torch.load(ref['path'],map_location='cpu',weights_only=True);import_original_h200(m,saved,ref['sha256']);del saved
                stamp=m.stamp();buffers=m.buffer_stamp()
                for layer,block in enumerate(m.transformer.h):
                    core=Absorbed(block.attn)
                    for b in (1,8):
                        for q,k in ((1,32),(1,128),(128,128)):
                            cq=array((b,q,64),1);c=array((b,k,64),13);rq=array((b,8,q,32),37);rk=array((b,k,32),91)
                            mask=torch.arange(k,device=a.device)[None,:]<=torch.arange(k-q,k,device=a.device)[:,None]
                            inputs=(cq,c,rq,rk,mask);oracle=original(block.attn,*inputs);candidate=core(*inputs)
                            torch.testing.assert_close(candidate,oracle,rtol=3e-4,atol=3e-5)
                            record=dict(seed=worker['config']['training']['seed'],layer=layer,batch=b,queries=q,keys=k,
                                maximum_absolute_error=float((candidate-oracle).abs().max()),macs=matmul_macs(b,q,k))
                            if a.device=='mps':
                                fns=dict(original=lambda:original(block.attn,*inputs),absorbed=lambda:core(*inputs));times={}
                                for name in (('original','absorbed') if layer%2==0 else ('absorbed','original')):
                                    fn=fns[name]
                                    for _ in range(2):value=fn();torch.mps.synchronize();del value
                                    timed=[]
                                    for _ in range(5):
                                        torch.mps.synchronize();started=time.perf_counter();value=fn();torch.mps.synchronize();timed.append(time.perf_counter()-started);del value
                                    times[name]=dict(median_seconds=statistics.median(timed),all_seconds=timed)
                                record['timings']=times;record['absorbed_over_original_median']=times['absorbed']['median_seconds']/times['original']['median_seconds']
                            reports.append(record);del oracle,candidate,inputs,cq,c,rq,rk,mask
                    del core
                require(m.stamp()==stamp and m.buffer_stamp()==buffers,'Retained model mutated');del m;gc.collect()
                if a.device=='mps':torch.mps.empty_cache()
                verify_artifact(ref)
            require(inventory()==pinned,'Source changed during core review')
        result=dict(status='passed_core_oracle_profile',device=a.device,plan=reference(planpath),sources=pinned,reports=reports,
            max_absolute_error=max(r['maximum_absolute_error'] for r in reports),full_model_integration=False,quality_claim=False,
            serving_adopted=False,training=False,new_scientific_seed=False,guard=reference(guardpath))
        if a.device=='mps':
            result['median_ratios_by_case']=[dict(batch=b,queries=q,keys=k,median=statistics.median(r['absorbed_over_original_median'] for r in reports if (r['batch'],r['queries'],r['keys'])==(b,q,k))) for b in (1,8) for q,k in ((1,32),(1,128),(128,128))]
        publish(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ('reports','sources','plan','guard')}))
    except Exception as exc:publish(E/f'STOP-{a.device}',dict(error=repr(exc),closed=True));raise
