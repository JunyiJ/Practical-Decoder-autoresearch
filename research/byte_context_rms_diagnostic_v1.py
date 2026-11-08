"""Isolate block0 RMSNorm/attention/FFN errors on the new failed weights only."""
import argparse,fcntl,json,os,traceback
from pathlib import Path
from research.byte_context_transform_pilot_v1 import ROOT,BASE,sha,put,read,runtime,pins
from research.byte_context_numerical_diagnostic_v1 import delta
OUT=BASE/'rms-diagnostic001'


def prepare():
    assert not OUT.exists()
    parent=read(BASE/'numerical-diagnostic001/plan.json');pins(parent)
    qp=read(BASE/'quality-plan.json')
    sources={str(p.relative_to(ROOT)):sha(p) for f in ('research','src') for p in (ROOT/f).rglob('*.py')}
    inputs=[BASE/'numerical-diagnostic001/plan.json',BASE/'numerical-diagnostic001/result.json']
    plan=dict(kind='byte_context_rms_localization_v1',sources=sources,assets=parent['assets']+[dict(path=str(p),sha256=sha(p)) for p in inputs],runtime=runtime(),
              block_starts=parent['block_starts'],weight_sha256=parent['weight_sha256'],
              scope='Same first and worst blocks, no training. Compare default RMSNorm with explicit contract epsilon2^-23 and diagnostic1e-5, against CPU64 formula. Isolate attention and FFN on common inputs.',
              limits=dict(seconds=300,rss=4*1024**3,mps=6*1024**3,reserve=8*1024**3))
    OUT.mkdir();put(OUT/'plan.json',plan);print(sha(OUT/'plan.json'),flush=True)


def run():
    from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
    import torch
    from torch.nn import functional as F
    from research.byte_context_transform_v1 import construct,load_weights
    from research.byte_input_model_v3 import build_codebook
    from research.byte_input_training_v6 import load_view
    plan=read(OUT/'plan.json');pins(plan);qp=read(BASE/'quality-plan.json');spec=qp['specs'][0]
    class Guard(CpuPrerequisiteGuard):
        def _sample(self):
            sample,reason=super()._sample();sample['mps_bytes']=torch.mps.driver_allocated_memory()
            if sample['mps_bytes']>=6*1024**3:reason='mps_limit'
            return sample,reason
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        put(OUT/'attempt.json',dict(pid=os.getpid(),plan_sha256=sha(OUT/'plan.json')))
        try:
            with Guard(OUT/'guard.json',ROOT,timeout_seconds=300,poll_seconds=.5):
                torch.set_num_threads(4);assert os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK')=='0'
                book=build_codebook(*(Path(r['path']).read_bytes() for r in qp['codebook_assets']),'raw_bytes')
                p=ROOT/'research_runs/byte-context-transform-001/00/h200.pt';assert sha(p)==plan['weight_sha256']
                saved=torch.load(p,map_location='cpu',weights_only=True)
                cpu=construct('byte',spec['roles'],spec['projection_seed'],book,'cpu');load_weights(cpu,saved['weights'])
                mps=construct('byte',spec['roles'],spec['projection_seed'],book,'mps');load_weights(mps,saved['weights'])
                assert saved['identity']==cpu.identity==mps.identity
                cpu.eval();mps.eval();view=load_view(spec['roles'],'cpu');reports=[]
                with torch.no_grad():
                    for start in plan['block_starts']:
                        ids=view.val_data[start:start+128].unsqueeze(0);base=cpu.base;idx=ids.reshape(-1)
                        h=base.composer(base.code_ids.index_select(0,idx),base.code_lengths.index_select(0,idx)).reshape(1,128,256)
                        cb=cpu.base.blocks[0];mb=mps.base.blocks[0]
                        n1=cb._normalize(cb.ln1,h)
                        attn,_=cb.attn(n1,cache=None,start_pos=0,freq_cos=cb.freq_cos,freq_sin=cb.freq_sin)
                        residual=h+attn;n2=cb._normalize(cb.ln2,residual)
                        ma,_=mb.attn(n1.to('mps'),cache=None,start_pos=0,freq_cos=mb.freq_cos,freq_sin=mb.freq_sin)
                        mf=mb.mlp(n2.to('mps'));cf=cb.mlp(n2)
                        rows=[]
                        for name,x in [('ln1',h),('ln2',residual)]:
                            cn=getattr(cb,name);mn=getattr(mb,name);xd=x.double();w=cn.weight.double()
                            expected=xd*torch.rsqrt(xd.square().mean(-1,keepdim=True)+2**-23)*w
                            outputs={}
                            for dev,norm in [('cpu',cn),('mps',mn)]:
                                z=x.to(dev)
                                values={'module_default':norm(z),'functional_explicit_contract':F.rms_norm(z,(256,),norm.weight,2**-23),
                                        'functional_explicit_1e5':F.rms_norm(z,(256,),norm.weight,1e-5)}
                                outputs[dev]={k:v.cpu() for k,v in values.items()}
                                rows.append(dict(norm=name,device=dev,module_eps=norm.eps,
                                    vs_fp64_contract={k:delta(expected,v) for k,v in values.items()},
                                    default_vs_explicit_contract=delta(values['module_default'],values['functional_explicit_contract']),
                                    default_vs_explicit_1e5=delta(values['module_default'],values['functional_explicit_1e5'])))
                            rows.append(dict(norm=name,cpu_mps_default=delta(outputs['cpu']['module_default'],outputs['mps']['module_default']),
                                             cpu_mps_explicit_contract=delta(outputs['cpu']['functional_explicit_contract'],outputs['mps']['functional_explicit_contract'])))
                        reports.append(dict(block_start=start,isolated_attention=delta(attn,ma),isolated_ffn=delta(cf,mf),norms=rows))
                put(OUT/'result.json',dict(completed=True,reports=reports,no_training=True,no_adoption=True));pins(plan)
            put(OUT/'acceptance.json',dict(accepted=True,result_sha256=sha(OUT/'result.json'),guard_sha256=sha(OUT/'guard.json')))
            print(json.dumps(read(OUT/'result.json')),flush=True)
        except BaseException:
            put(OUT/'failure.json',dict(traceback=traceback.format_exc()));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('prepare','run'));a=p.parse_args()
    prepare() if a.action=='prepare' else run()
