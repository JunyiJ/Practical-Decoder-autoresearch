"""Independent literal/symbolic source-plan review only; no component imports."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from src.research.cpu_prerequisite_guard import CpuPrerequisiteGuard
ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'research/tokenizer-evidence/byte-composition-boundary-001'
OUT = BASE/'component-plan-review001'

def require(value, reason):
    if not value:
        raise ValueError(reason)

def write(name, value):
    data = (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()
    require(sum(p.stat().st_size for p in OUT.iterdir())+len(data)<2097152,'Metadata cap')
    with (OUT/name).open('xb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())

def main(expected):
    with (ROOT/'research_state/tokenizer-worker-lock/.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require({p.name for p in OUT.iterdir()}=={'plan.json'},'One original plan review only')
        write('attempt.json',dict(pid=os.getpid(),plan_sha256=expected,no_retry=True,no_replay=True))
        try:
            with CpuPrerequisiteGuard(OUT/'guard.json',ROOT,timeout_seconds=120,rss_limit_bytes=4294967296,reserve_bytes=8589934592,poll_seconds=.1):
                raw=(OUT/'plan.json').read_bytes();require(hashlib.sha256(raw).hexdigest()==expected,'Plan pin')
                outer=json.loads(raw)
                for pin in outer['pins']:
                    data=(ROOT/pin['path']).read_bytes()
                    require(len(data)==pin['size'] and hashlib.sha256(data).hexdigest()==pin['sha256'],'Source/design pin')
                p=json.loads((BASE/'component-source-plan001.json').read_text())
                a=p['architecture'];n=p['parameters'];mem=p['working_set_forecast_UNMEASURED'];archive=p['complete_future_output_envelope_UNFROZEN']
                require(a['byte_dimension']==15 and a['filter_widths']==list(range(1,7)) and a['filter_channels']==[25*w for w in range(1,7)] and a['highway_layers']==1 and a['pooled_width']==525,'One published-small structural recipe, no grid')
                q=259*15;conv=sum(25*w*(15*w+1) for w in range(1,7));high=2*(525*525+525);proj=525*256
                require(n['byte_table']==q and n['convolution_with_bias']==conv and n['highway']==high and n['projection']==proj and n['component_total']==q+conv+high+proj,'All component parameter coordinates counted')
                require(n['prospective_model_total']==28492416-12865792+n['component_total'] and n['component_parameter_tensors']==1+12+4+1 and n['prospective_model_parameter_tensors']==88-1+18,'No hidden native table or extra tensors')
                require(n['prospective_model_Muon_groups']==44+6+2+1 and n['prospective_model_AdamW_groups']==44-1+1+6+2,'Own complete groups')
                require(mem['conv_saved_and_backward']==2*1024*sum(25*w*(131-w) for w in range(1,7))*4 and mem['unfold_patches']==1024*15*sum(w*(131-w) for w in range(1,7))*4 and mem['byte_embeddings']==1024*130*15*4,'Worst full-byte graph lifetimes')
                cats=['five_parameter_copies','six_vocab_tensors','body_allowance','conv_saved_and_backward','unfold_patches','byte_embeddings','max_gradient_or_index_allowance','highway_projection_allowance','lookup_and_ids','runtime_allowance','workspace_allowance']
                require(mem['five_parameter_copies']==20*n['prospective_model_total'] and mem['six_vocab_tensors']==6*1024*50257*4 and mem['category_sum']==sum(mem[k] for k in cats) and mem['RSS_forecast']==mem['category_sum']+mem['extra_host_clone_allowance'] and mem['RSS_forecast']<4294967296 and mem['MPS_forecast']<6442450944,'Complete category sum, no measured allocator claim')
                fields=['generated_state_archives_bytes','real_four_twenty_archives_bytes','eighteen_quality_RAW_H200_bytes','serialization_overlap_bytes','all_metadata_bytes','reserve_bytes','extra_headroom_bytes']
                require(archive['generated_state_archives_bytes']==6*536870912+3*134217728 and archive['real_four_twenty_archives_bytes']==6*(536870912+134217728) and archive['eighteen_quality_RAW_H200_bytes']==18*(536870912+134217728) and sum(archive[k] for k in fields)==archive['entry_target_bytes'],'Full prospective retained-output high-water, not panel admission')
                result=json.loads((BASE/'vocabulary-bytes001/result.json').read_text())
                lengths=result['length_histogram']
                require(all(count<=256**int(length) for length,count in lengths.items()) and lengths['1']==256,'Exact unique length-code domain capacity')
                require(p['control']['round_function_calls']==8*50256 and p['control']['no_rng_or_code_generation_now'] is True,'Bounded prospective Feistel code work, no role drawn')
                require(p['no_Torch_model_RNG_or_numerical_execution_admission'] is True and mem['not_execution_admission'] is True and a['state_cache_optimizer_model_adapters'] is False,'Minimal source-only scope')
                write('review.json',dict(status='complete_independent_component_SOURCE_plan_review_only',accepted=True,component_parameters=n['component_total'],prospective_model_parameters=n['prospective_model_total'],RSS_forecast_UNMEASURED=mem['RSS_forecast'],future_retained_output_envelope_UNFROZEN=archive['entry_target_bytes'],no_component_import_or_execution=True,scope='Admits one minimal component math SOURCE only; no functional proof or scientific run'))
            write('STOP',dict(status='complete_original_source_plan_review',no_retry=True,no_replay=True))
        except BaseException as exc:
            write('STOP',dict(status='closed_failed_original_source_plan_review',error=repr(exc),no_retry=True,no_replay=True));raise
if __name__=='__main__':main(sys.argv[1])
