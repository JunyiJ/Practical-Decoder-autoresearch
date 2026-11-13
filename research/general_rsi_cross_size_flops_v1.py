"""Static operator estimate from bound tensor inventory and executed dense math."""
import hashlib,inspect,json
from pathlib import Path
import torch
import torch.optim._muon as muon
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def estimate():
    allocation=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-boundary-001/allocation.json'
    inv=json.loads(allocation.read_text())['baseline_parameter_inventory']
    block={n:r['shape'] for n,r in inv['native'].items() if n.startswith('transformer.h.0.') and len(r['shape'])==2}
    assert len(block)==11
    composer={n:r['shape'] for n,r in inv['raw_bytes'].items() if len(r['shape'])==2 and (n.startswith('composer.filters.') or n in ('composer.highway_weight','composer.transform_weight','composer.projection_weight'))}
    assert len(composer)==9
    def ns(shapes):
        return sum(5*(4*min(s)**2*max(s)+2*min(s)**3) for s in shapes)
    T=1024;B=8;S=128;heads=8;head=32
    linear=2*T*sum(a*b for a,b in block.values())
    # Dense masked attention executes the full square, including masked entries.
    attention=2*B*heads*S*S*(2*head+head)
    head_forward=2*T*50257*256
    # Composer uses code_ids width130 for EVERY occurrence, not unique-token or
    # variable-length packing; masked wide windows still execute their matmuls.
    composer_forward=2*T*(sum((131-w)*15*w*c for w,c in zip(range(1,7),(25,50,75,100,125,150)))+2*525*525+256*525)
    rows=[]
    for d in (4,8,16):
        for arm in ('native','raw_bytes'):
            f=dict(body_linear=d*linear,attention_dense=d*attention,output_head=head_forward,composer=0 if arm=='native' else composer_forward)
            matrix_train=3*sum(f.values());optimizer=d*ns(block.values())+(ns(composer.values()) if arm=='raw_bytes' else 0)
            rows.append(dict(depth=d,arm=arm,forward_matrix_flops=f,estimated_forward_backward_matrix_flops_per_update=matrix_train,muon_newton_schulz_matrix_flops_per_update=optimizer,estimated_accounted_flops_per_update=matrix_train+optimizer,estimated_accounted_flops_per_run=(matrix_train+optimizer)*10240))
    paths=[allocation,ROOT/'research/general_rsi_cross_size_v1/model.py',ROOT/'research/general_rsi_cross_size_v1/optimizer.py',ROOT/'research/byte_composer_v2.py',ROOT/'research/block_norm_v1/runtime/attention/mla.py',Path(inspect.getfile(muon)),Path(__file__)]
    return dict(policy='dense_matrix_fwd_bwd_plus_muon_v1',sources={str(p):sha(p) for p in paths},torch_version=torch.__version__,per_block_matrix_shapes=block,composer_matrix_shapes=composer,rows=rows,assumptions=['One multiply-add=2FLOPs; forward+two reverse matmuls=3xforward for trainable dense inputs and weights.','1024targets as8x128; full dense attention scores64wide and values32wide.','Native input lookup is not a dense vocabulary matmul. Composer processes all1024occurrences at padded130positions.','Muon5Newton-Schulz steps: each4*m*m*n+2*m*m*m FLOPs after orienting m<=n; internalBF16 unchanged.','Raw/H200 averaging, norm/activation/bias/softmax/CE/embedding scatter/clipping/AdamW/decay/reductions/copies and data/validation/generation/checkpoint work are uncounted. This is an explicitly partial operator estimate, not total hardware instructions or a6NT proxy.','Curves use cumulative successful updates times this estimate; failures/replays separately charged. Mac synchronized time includes the uncounted update work.'],test_opened=False)
if __name__=='__main__':print(json.dumps(estimate(),indent=2))
