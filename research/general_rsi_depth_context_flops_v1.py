"""Partial dense-matrix accounting for both frozen native depth/context shapes."""
import hashlib,inspect,json
from pathlib import Path
import torch
import torch.optim._muon as muon
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def estimate():
    allocation=ROOT/'research/tokenizer-evidence/general-rsi-cross-size-boundary-001/allocation.json'
    inv=json.loads(allocation.read_text())['baseline_parameter_inventory']['native']
    block={n:r['shape'] for n,r in inv.items() if n.startswith('transformer.h.0.') and len(r['shape'])==2};assert len(block)==11
    T=1024;heads=8;head=32
    linear=2*T*sum(a*b for a,b in block.values());output=2*T*50257*256
    muon_block=sum(5*(4*min(s)**2*max(s)+2*min(s)**3) for s in block.values())
    rows=[]
    for depth in (4,16):
        for context in (128,256):
            batch=T//context;attention=2*batch*heads*context**2*(2*head+head)
            forward=dict(body_linear=depth*linear,attention_dense=depth*attention,output_head=output)
            matrix=3*sum(forward.values());optimizer=depth*muon_block
            rows.append(dict(depth=depth,context=context,forward_matrix_flops=forward,estimated_forward_backward_matrix_flops_per_update=matrix,muon_newton_schulz_matrix_flops_per_update=optimizer,estimated_accounted_flops_per_update=matrix+optimizer))
    paths=[allocation,ROOT/'research/general_rsi_depth_context_v1/model.py',ROOT/'research/general_rsi_depth_context_v1/optimizer.py',ROOT/'research/general_rsi_context_length_v1/runtime/attention/__init__.py',ROOT/'research/block_norm_v1/runtime/attention/mla.py',Path(inspect.getfile(muon)),Path(__file__)]
    return dict(policy='dense_matrix_fwd_bwd_plus_muon_v1',sources={str(p):sha(p) for p in paths},torch_version=torch.__version__,per_block_matrix_shapes=block,rows=rows,assumptions=['Multiply-add=2FLOPs; forward+two reverse matmuls=3xforward.','1024targets as8x128 or4x256; full masked attention scores64wide and values32wide. Native input lookup is not a dense vocabulary matmul.','Muon5Newton-Schulz steps each4*m*m*n+2*m*m*m after orienting m<=n; internalBF16 unchanged.','EMA,norm/activation/bias/softmax/CE/embedding scatter/clipping/AdamW/decay/reductions/copies and data/validation/generation/checkpoint work excluded. Partial operator estimate, not total hardware telemetry or6NT proxy.','Multiply by actual successful updates for curves; charge failures/replays separately. Synchronized Mac time includes uncounted update operations.'],test_opened=False)
if __name__=='__main__':print(json.dumps(estimate(),indent=2))
