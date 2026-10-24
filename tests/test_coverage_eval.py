import math
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from src.research.coverage_eval import score_validation, paired_blocks

class KnownLogits(torch.nn.Module):
    def __init__(self):
        super().__init__();self.anchor=torch.nn.Parameter(torch.zeros(()));self.cfg=SimpleNamespace(block_size=512);self.seen=[]
    def forward(self,x):
        self.seen.extend(x.cpu().tolist())
        return torch.stack([x.float()/7,-x.float()/7],dim=-1),None

@pytest.mark.parametrize('length',[2,129,130,1031,4200])
@pytest.mark.parametrize('batch',[1,8])
def test_every_target_exactly_once_and_tail_token_weighted(length,batch):
    torch.set_num_threads(1);data=torch.arange(length)%2;model=KnownLogits()
    out=score_validation(model,data,128,batch,2048)
    logits=torch.stack([data[:-1].float()/7,-data[:-1].float()/7],dim=-1)
    expected=torch.nn.functional.cross_entropy(logits,data[1:],reduction='none').double().numpy()
    assert out['target_count']==length-1
    assert out['mean_ce']==pytest.approx(expected.mean(),abs=1e-7)
    assert sum(out['block_target_counts'])==length-1
    assert sum(out['block_loss_sums'])==pytest.approx(expected.sum(),abs=1e-5)
    assert [x for seq in model.seen for x in seq]==data[:-1].tolist()

def test_paired_bootstrap_identity_offset_and_panel_rejection():
    model=KnownLogits();data=torch.arange(10000)%2
    a=score_validation(model,data,128,8,2048)
    same=paired_blocks({1:a,2:a},{1:a,2:a})
    assert same['conditional_text_bootstrap_95pct']==[0.,0.]
    b={**a,'mean_ce':a['mean_ce']+.2,'block_loss_sums':(np.array(a['block_loss_sums'])+.2*np.array(a['block_target_counts'])).tolist()}
    for block in [2048,4096,8192]:
        result=paired_blocks({1:b,2:b},{1:a,2:a},block)
        assert result['mean_delta']==pytest.approx(.2)
        assert result['conditional_text_bootstrap_95pct']==pytest.approx([.2,.2])
    with pytest.raises(ValueError):paired_blocks({1:a},{2:a})
    with pytest.raises(ValueError):paired_blocks({1:a},{1:{**a,'context':256}})

def test_context_does_not_exceed_saved_capacity():
    model=KnownLogits();model.cfg.block_size=128
    with pytest.raises(ValueError):score_validation(model,torch.arange(1000)%2,256)


def test_worker_result_rejects_missing_targets_and_nonfinite_losses():
    from src.research.coverage_eval import validate_evaluation
    import copy
    value=score_validation(KnownLogits(),torch.arange(4200)%2,128,8,2048)
    validate_evaluation(value,4199)
    for field,new in [('mean_ce',float('nan')),('target_count',4198),('window_count',32),('eval_batch_size',16)]:
        bad=copy.deepcopy(value);bad[field]=new
        with pytest.raises(ValueError):validate_evaluation(bad,4199)
    bad=copy.deepcopy(value);bad['block_loss_sums'][0]=float('inf')
    with pytest.raises(ValueError):validate_evaluation(bad,4199)
    bad=copy.deepcopy(value);bad['block_target_counts'][-1]-=1
    with pytest.raises(ValueError):validate_evaluation(bad,4199)


def test_checkpoint_integrity_guard_precedes_deserialization(tmp_path):
    from src.research.coverage_eval import evaluate_job
    p=tmp_path/'corrupt.pt';p.write_bytes(b'not a checkpoint')
    with pytest.raises(ValueError,match='Checkpoint changed'):
        evaluate_job({'checkpoint':str(p),'checkpoint_sha256':'0'*64})
