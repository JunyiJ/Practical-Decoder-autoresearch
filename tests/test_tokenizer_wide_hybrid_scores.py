"""Small meta-only checkpoint fixtures; no weight values, forwards or training."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from src.models.gpt import GPT
from src.research.protocol import file_digest
from src.research.tokenizer_wide_hybrid_config import rank_config
from src.research.tokenizer_wide_hybrid_scores import review_trial


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2)+'\n')


def fixture(root, family='hybrid', divisor=4):
    root.mkdir(); cp = root/'checkpoints'; cp.mkdir()
    spec = dict(stage='confirm', family=family, seed=3511, lr=.00125, divisor=divisor, width=512, steps=1200)
    cfg = rank_config(root, family, seed=3511, device='mps', bottleneck_divisor=divisor)
    request = dict(spec=spec, config=cfg, result_path=str(root/'result.json'))
    write_json(root/'request.json', request)
    with torch.device('meta'):
        model = GPT(OmegaConf.create(cfg['model']))
    names = dict(model.named_parameters()); parameters = sum(p.numel() for p in names.values())
    contract = dict(model=cfg['model'], tokenizer={'fixture': 'synthetic-identity-no-corpus'},
                    weight_averaging=cfg['training']['weight_averaging'])
    ema = dict(spec=cfg['training']['weight_averaging'], completed_steps=1200, snapshots=41,
               parameter_names=list(names))
    base = dict(cfg=cfg, step=1200, model_math_version=2, parameter_layout_version=2, model_state=model.state_dict())
    raw = dict(base, optimizer_state={}, training_state=dict(contract=contract,
               weight_averaging_state=dict(ema, version=1, averages={'200': names})))
    torch.save(raw, cp/'raw.pt')
    avg = dict(base, tokenizer_contract=contract['tokenizer'],
               parent_checkpoint=dict(path=str(cp/'raw.pt'), sha256=file_digest(cp/'raw.pt')),
               averaged_weights=dict(ema, half_life=200, purpose='inference_only_not_optimizer_resume'))
    torch.save(avg, cp/'h200.pt')
    counts = [min(2048, 34278-i) for i in range(0, 34278, 2048)]
    score = dict(context=128, eval_batch_size=8, block_tokens=2048, target_count=34278, target_bytes=111536,
                 window_count=math.ceil(34278/128), parameters=parameters, block_target_counts=counts,
                 block_loss_sums=[5*n for n in counts], mean_ce=5., perplexity=math.exp(5),
                 bits_per_byte=5*34278/111536/math.log(2))
    result = dict(spec, status='completed', holdout_consumed=False, request_sha256=file_digest(root/'request.json'),
                  training=dict(completed_steps=1200, final_checkpoint=str(cp/'raw.pt'), weight_averaging=dict(snapshots=41)))
    for k in ('raw', 'h200'):
        result[k] = dict(score, checkpoint=str(cp/(k+'.pt')), checkpoint_sha256=file_digest(cp/(k+'.pt')))
    write_json(root/'result.json', result)
    return request, contract, result


def review(root, request, contract):
    return review_trial(root, request, file_digest(root/'request.json'), file_digest(root/'result.json'), contract)


@pytest.mark.parametrize('family', ['hybrid', 'gated_mla', 'kda'])
@pytest.mark.parametrize('divisor', [2, 4, 8])
def test_meta_only_valid_pair(tmp_path, family, divisor):
    root = tmp_path/'trial'; request, contract, result = fixture(root, family, divisor)
    hashes = {str(p): file_digest(p) for p in root.rglob('*') if p.is_file()}
    report = review(root, request, contract)
    assert report['selection_row']['ce'] == 5.
    assert report['metadata_reviewed'] and not report['tensor_values_audited'] and not report['adopted']
    assert report['target_counts'] == result['h200']['block_target_counts']
    assert hashes == {str(p): file_digest(p) for p in root.rglob('*') if p.is_file()}
    assert sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) < 512*1024


@pytest.mark.parametrize('bad,match', [('mean', 'arithmetic'), ('ppl', 'arithmetic'), ('bpb', 'arithmetic'),
    ('tail', 'coverage'), ('targets', 'coverage'), ('bytes', 'coverage'), ('parameters', 'coverage'),
    ('nan', 'loss sums'), ('seed', 'identity'), ('holdout', 'identity'), ('incomplete', 'Incomplete'),
    ('snapshots', 'Incomplete'), ('request_hash', 'identity'), ('alias', 'distinct'), ('endpoint', 'endpoint')])
def test_corrupt_score_or_identity(tmp_path, bad, match):
    root = tmp_path/'trial'; request, contract, result = fixture(root)
    if bad in ('mean', 'ppl', 'bpb'):
        result['h200'][dict(mean='mean_ce', ppl='perplexity', bpb='bits_per_byte')[bad]] += .1
    elif bad == 'tail': result['h200']['block_target_counts'][-1] -= 1
    elif bad in ('targets', 'bytes', 'parameters'):
        result['h200'][dict(targets='target_count', bytes='target_bytes', parameters='parameters')[bad]] -= 1
    elif bad == 'nan': result['raw']['block_loss_sums'][0] = float('nan')
    elif bad == 'seed': result['seed'] = 3512
    elif bad == 'holdout': result['holdout_consumed'] = True
    elif bad == 'incomplete': result['training']['completed_steps'] = 1199
    elif bad == 'snapshots': result['training']['weight_averaging']['snapshots'] = 40
    elif bad == 'request_hash': result['request_sha256'] = '0'*64
    elif bad == 'alias': result['h200']['checkpoint'] = result['raw']['checkpoint']
    elif bad == 'endpoint': result['training']['final_checkpoint'] = result['h200']['checkpoint']
    write_json(root/'result.json', result)
    with pytest.raises(ValueError, match=match): review(root, request, contract)


@pytest.mark.parametrize('bad,kind,match', [('tokenizer', 'raw', 'contract'), ('tokenizer', 'h200', 'lineage'),
    ('config', 'raw', 'config'), ('half_life', 'h200', 'half-life'), ('parent', 'h200', 'lineage'),
    ('shape', 'raw', 'shape'), ('dtype', 'h200', 'dtype'), ('names', 'raw', 'name order'),
    ('ema_shape', 'raw', 'shape'), ('ema_count', 'h200', 'EMA metadata'), ('optimizer', 'h200', 'inference-only')])
def test_corrupt_checkpoint_with_updated_file_hash(tmp_path, bad, kind, match):
    root = tmp_path/'trial'; request, contract, result = fixture(root)
    path = Path(result[kind]['checkpoint']); p = torch.load(path, map_location='meta', weights_only=False)
    if bad == 'tokenizer':
        if kind == 'raw': p['training_state']['contract']['tokenizer'] = {'wrong': True}
        else: p['tokenizer_contract'] = {'wrong': True}
    elif bad == 'config': p['cfg']['training']['seed'] = 3512
    elif bad == 'half_life': p['averaged_weights']['half_life'] = 400
    elif bad == 'parent': p['parent_checkpoint']['sha256'] = '0'*64
    elif bad == 'shape': p['model_state']['lm_head.weight'] = torch.empty(1, device='meta')
    elif bad == 'dtype': p['model_state']['lm_head.weight'] = p['model_state']['lm_head.weight'].half()
    elif bad == 'names': p['training_state']['weight_averaging_state']['parameter_names'].reverse()
    elif bad == 'ema_shape': p['training_state']['weight_averaging_state']['averages']['200']['lm_head.weight'] = torch.empty(1, device='meta')
    elif bad == 'ema_count': p['averaged_weights']['snapshots'] = 40
    elif bad == 'optimizer': p['optimizer_state'] = {}
    torch.save(p, path); result[kind]['checkpoint_sha256'] = file_digest(path)
    write_json(root/'result.json', result)
    with pytest.raises(ValueError, match=match): review(root, request, contract)


def test_independent_expected_inputs_and_file_hashes(tmp_path):
    root = tmp_path/'trial'; request, contract, result = fixture(root)
    other = copy.deepcopy(request); other['config']['training']['seed'] = 3512
    with pytest.raises(ValueError, match='expected plan'): review(root, other, contract)
    with pytest.raises(ValueError, match='hash mismatch'):
        review_trial(root, request, '0'*64, file_digest(root/'result.json'), contract)
    with (root/'checkpoints/raw.pt').open('ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError, match='Checkpoint hash'): review(root, request, contract)
