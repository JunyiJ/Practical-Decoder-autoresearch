"""Own bounded two-token training objective; no model, sampler or scorer hooks."""
import torch
from torch.nn import functional as F

ARMS = ('native_next', 'duplicate_next', 'future_two')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def align(sequence, vocab):
    """Explicit right-two context, without wrap/padding; caller supplies one split."""
    require(type(vocab) is int and 2 <= vocab <= 50257, 'Bounded integer vocabulary required')
    require(type(sequence) is torch.Tensor, 'Plain sequence tensor required')
    require(sequence.layout == torch.strided and sequence.device.type in ('cpu', 'mps')
            and sequence.dtype == torch.int64 and sequence.ndim == 2,
            'Dense CPU/MPS int64 sequence required')
    require(1 <= sequence.shape[0] <= 8 and 3 <= sequence.shape[1] <= 130,
            'Bounded batch and explicit two right targets required')
    require(bool(((sequence >= 0) & (sequence < vocab)).all()), 'In-range IDs required')
    return tuple(sequence[:, offset:sequence.shape[1]-2+offset].clone()
                 for offset in (0, 1, 2))


def objective(arm, primary, auxiliary, sequence):
    """Primary CE is always reported separately; auxiliary arms average two CEs.

    duplicate_next and future_two have identical loss weights and logit interfaces;
    only the auxiliary target offset changes. This function cannot establish that
    supplied logits came from a causal model or independently parameterized heads.
    """
    require(type(arm) is str and arm in ARMS, 'Exact own objective arm required')
    require(type(primary) is torch.Tensor, 'Plain primary logits required')
    require(primary.layout == torch.strided and primary.device.type in ('cpu', 'mps')
            and primary.ndim == 3 and primary.dtype in (torch.float32, torch.float64),
            'Dense bounded CPU64/CPU32/MPS32 logits required')
    require(primary.device.type != 'mps' or primary.dtype == torch.float32, 'MPS FP32 required')
    require(not torch.is_autocast_enabled('cpu') and not torch.is_autocast_enabled('mps'),
            'Autocast forbidden')
    require(type(sequence) is torch.Tensor and sequence.device == primary.device,
            'Same-device explicit sequence required')
    vocab = primary.shape[-1]
    x, y1, y2 = align(sequence, vocab)
    require(primary.shape[:2] == x.shape and bool(torch.isfinite(primary).all()),
            'Finite primary logits aligned to every input required')
    if arm == 'native_next':
        require(auxiliary is None, 'Native arm forbids an auxiliary head')
    else:
        require(type(auxiliary) is torch.Tensor and auxiliary.layout == torch.strided
                and auxiliary.shape == primary.shape and auxiliary.device == primary.device
                and auxiliary.dtype == primary.dtype and bool(torch.isfinite(auxiliary).all()),
                'Finite same-shape/dtype/device auxiliary logits required')
    main = F.cross_entropy(primary.reshape(-1, vocab), y1.reshape(-1), reduction='mean')
    aux = None
    if arm != 'native_next':
        target = y1 if arm == 'duplicate_next' else y2
        aux = F.cross_entropy(auxiliary.reshape(-1, vocab), target.reshape(-1), reduction='mean')
    total = main if aux is None else (main + aux) * .5
    for value in (main, aux, total):
        if value is not None:
            require(value.ndim == 0 and value.dtype == primary.dtype
                    and bool(torch.isfinite(value)) and bool(value >= 0), 'Finite genuine CE required')
    return dict(primary_ce=main, auxiliary_ce=aux, training_loss=total)
