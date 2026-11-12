"""Receipt-bound starting weights on the qualified no-decay continuation context."""
import torch
from pathlib import Path
from research.general_rsi_continuation_decay_v1 import Context as Base
from research.general_rsi_training_v2 import sha

class Context(Base):
    def __init__(self,endpoint,starting_weights,budget_ns,half_life_ns,roles,pseudo_seed,parent,binding):
        assert endpoint in ['raw','time_ema']
        super().__init__('none',budget_ns,half_life_ns,roles,pseudo_seed,parent,binding)
        path=Path(starting_weights['path'])
        assert not path.is_symlink() and path.stat().st_size==starting_weights['size'] and sha(path)==starting_weights['sha256']
        payload=torch.load(path,map_location='cpu',weights_only=True)
        assert payload['kind']=='general_rsi_equal_time_inference_v1' and payload['endpoint']==endpoint
        assert payload['binding']==parent['binding'] and payload['slot']['slot']==parent['slot'] and payload['slot']['arm']=='native'
        assert payload['model_identity']==self.model.identity and payload['immutable_buffers']==self.model._own_expected_buffer_hashes
        items=dict(self.model.ordered_parameters());assert list(items)==list(payload['parameters'])
        with torch.no_grad():
            for name,p in items.items():
                saved=payload['parameters'][name]
                assert saved.dtype==torch.float32 and saved.shape==p.shape and bool(torch.isfinite(saved).all())
                p.copy_(saved)
        assert all(torch.equal(p.detach().cpu(),payload['parameters'][n]) for n,p in items.items())
        assert self.step==0 and not self.history and not self.ema and self.pseudo.draws==0
        assert all(not child.state for child in self.optimizer._optimizers)
        assert sha(path)==starting_weights['sha256'];self.optimizer.validate()
        self.config.update(starting_endpoint=endpoint,starting_weights=dict(starting_weights))
        self.start_loader_check=dict(endpoint=endpoint,receipt=dict(starting_weights),exact_parameter_tensors=len(items),
            model_buffer_identity_checked=True,optimizer_pristine=True,training_updates=0)
