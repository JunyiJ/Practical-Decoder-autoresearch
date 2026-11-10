"""Separately identified composer-reuse execution; no frozen source edits."""
import copy
import math
import torch
from research.byte_input_model_v6.model import ByteInputModel as OriginalModel
from research.byte_input_model_v6.optimizer import ByteInputOptimizer as OriginalOptimizer, is_muon
from research.byte_input_model_v3 import contract as C
from research.byte_input_model_v6 import Spec
from research.general_rsi_composer_reuse_v1 import compose
from research.general_rsi_data_v1 import GeneralData, MANIFEST_SHA
from research.general_rsi_training_v2 import Context as OriginalContext, KIND as ORIGINAL_KIND

EXECUTION = 'general_rsi_whole_token_composer_reuse_v1'
KIND = 'general_rsi_reuse_complete_training_v1'


class ReuseModel(OriginalModel):
    @property
    def identity(self):
        return (EXECUTION, super().identity)

    def _composed_hidden(self, inputs):
        hidden = compose(self.composer, self.code_ids, self.code_lengths, inputs, True)
        hidden = hidden.reshape(inputs.shape[0], inputs.shape[1], 256)
        for block in self.blocks:
            hidden, auxiliary = block(hidden, None, 0)
            C.require(auxiliary is None, 'No block auxiliary objective')
        return self.final_norm(hidden)


class ReuseOptimizer(OriginalOptimizer):
    # Exact inherited grouping/options, with an explicit new model type admission.
    def __init__(self, model, learning_rate):
        C.require(type(model) is ReuseModel and model.spec.variant == 'raw_bytes', 'Own reuse model required')
        model.validate()
        C.require(type(learning_rate) is float and math.isfinite(learning_rate) and learning_rate > 0., 'Positive finite base LR')
        self.model = model
        self._items = model.ordered_parameters()
        muon, adam = [], []
        self._scales = []
        for name, parameter in self._items:
            use_muon = is_muon(name, parameter)
            group = dict(params=[parameter], lr=learning_rate*(.5 if use_muon else 1.), weight_decay=0.)
            (muon if use_muon else adam).append(group)
        _, _, muon_count, adam_count = C.counts(model.spec.variant)
        C.require(len(muon) == muon_count and len(adam) == adam_count, 'Full canonical optimizer grouping')
        self._optimizers = (torch.optim.Muon(muon, adjust_lr_fn='match_rms_adamw'),
                            torch.optim.AdamW(adam, betas=(.9, .999), eps=1e-6))
        self._children = self._optimizers
        self._steps = 0
        self.param_groups = [group for optimizer in self._optimizers for group in optimizer.param_groups]
        by_id = {id(parameter): name for name, parameter in self._items}
        self._bindings = tuple((group, group['params'][0], .5 if is_muon(by_id[id(group['params'][0])], group['params'][0]) else 1.) for group in self.param_groups)
        self._options = tuple({key: value for key, value in group.items() if key not in ('params', 'lr')} for group in self.param_groups)
        self.validate()


class ReuseContext(OriginalContext):
    def __init__(self, arm, lr, horizon, roles, book, binding, device='mps'):
        assert arm == 'raw_bytes' and lr in (.00125, .0025)
        assert horizon in (40, 2048, 10240, 20480) and device == 'mps'
        assert set(roles) == {'native_initializer', 'composer_initializer', 'train_sampler', 'validation_sampler'}
        assert len(set(roles.values())) == 4 and type(binding) is str and len(binding) == 64
        self.config = dict(arm=arm, lr=lr, horizon=horizon, roles=copy.deepcopy(roles), manifest_sha256=MANIFEST_SHA,
                           binding=binding, device=device, execution=EXECUTION)
        mps_rng = torch.mps.get_rng_state().clone()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(roles['native_initializer'])
                self.model = ReuseModel(Spec(arm, roles['composer_initializer'], book.identity), book).to(device)
        finally:
            torch.mps.set_rng_state(mps_rng)
        self.optimizer = ReuseOptimizer(self.model, float(lr))
        self.base_lrs = tuple(g['lr'] for g in self.optimizer.param_groups)
        self.data = GeneralData(roles['train_sampler'], roles['validation_sampler'], device)
        self.schedule = dict(kind='linear_cooldown', total_steps=horizon, min_ratio=.1, start_step=2*horizon//3)
        self.ema_start = math.ceil((2*horizon/3)/10)*10
        self.ema = {}
        self.snapshots = 0
        self.history = []

    def capture(self):
        result = super().capture()
        result['kind'] = KIND
        return result

    def restore(self, payload):
        assert payload['kind'] == KIND and payload['config']['execution'] == EXECUTION
        adapted = dict(payload, kind=ORIGINAL_KIND)
        return super().restore(adapted)
