"""Independent TOP model-math SOURCE ONLY; import only under future admission."""
from .contract import Spec
from .model import TopModel, PrimaryLogits, TrainingResult
from .optimizer import build_optimizer

__all__ = ('Spec', 'TopModel', 'PrimaryLogits', 'TrainingResult', 'build_optimizer')
