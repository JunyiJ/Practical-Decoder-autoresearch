"""Own native readout-distribution model, separate from closed tiny proofs."""
from .contract import model_config, context_seed
from .model import GPT, load_weights, Distribution
from .optimizer import build_optimizer
