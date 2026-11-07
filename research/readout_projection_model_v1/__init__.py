"""Source-prepared single-projection model; numerical admission is separate."""
from .contract import model_config, projection_seed
from .model import GPT, load_weights
from .optimizer import build_optimizer
