"""Reuse unchanged predecessor modules under the explicit new namespace."""
from importlib import import_module
__path__.extend(import_module('research.branch_scale_model_v1.runtime.attention').__path__)
