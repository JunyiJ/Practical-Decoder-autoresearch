"""Reuse unchanged frozen modules under the isolated local-mixing namespace."""
from importlib import import_module
__path__.extend(import_module('research.ffn_norm_v1.runtime.train').__path__)
