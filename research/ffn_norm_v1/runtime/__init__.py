"""Reuse frozen modules under the new isolated namespace."""
from importlib import import_module
__path__.extend(import_module('research.mla_score_v1.runtime').__path__)
