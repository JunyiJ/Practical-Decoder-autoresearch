"""Resolve unchanged predecessor modules in this isolated runtime."""
from importlib import import_module
__path__.extend(import_module('research.local_mix_v1.runtime.moe').__path__)
