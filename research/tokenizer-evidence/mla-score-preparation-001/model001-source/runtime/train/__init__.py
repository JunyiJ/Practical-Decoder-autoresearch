"""Reuse unchanged FFN runtime modules under this isolated namespace."""
from importlib import import_module
__path__.extend(import_module('research.ffn_structure_v1.runtime.train').__path__)
