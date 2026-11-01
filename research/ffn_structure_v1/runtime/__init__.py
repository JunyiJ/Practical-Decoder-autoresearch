"""Narrow runtime overlay; unchanged modules resolve from the frozen norm runtime."""
from importlib import import_module
from pathlib import Path
__path__.append(str(Path(import_module('research.block_norm_v1.runtime').__file__).parent))
