"""Config-driven parsers for UDIVA spotting annotations."""

from .config import ParserConfig, load_config
from .pipeline import build_outputs, run_config

__all__ = [
    "ParserConfig",
    "build_outputs",
    "load_config",
    "run_config",
]
