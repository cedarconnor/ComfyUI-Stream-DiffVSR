"""
Stream-DiffVSR Core Library
===========================

Core implementation of Stream-DiffVSR video super-resolution pipeline.

This module provides the inference pipeline, state management, and model
wrappers for 4x video upscaling with auto-regressive temporal guidance.
"""

from .state import StreamDiffVSRState
from .compat import check_dependencies

__all__ = [
    "StreamDiffVSRState",
    "check_dependencies",
]

# Run dependency checks on import
check_dependencies()
