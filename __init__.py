"""
ComfyUI-Stream-DiffVSR
======================

ComfyUI custom nodes for Stream-DiffVSR video super-resolution.

This node pack wraps the Stream-DiffVSR model for low-latency
video super-resolution with auto-regressive temporal guidance.

License: Apache-2.0
Based on: https://github.com/jamichss/Stream-DiffVSR

Copyright 2025 Stream-DiffVSR Authors (upstream model)
Copyright 2026 Cedar (ComfyUI integration)
"""

from .nodes import (
    StreamDiffVSR_Loader,
    StreamDiffVSR_Upscale,
    StreamDiffVSR_ProcessFrame,
    StreamDiffVSR_CreateState,
    StreamDiffVSR_ExtractState,
)

# Node registration for ComfyUI
NODE_CLASS_MAPPINGS = {
    "StreamDiffVSR_Loader": StreamDiffVSR_Loader,
    "StreamDiffVSR_Upscale": StreamDiffVSR_Upscale,
    "StreamDiffVSR_ProcessFrame": StreamDiffVSR_ProcessFrame,
    "StreamDiffVSR_CreateState": StreamDiffVSR_CreateState,
    "StreamDiffVSR_ExtractState": StreamDiffVSR_ExtractState,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StreamDiffVSR_Loader": "Load Stream-DiffVSR Model",
    "StreamDiffVSR_Upscale": "Stream-DiffVSR Upscale",
    "StreamDiffVSR_ProcessFrame": "Process Single Frame (Advanced)",
    "StreamDiffVSR_CreateState": "Create Empty State",
    "StreamDiffVSR_ExtractState": "Extract State Info",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

# Version info
__version__ = "1.0.0"
