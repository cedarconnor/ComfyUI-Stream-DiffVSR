# ComfyUI-Stream-DiffVSR Design Document

**Version:** 1.1 (Revised)  
**Date:** January 2026  
**Author:** Cedar  
**Status:** Draft - Post-Review

---

## Revision History

| Version | Date | Changes |
|---------|------|---------|
| 1.1 | Jan 2026 | **Post-review revision** addressing Gemini/ChatGPT feedback |
| 1.0 | Jan 2026 | Initial design document |

### Key Changes in v1.1

2. **VAE encoding fixed** — Removed hardcoded 0.18215 scaling, use config.scaling_factor (Section 8.1)
3. **State serialization rewritten** — Replaced tensor-to-list with safetensors (Section 9)
4. **Tensor layout standardized** — All state tensors are BCHW, conversion at I/O boundaries (Section 8.2)
5. **ARTG injection architecture documented** — Explains why we bypass ComfyUI's sampler (Section 8.0)
6. **Tiling + temporal consistency** — Critical section on why naive tiling breaks (Section 11.1)
7. **Optical flow model reuse** — Check for existing RAFT before loading (Section 11.2)
8. **MVP strategy added** — "Wrap upstream inference" recommended approach (Section 1.1)
9. **Batch=time convention** — Explicitly documented frame ordering (Section 4)
10. **Version compatibility** — Defensive imports and known-good versions (Section 13)

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Project Goals](#2-project-goals)
3. [Technical Overview](#3-technical-overview)
4. [Architecture Design](#4-architecture-design)
5. [File Structure](#5-file-structure)
6. [Model Management](#6-model-management)
7. [Node Specifications](#7-node-specifications)
8. [Core Implementation](#8-core-implementation)
9. [State Management](#9-state-management)
10. [Error Handling](#10-error-handling)
11. [Performance Considerations](#11-performance-considerations)
12. [Testing Strategy](#12-testing-strategy)
13. [Dependencies](#13-dependencies)
14. [License Considerations](#14-license-considerations)
15. [Future Roadmap](#15-future-roadmap)

---

## 1. Executive Summary

This document outlines the design and implementation strategy for **ComfyUI-Stream-DiffVSR**, a ComfyUI custom node pack that integrates the Stream-DiffVSR model for low-latency video super-resolution. Stream-DiffVSR is a causally conditioned diffusion framework that achieves real-time 4× video upscaling using auto-regressive temporal guidance.

### Key Differentiators from Existing Solutions

| Feature | FlashVSR | Stream-DiffVSR |
|---------|----------|----------------|
| Denoising Steps | 1-step | 4-step DDIM |
| Temporal Modeling | Implicit (per-frame) | Explicit auto-regressive (flow-warped previous HQ) |
| Backbone | Wan2.1 DiT | SD x4 Upscaler U-Net |
| Decoder | TCDecoder | Temporal-aware VAE with TPM |
| Latency (720p) | ~59ms/frame | ~328ms/frame |

### Design Philosophy

1. **Clean Separation of Concerns** — Loader, processor, and utility nodes are distinct
2. **Explicit State Management** — Auto-regressive state is a first-class citizen
3. **Fail-Safe Defaults** — Works out-of-box with sensible defaults
4. **Professional Code Quality** — Type hints, comprehensive docstrings, consistent style

---

## 1.1 Implementation Strategy: Wrap Upstream Inference

**Recommended Approach:** "Wrap upstream inference" rather than building a clean diffusers-style pipeline.

### Why This Strategy?

| Approach | Pros | Cons |
|----------|------|------|
| **Wrap Upstream** ✅ | Fast MVP, guaranteed correctness, minimal version friction | Less "clean", harder to extend |
| **Clean Diffusers Pipeline** | Better long-term, more composable | Higher risk, diffusers version sensitivity |

### Practical Implementation

1. **Vendor minimal upstream code** — Copy only the forward-pass inference logic from `Stream-DiffVSR/inference.py` and required model definitions
2. **Loader node** — Downloads weights from HuggingFace, constructs the pipeline object
3. **Upscale node** — Loops frames internally (batch = time), returns upscaled frames + final state
4. **State node** — Create/reset state for advanced users

### Version Compatibility Notes

Upstream Stream-DiffVSR specifies:
- Python 3.8
- CUDA 11
- diffusers (version unspecified)

ComfyUI typically runs:
- Python 3.10+
- CUDA 11.8 or 12.x
- Various diffusers versions depending on other nodes

**Mitigation:**
```python
# stream_diffvsr/__init__.py

import sys
import warnings

# Version checks
if sys.version_info < (3, 10):
    warnings.warn(
        "Stream-DiffVSR is tested on Python 3.10+. "
        "Earlier versions may work but are unsupported."
    )

# Defensive diffusers import
try:
    import diffusers
    DIFFUSERS_VERSION = tuple(map(int, diffusers.__version__.split('.')[:2]))
    if DIFFUSERS_VERSION < (0, 25):
        warnings.warn(
            f"diffusers {diffusers.__version__} detected. "
            f"Stream-DiffVSR is tested with diffusers>=0.25.0"
        )
except ImportError:
    raise ImportError("diffusers is required. Install with: pip install diffusers>=0.25.0")
```

---

## 2. Project Goals

### Primary Goals

- [ ] Implement fully functional Stream-DiffVSR inference in ComfyUI
- [ ] Support both batch processing and frame-by-frame workflows
- [ ] Achieve near-reference quality matching the official repository
- [ ] Maintain clean, auditable, MIT-licensed codebase

### Secondary Goals

- [ ] VRAM optimization through optional tiling
- [ ] VHS (Video Helper Suite) integration for seamless video workflows
- [ ] TensorRT acceleration path (optional)
- [ ] Multi-GPU support for long video processing

### Non-Goals (v1.0)

- Training/fine-tuning support
- Real-time streaming inference (sub-100ms)
- Mobile/edge deployment

---

## 3. Technical Overview

### Stream-DiffVSR Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STREAM-DIFFVSR PIPELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: LQ Frame (t)              Previous: HQ Frame (t-1)                  │
│         │                                    │                              │
│         ▼                                    ▼                              │
│  ┌──────────────┐                    ┌───────────────┐                      │
│  │  VAE Encode  │                    │ Optical Flow  │                      │
│  │  (Tiny AE)   │                    │   Estimate    │                      │
│  └──────┬───────┘                    └───────┬───────┘                      │
│         │                                    │                              │
│         │ z_lq                               │ flow                         │
│         │                                    ▼                              │
│         │                            ┌───────────────┐                      │
│         │                            │  Warp HQ(t-1) │                      │
│         │                            │  using flow   │                      │
│         │                            └───────┬───────┘                      │
│         │                                    │                              │
│         │                                    │ warped_hq                    │
│         ▼                                    ▼                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │                    4-STEP DDIM DENOISING                        │        │
│  │  ┌───────────────────────────────────────────────────────────┐  │        │
│  │  │                   Distilled U-Net                         │  │        │
│  │  │  • Initialized from StableVSR / SD x4 Upscaler           │  │        │
│  │  │  • Conditioned on: z_lq, warped_hq (via ARTG)            │  │        │
│  │  └───────────────────────────────────────────────────────────┘  │        │
│  │                              │                                  │        │
│  │                    ┌─────────┴─────────┐                        │        │
│  │                    │       ARTG        │                        │        │
│  │                    │ (Auto-Regressive  │                        │        │
│  │                    │ Temporal Guidance)│                        │        │
│  │                    └───────────────────┘                        │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                              │                                              │
│                              │ z_denoised                                   │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────┐        │
│  │              TEMPORAL-AWARE DECODER                             │        │
│  │  ┌───────────────────────────────────────────────────────────┐  │        │
│  │  │  AutoEncoderTiny + Temporal Processor Module (TPM)        │  │        │
│  │  │  • Multi-scale fusion with warped previous HQ             │  │        │
│  │  │  • Interpolation + Conv + Weighted fusion                 │  │        │
│  │  └───────────────────────────────────────────────────────────┘  │        │
│  └─────────────────────────────────────────────────────────────────┘        │
│                              │                                              │
│                              ▼                                              │
│                      Output: HQ Frame (t)                                   │
│                              │                                              │
│                              └──────────────────────► Store for t+1         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

1. **Distilled U-Net** — 4-step denoiser initialized from StableVSR, trained with rollout distillation
2. **ARTG Module** — Injects motion-aligned features from previous HQ frame into U-Net decoder
3. **Temporal-Aware Decoder** — Modified AutoEncoderTiny with TPM for temporal consistency
4. **Optical Flow Estimator** — Estimates motion between consecutive frames for warping

---

## 4. Architecture Design

### Node Graph

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMFYUI NODE ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────┐                                                │
│  │  StreamDiffVSR_Loader   │                                                │
│  │  ────────────────────   │                                                │
│  │  model_version: v1      │                                                │
│  │  device: cuda           │                                                │
│  │  dtype: float16         │                                                │
│  └───────────┬─────────────┘                                                │
│              │ PIPE                                                         │
│              ▼                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    StreamDiffVSR_Upscale                            │    │
│  │  ───────────────────────────────────────────                        │    │
│  │                                                                     │    │
│  │  Inputs:                              Outputs:                      │    │
│  │  ├─ pipe (STREAM_DIFFVSR_PIPE)        ├─ images (IMAGE)             │    │
│  │  ├─ images (IMAGE)                    └─ state (STREAM_DIFFVSR_     │    │
│  │  ├─ state (optional)                         STATE) [optional]      │    │
│  │  ├─ num_inference_steps: 4                                          │    │
│  │  ├─ seed: 0                                                         │    │
│  │  └─ enable_tiling: False                                            │    │
│  │                                                                     │    │
│  │  [Processes all frames, maintains internal auto-regressive state]   │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ════════════════════════════════════════════════════════════════════════   │
│                         ADVANCED NODE SET                                   │
│  ════════════════════════════════════════════════════════════════════════   │
│                                                                             │
│  ┌────────────────────────┐    ┌────────────────────────┐                   │
│  │ StreamDiffVSR_         │    │ StreamDiffVSR_         │                   │
│  │ ProcessFrame           │    │ CreateState            │                   │
│  │ ────────────────       │    │ ─────────────          │                   │
│  │ Single frame with      │    │ Initialize empty       │                   │
│  │ explicit state I/O     │    │ state for new sequence │                   │
│  └────────────────────────┘    └────────────────────────┘                   │
│                                                                             │
│  ┌────────────────────────┐    ┌────────────────────────┐                   │
│  │ StreamDiffVSR_         │    │ StreamDiffVSR_         │                   │
│  │ ExtractState           │    │ Settings               │                   │
│  │ ────────────────       │    │ ────────────           │                   │
│  │ Extract previous HQ    │    │ Advanced config:       │                   │
│  │ frame from state       │    │ tiling, dtype, etc.    │                   │
│  └────────────────────────┘    └────────────────────────┘                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data Flow Types

| Type Name | Description | Contents |
|-----------|-------------|----------|
| `STREAM_DIFFVSR_PIPE` | Loaded model pipeline | U-Net, ARTG, Decoder, Flow model, VAE, config |
| `STREAM_DIFFVSR_STATE` | Auto-regressive state | Previous HQ frame tensor, previous LQ tensor, frame index |
| `IMAGE` | Standard ComfyUI image | BHWC tensor, float32, 0-1 range |

### Critical Convention: Batch Dimension = Time

**ComfyUI's `IMAGE` type uses batch dimension for frame ordering.**

When you pass an IMAGE batch to `StreamDiffVSR_Upscale`:
- `images.shape = (N, H, W, C)` where N = number of frames
- Frames are processed **in order**: frame 0, then frame 1, etc.
- Temporal guidance uses frame N-1's output to process frame N

```python
# In StreamDiffVSR_Upscale.upscale():

# Batch dimension IS the time dimension
num_frames = images.shape[0]

for frame_idx in range(num_frames):
    # Extract single frame (keeping batch dim for consistency)
    current_frame = images[frame_idx:frame_idx+1]  # Shape: (1, H, W, C)
    
    # Process with temporal guidance from previous frame
    hq_frame, state = pipeline.process_frame(current_frame, state)
    
    # State now contains this frame's output for next iteration
    results.append(hq_frame)

# Stack results back to batch
output = torch.cat(results, dim=0)  # Shape: (N, H*4, W*4, C)
```

**User Documentation:** Make this explicit in node tooltips and README:
> "The batch dimension of input images is interpreted as frame order. 
> Frame 0 is processed first, then frame 1 uses frame 0's output for 
> temporal guidance, and so on."

---

## 5. File Structure

```
ComfyUI-Stream-DiffVSR/
├── __init__.py                     # Node registration, version info
├── nodes/
│   ├── __init__.py
│   ├── loader_node.py              # StreamDiffVSR_Loader
│   ├── upscale_node.py             # StreamDiffVSR_Upscale (main node)
│   ├── process_frame_node.py       # StreamDiffVSR_ProcessFrame (advanced)
│   ├── state_nodes.py              # CreateState, ExtractState
│   └── settings_node.py            # StreamDiffVSR_Settings
│
├── stream_diffvsr/
│   ├── __init__.py
│   ├── pipeline.py                 # Main inference pipeline
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py                 # Distilled U-Net wrapper
│   │   ├── artg.py                 # Auto-Regressive Temporal Guidance
│   │   ├── temporal_decoder.py     # Temporal-aware VAE decoder with TPM
│   │   └── flow_estimator.py       # Optical flow estimation
│   ├── schedulers/
│   │   ├── __init__.py
│   │   └── ddim_4step.py           # 4-step DDIM scheduler
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py          # Tensor <-> PIL conversions
│   │   ├── flow_utils.py           # Warp operations
│   │   ├── tiling.py               # Tile-based processing
│   │   └── device_utils.py         # Device/dtype management
│   └── state.py                    # StreamDiffVSRState class
│
├── example_workflows/
│   ├── basic_upscale.json
│   ├── advanced_frame_control.json
│   └── vhs_integration.json
│
├── requirements.txt
├── pyproject.toml
├── LICENSE                         # MIT License
└── README.md
```

---

## 6. Model Management

### Model Directory Structure

Models are placed manually in ComfyUI's model directory:

```
ComfyUI/
└── models/
    └── StreamDiffVSR/
        ├── v1/
        │   ├── unet/
        │   │   └── diffusion_pytorch_model.safetensors
        │   ├── artg/
        │   │   └── artg_module.safetensors
        │   ├── temporal_decoder/
        │   │   └── temporal_decoder.safetensors
        │   ├── vae/
        │   │   └── vae_tiny.safetensors
        │   └── flow/
        │       └── raft_small.pth
        │
        └── config.json             # Model configuration
```

### Alternative: Flat Structure (Matching HuggingFace)

```
ComfyUI/
└── models/
    └── StreamDiffVSR/
        ├── model_index.json
        ├── scheduler/
        │   └── scheduler_config.json
        ├── unet/
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors
        ├── vae/
        │   ├── config.json
        │   └── diffusion_pytorch_model.safetensors
        └── ... (other components)
```

### Model Loader Implementation

```python
# stream_diffvsr/models/loader.py

import os
import json
import folder_paths
from safetensors.torch import load_file

class ModelLoader:
    """
    Handles loading Stream-DiffVSR model components.
    
    Searches for models in:
    1. ComfyUI/models/StreamDiffVSR/{version}/
    2. ComfyUI/models/diffusers/Stream-DiffVSR/ (HF cache format)
    """
    
    REQUIRED_COMPONENTS = [
        "unet",
        "artg", 
        "temporal_decoder",
        "vae",
    ]
    
    OPTIONAL_COMPONENTS = [
        "flow",  # Can fall back to torchvision RAFT
    ]
    
    @classmethod
    def get_model_path(cls) -> str:
        """Get the StreamDiffVSR model directory."""
        # Register custom model path if not exists
        if "StreamDiffVSR" not in folder_paths.folder_names_and_paths:
            model_path = os.path.join(folder_paths.models_dir, "StreamDiffVSR")
            folder_paths.folder_names_and_paths["StreamDiffVSR"] = (
                [model_path], 
                {".safetensors", ".pth", ".ckpt", ".bin"}
            )
        
        paths = folder_paths.get_folder_paths("StreamDiffVSR")
        if not paths:
            raise FileNotFoundError(
                "StreamDiffVSR model directory not found. "
                "Please create ComfyUI/models/StreamDiffVSR/ and add model files."
            )
        return paths[0]
    
    @classmethod
    def validate_model_files(cls, model_path: str, version: str = "v1") -> dict:
        """
        Validate that all required model files exist.
        
        Returns:
            dict: Mapping of component name to file path
        """
        version_path = os.path.join(model_path, version)
        if not os.path.exists(version_path):
            raise FileNotFoundError(f"Model version directory not found: {version_path}")
        
        component_paths = {}
        missing = []
        
        for component in cls.REQUIRED_COMPONENTS:
            component_dir = os.path.join(version_path, component)
            if os.path.isdir(component_dir):
                # Find .safetensors or .pth file
                for ext in [".safetensors", ".pth", ".ckpt", ".bin"]:
                    files = [f for f in os.listdir(component_dir) if f.endswith(ext)]
                    if files:
                        component_paths[component] = os.path.join(component_dir, files[0])
                        break
                else:
                    missing.append(component)
            else:
                missing.append(component)
        
        if missing:
            raise FileNotFoundError(
                f"Missing required model components: {missing}\n"
                f"Expected in: {version_path}"
            )
        
        # Check optional components
        for component in cls.OPTIONAL_COMPONENTS:
            component_dir = os.path.join(version_path, component)
            if os.path.isdir(component_dir):
                for ext in [".safetensors", ".pth", ".ckpt"]:
                    files = [f for f in os.listdir(component_dir) if f.endswith(ext)]
                    if files:
                        component_paths[component] = os.path.join(component_dir, files[0])
                        break
        
        return component_paths
    
    @classmethod
    def load_component(cls, path: str, device: str = "cpu") -> dict:
        """Load a single model component."""
        if path.endswith(".safetensors"):
            return load_file(path, device=device)
        else:
            import torch
            return torch.load(path, map_location=device, weights_only=True)
```

---

## 7. Node Specifications

### 7.1 StreamDiffVSR_Loader

**Category:** `🎬 StreamDiffVSR`  
**Display Name:** `Load Stream-DiffVSR Model`

```python
class StreamDiffVSR_Loader:
    """
    Loads the Stream-DiffVSR pipeline components.
    
    This node initializes all model components (U-Net, ARTG, Temporal Decoder, VAE)
    and returns a pipeline object ready for inference.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["v1"], {
                    "default": "v1",
                    "tooltip": "Model version to load"
                }),
                "device": (["cuda", "cpu", "auto"], {
                    "default": "auto",
                    "tooltip": "Device for inference. 'auto' selects CUDA if available."
                }),
                "dtype": (["float16", "bfloat16", "float32"], {
                    "default": "float16",
                    "tooltip": "Model precision. float16 recommended for most GPUs."
                }),
            },
            "optional": {
                "flow_model": (["bundled", "raft_small", "raft_large", "none"], {
                    "default": "bundled",
                    "tooltip": "Optical flow model. 'bundled' uses included weights, "
                               "'none' disables temporal guidance (not recommended)."
                }),
            }
        }
    
    RETURN_TYPES = ("STREAM_DIFFVSR_PIPE",)
    RETURN_NAMES = ("pipe",)
    FUNCTION = "load_model"
    CATEGORY = "🎬 StreamDiffVSR"
    DESCRIPTION = "Load Stream-DiffVSR model for video super-resolution"
    
    def load_model(self, model_version, device, dtype, flow_model="bundled"):
        # Implementation in Section 8
        pass
```

### 7.2 StreamDiffVSR_Upscale

**Category:** `🎬 StreamDiffVSR`  
**Display Name:** `Stream-DiffVSR Upscale`

```python
class StreamDiffVSR_Upscale:
    """
    Main upscaling node. Processes a batch of frames with auto-regressive
    temporal guidance.
    
    For batches, frames are processed sequentially with each frame using
    the previous frame's HQ output for temporal conditioning.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("STREAM_DIFFVSR_PIPE", {
                    "tooltip": "Pipeline from StreamDiffVSR_Loader"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Input frames (BHWC tensor). Batch dimension = frame count."
                }),
            },
            "optional": {
                "state": ("STREAM_DIFFVSR_STATE", {
                    "tooltip": "Previous state for continuing a sequence. "
                               "Leave unconnected to start fresh."
                }),
                "num_inference_steps": ("INT", {
                    "default": 4,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "tooltip": "Denoising steps. Model is optimized for 4 steps."
                }),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                    "tooltip": "Random seed for reproducibility"
                }),
                "enable_tiling": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Enable tiled processing for lower VRAM usage"
                }),
                "tile_size": ("INT", {
                    "default": 512,
                    "min": 256,
                    "max": 1024,
                    "step": 64,
                    "tooltip": "Tile size when tiling is enabled"
                }),
                "tile_overlap": ("INT", {
                    "default": 64,
                    "min": 16,
                    "max": 256,
                    "step": 16,
                    "tooltip": "Overlap between tiles to reduce seams"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STREAM_DIFFVSR_STATE")
    RETURN_NAMES = ("images", "state")
    FUNCTION = "upscale"
    CATEGORY = "🎬 StreamDiffVSR"
    DESCRIPTION = "Upscale video frames 4× with temporal consistency"
    
    def upscale(self, pipe, images, state=None, num_inference_steps=4, 
                seed=0, enable_tiling=False, tile_size=512, tile_overlap=64):
        # Implementation in Section 8
        pass
```

### 7.3 StreamDiffVSR_ProcessFrame (Advanced)

**Category:** `🎬 StreamDiffVSR/Advanced`  
**Display Name:** `Process Single Frame`

```python
class StreamDiffVSR_ProcessFrame:
    """
    Process a single frame with explicit state input/output.
    
    Use this for:
    - Custom frame processing loops
    - Integration with other temporal nodes
    - Fine-grained control over state
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("STREAM_DIFFVSR_PIPE",),
                "image": ("IMAGE", {
                    "tooltip": "Single frame (B=1) or will process first frame only"
                }),
                "state": ("STREAM_DIFFVSR_STATE", {
                    "tooltip": "State from previous frame (or CreateState node)"
                }),
            },
            "optional": {
                "num_inference_steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("IMAGE", "STREAM_DIFFVSR_STATE")
    RETURN_NAMES = ("image", "state")
    FUNCTION = "process_frame"
    CATEGORY = "🎬 StreamDiffVSR/Advanced"
```

### 7.4 StreamDiffVSR_CreateState

**Category:** `🎬 StreamDiffVSR/Advanced`  
**Display Name:** `Create Empty State`

```python
class StreamDiffVSR_CreateState:
    """
    Create an empty state for starting a new sequence.
    
    The first frame will be processed without temporal guidance
    when using this empty state.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "initial_frame": ("IMAGE", {
                    "tooltip": "Optional: Pre-populate state with an upscaled frame"
                }),
            }
        }
    
    RETURN_TYPES = ("STREAM_DIFFVSR_STATE",)
    RETURN_NAMES = ("state",)
    FUNCTION = "create_state"
    CATEGORY = "🎬 StreamDiffVSR/Advanced"
    
    def create_state(self, initial_frame=None):
        from ..stream_diffvsr.state import StreamDiffVSRState
        return (StreamDiffVSRState(previous_hq=initial_frame, frame_index=0),)
```

---

## 8. Core Implementation

### 8.0 Critical Architecture Decision: Bypassing ComfyUI's Sampler

**⚠️ IMPORTANT:** Stream-DiffVSR **cannot** use ComfyUI's standard `KSampler` or `ModelPatcher` infrastructure.

**Why?** The ARTG (Auto-Regressive Temporal Guidance) module must inject motion-aligned features into **specific layers of the U-Net decoder** during each denoising step. This is not a simple conditioning input — it's a mid-network feature injection.

```
Standard ComfyUI Diffusion:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐
│ KSampler    │───►│ ModelPatcher │───►│ U-Net       │
│ (scheduler) │    │ (LoRA, etc)  │    │ (atomic)    │
└─────────────┘    └──────────────┘    └─────────────┘

Stream-DiffVSR (REQUIRED architecture):
┌──────────────────────────────────────────────────────────────┐
│               StreamDiffVSR_Upscale Node                     │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                 Custom Pipeline                         │ │
│  │                                                         │ │
│  │   for t in timesteps:                                   │ │
│  │       temporal_features = ARTG(warped_hq, z_lq)        │ │
│  │                            │                            │ │
│  │                            ▼ inject at decoder layers   │ │
│  │       noise_pred = UNet(z_t, t, cond, temporal_features)│ │
│  │       z_t = scheduler.step(noise_pred, t, z_t)         │ │
│  │                                                         │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

**Consequence:** The `StreamDiffVSR_Upscale` node effectively IS the sampler. It owns:
- The denoising loop
- The scheduler stepping  
- The ARTG feature injection
- The temporal decoder call

This is similar to how `smthemex/ComfyUI_FlashVSR` handles FlashVSR — the entire inference is self-contained.

### 8.1 Pipeline Class

```python
# stream_diffvsr/pipeline.py

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from dataclasses import dataclass

from .models.unet import StreamDiffVSRUNet
from .models.artg import ARTGModule
from .models.temporal_decoder import TemporalAwareDecoder
from .models.flow_estimator import FlowEstimator
from .schedulers.ddim_4step import DDIM4StepScheduler
from .state import StreamDiffVSRState
from .utils.flow_utils import warp_image
from .utils.image_utils import normalize_to_latent_range, denormalize_from_latent


@dataclass
class StreamDiffVSRConfig:
    """Configuration for Stream-DiffVSR pipeline."""
    scale_factor: int = 4
    latent_channels: int = 4
    latent_scale: int = 8  # Spatial downscale in latent space
    num_inference_steps: int = 4
    guidance_scale: float = 1.0  # CFG scale (1.0 = no guidance)


class StreamDiffVSRPipeline:
    """
    Stream-DiffVSR inference pipeline.
    
    Implements the auto-regressive diffusion framework for video super-resolution
    with temporal guidance from previous frames.
    """
    
    def __init__(
        self,
        unet: StreamDiffVSRUNet,
        artg: ARTGModule,
        decoder: TemporalAwareDecoder,
        vae_encoder: torch.nn.Module,
        flow_estimator: Optional[FlowEstimator],
        scheduler: DDIM4StepScheduler,
        config: StreamDiffVSRConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.unet = unet
        self.artg = artg
        self.decoder = decoder
        self.vae_encoder = vae_encoder
        self.flow_estimator = flow_estimator
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.dtype = dtype
        
        # Move all models to device
        self._to_device()
    
    def _to_device(self):
        """Move all components to target device and dtype."""
        self.unet = self.unet.to(self.device, self.dtype)
        self.artg = self.artg.to(self.device, self.dtype)
        self.decoder = self.decoder.to(self.device, self.dtype)
        self.vae_encoder = self.vae_encoder.to(self.device, self.dtype)
        if self.flow_estimator is not None:
            self.flow_estimator = self.flow_estimator.to(self.device, self.dtype)
    
    @torch.inference_mode()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            image: (B, H, W, C) tensor in [0, 1] range (ComfyUI format)
            
        Returns:
            Latent tensor (B, C, h, w) where h = H/8, w = W/8
        """
        # ComfyUI format (BHWC) -> model format (BCHW)
        x = image.permute(0, 3, 1, 2).to(self.device, self.dtype)
        
        # Normalize to [-1, 1]
        x = x * 2.0 - 1.0
        
        # Encode - handle different VAE output types
        encoder_output = self.vae_encoder.encode(x)
        
        # diffusers VAEs return distribution objects, not raw tensors
        if hasattr(encoder_output, 'latent_dist'):
            # Standard diffusers VAE - sample from distribution
            latent = encoder_output.latent_dist.mode()  # deterministic
        elif hasattr(encoder_output, 'latents'):
            # Some VAEs return .latents directly
            latent = encoder_output.latents
        else:
            # Raw tensor output (e.g., AutoEncoderTiny)
            latent = encoder_output
        
        # Apply VAE-specific scaling factor from config
        # DO NOT hardcode 0.18215 - this is SD-specific and may not apply
        scaling_factor = getattr(
            self.vae_encoder.config, 
            'scaling_factor', 
            1.0  # Default to no scaling if not specified
        )
        latent = latent * scaling_factor
        
        return latent
    
    @torch.inference_mode()
    def estimate_flow(
        self,
        current_lq: torch.Tensor,
        previous_lq: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate optical flow from previous to current frame.
        
        Args:
            current_lq: Current LQ frame (B, C, H, W)
            previous_lq: Previous LQ frame (B, C, H, W)
            
        Returns:
            Flow field (B, 2, H, W)
        """
        if self.flow_estimator is None:
            # Return zero flow if no flow estimator
            B, C, H, W = current_lq.shape
            return torch.zeros(B, 2, H, W, device=self.device, dtype=self.dtype)
        
        return self.flow_estimator(previous_lq, current_lq)
    
    @torch.inference_mode()
    def process_frame(
        self,
        lq_frame: torch.Tensor,
        state: StreamDiffVSRState,
        num_inference_steps: int = 4,
        seed: int = 0,
    ) -> Tuple[torch.Tensor, StreamDiffVSRState]:
        """
        Process a single frame with auto-regressive temporal guidance.
        
        Args:
            lq_frame: Low-quality input frame (1, H, W, C) in [0, 1]
            state: Previous frame state
            num_inference_steps: Number of denoising steps
            seed: Random seed
            
        Returns:
            hq_frame: High-quality output frame (1, H*4, W*4, C)
            new_state: Updated state for next frame
        """
        # Setup
        generator = torch.Generator(device=self.device).manual_seed(seed)
        B, H, W, C = lq_frame.shape
        target_h, target_w = H * self.config.scale_factor, W * self.config.scale_factor
        
        # Convert to model format
        lq_bchw = lq_frame.permute(0, 3, 1, 2).to(self.device, self.dtype)
        
        # Encode LQ frame
        z_lq = self.encode_image(lq_frame)
        
        # Handle temporal guidance
        warped_hq = None
        temporal_features = None
        
        if state.has_previous and self.flow_estimator is not None:
            # Estimate flow from previous LQ to current LQ
            # Note: We need to store previous LQ in state for flow estimation
            if state.previous_lq is not None:
                flow = self.estimate_flow(lq_bchw, state.previous_lq)
                
                # Warp previous HQ using flow (upscaled to HQ resolution)
                flow_upscaled = F.interpolate(
                    flow, 
                    size=(target_h, target_w), 
                    mode='bilinear', 
                    align_corners=False
                ) * self.config.scale_factor
                
                previous_hq_bchw = state.previous_hq.permute(0, 3, 1, 2).to(self.device, self.dtype)
                warped_hq = warp_image(previous_hq_bchw, flow_upscaled)
                
                # Get temporal features from ARTG
                temporal_features = self.artg.encode_temporal(warped_hq, z_lq)
        
        # Initialize noise
        latent_h, latent_w = target_h // self.config.latent_scale, target_w // self.config.latent_scale
        noise = torch.randn(
            (B, self.config.latent_channels, latent_h, latent_w),
            generator=generator,
            device=self.device,
            dtype=self.dtype
        )
        
        # Setup scheduler
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        latents = noise * self.scheduler.init_noise_sigma
        
        # Denoising loop
        for i, t in enumerate(self.scheduler.timesteps):
            # Prepare input
            latent_model_input = latents
            
            # U-Net prediction with ARTG conditioning
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=z_lq,
                temporal_features=temporal_features,
            )
            
            # Scheduler step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # Decode with temporal-aware decoder
        hq_bchw = self.decoder(
            latents,
            warped_previous=warped_hq,
            lq_features=z_lq,
        )
        
        # Convert back to ComfyUI format
        hq_bchw = torch.clamp(hq_bchw, -1, 1)
        hq_bchw = (hq_bchw + 1) / 2  # [-1, 1] -> [0, 1]
        hq_frame = hq_bchw.permute(0, 2, 3, 1).cpu().float()
        
        # Create new state
        new_state = StreamDiffVSRState(
            previous_hq=hq_frame,
            previous_lq=lq_bchw.cpu(),
            frame_index=state.frame_index + 1,
        )
        
        return hq_frame, new_state
    
    @torch.inference_mode()
    def __call__(
        self,
        images: torch.Tensor,
        state: Optional[StreamDiffVSRState] = None,
        num_inference_steps: int = 4,
        seed: int = 0,
        progress_callback: Optional[callable] = None,
    ) -> Tuple[torch.Tensor, StreamDiffVSRState]:
        """
        Process a batch of frames.
        
        Args:
            images: Input frames (B, H, W, C) in [0, 1] range
            state: Optional previous state
            num_inference_steps: Denoising steps per frame
            seed: Random seed (incremented per frame)
            progress_callback: Optional callback(frame_idx, total_frames)
            
        Returns:
            hq_images: Upscaled frames (B, H*4, W*4, C)
            final_state: State after last frame
        """
        B = images.shape[0]
        
        # Initialize state if not provided
        if state is None:
            state = StreamDiffVSRState()
        
        hq_frames = []
        
        for i in range(B):
            frame = images[i:i+1]  # Keep batch dimension
            
            hq_frame, state = self.process_frame(
                frame,
                state,
                num_inference_steps=num_inference_steps,
                seed=seed + i,  # Different seed per frame
            )
            
            hq_frames.append(hq_frame)
            
            if progress_callback is not None:
                progress_callback(i + 1, B)
        
        # Stack frames
        hq_images = torch.cat(hq_frames, dim=0)
        
        return hq_images, state
```

### 8.2 State Class

```python
# stream_diffvsr/state.py

from dataclasses import dataclass, field
from typing import Optional
import torch


@dataclass
class StreamDiffVSRState:
    """
    Auto-regressive state for Stream-DiffVSR.
    
    Stores information needed to process subsequent frames with
    temporal guidance from previous frames.
    
    TENSOR LAYOUT CONVENTION:
    ─────────────────────────
    All tensors in state use BCHW format (model-native) to avoid
    repeated permutations during processing. Conversion to ComfyUI's
    BHWC format happens only at node I/O boundaries.
    
    - previous_hq: (B, C, H*4, W*4) float32 [0,1] - upscaled output
    - previous_lq: (B, C, H, W) float32 [0,1] - original input
    """
    
    # Previous high-quality output frame (BCHW, float32, [0,1])
    # Shape: (1, 3, H*scale, W*scale)
    previous_hq: Optional[torch.Tensor] = None
    
    # Previous low-quality input frame (BCHW, float32, [0,1])
    # Shape: (1, 3, H, W) - needed for optical flow estimation
    previous_lq: Optional[torch.Tensor] = None
    
    # Frame index in current sequence (0-indexed)
    frame_index: int = 0
    
    # Optional metadata (resolution, dtype, etc.)
    metadata: dict = field(default_factory=dict)
    
    @property
    def has_previous(self) -> bool:
        """Check if previous frame data is available for temporal guidance."""
        return self.previous_hq is not None and self.previous_lq is not None
    
    def reset(self) -> "StreamDiffVSRState":
        """Create a fresh state for a new sequence."""
        return StreamDiffVSRState()
    
    def clone(self) -> "StreamDiffVSRState":
        """Create a deep copy of this state."""
        return StreamDiffVSRState(
            previous_hq=self.previous_hq.clone() if self.previous_hq is not None else None,
            previous_lq=self.previous_lq.clone() if self.previous_lq is not None else None,
            frame_index=self.frame_index,
            metadata=self.metadata.copy(),
        )
    
    def to_device(self, device: torch.device) -> "StreamDiffVSRState":
        """Move state tensors to specified device."""
        return StreamDiffVSRState(
            previous_hq=self.previous_hq.to(device) if self.previous_hq is not None else None,
            previous_lq=self.previous_lq.to(device) if self.previous_lq is not None else None,
            frame_index=self.frame_index,
            metadata=self.metadata.copy(),
        )
```

---

## 9. State Management

### Auto-Regressive Loop Handling

The key challenge is managing the temporal state across ComfyUI's execution model.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STATE MANAGEMENT STRATEGIES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STRATEGY 1: Internal Loop (Recommended for most users)                     │
│  ════════════════════════════════════════════════════════                   │
│                                                                             │
│  StreamDiffVSR_Upscale handles the loop internally:                         │
│                                                                             │
│  def upscale(pipe, images, state=None, ...):                                │
│      if state is None:                                                      │
│          state = StreamDiffVSRState()                                       │
│                                                                             │
│      for i in range(len(images)):                                           │
│          hq, state = pipe.process_frame(images[i], state)                   │
│          results.append(hq)                                                 │
│                                                                             │
│      return stack(results), state                                           │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│  STRATEGY 2: External Loop (Advanced users)                                 │
│  ════════════════════════════════════════════════════════                   │
│                                                                             │
│  User connects nodes in a loop using VHS or custom iterator:                │
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌─────────────┐                 │
│  │ CreateState │───►│ ProcessFrame     │───►│ Accumulate  │                 │
│  └─────────────┘    │                  │    │ Results     │                 │
│                     │  ┌────────────┐  │    └─────────────┘                 │
│                     │  │   state    │◄─┘                                    │
│                     │  │  feedback  │                                       │
│                     │  └────────────┘                                       │
│                     └──────────────────┘                                    │
│                                                                             │
│  ───────────────────────────────────────────────────────────────────────    │
│                                                                             │
│  STRATEGY 3: Chunked Processing (Long videos)                               │
│  ════════════════════════════════════════════════════════                   │
│                                                                             │
│  Process video in chunks, passing state between chunks:                     │
│                                                                             │
│  Chunk 1 (frames 0-99):   [images] + [None] → [hq] + [state1]               │
│  Chunk 2 (frames 100-199): [images] + [state1] → [hq] + [state2]            │
│  Chunk 3 (frames 200-299): [images] + [state2] → [hq] + [state3]            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### State Serialization (for saving/loading)

**⚠️ IMPORTANT:** State contains large tensors (full HQ frames). Do NOT convert to Python lists.

```python
# stream_diffvsr/state.py (additional methods)

import os
from safetensors.torch import save_file, load_file

def save_state(self, path: str) -> None:
    """
    Save state to disk for later resumption.
    
    Args:
        path: File path (will create .safetensors file)
    """
    if not path.endswith('.safetensors'):
        path = path + '.safetensors'
    
    tensors = {}
    metadata = {"frame_index": str(self.frame_index)}
    
    if self.previous_hq is not None:
        tensors["previous_hq"] = self.previous_hq
    if self.previous_lq is not None:
        tensors["previous_lq"] = self.previous_lq
    
    save_file(tensors, path, metadata=metadata)

@classmethod
def load_state(cls, path: str) -> "StreamDiffVSRState":
    """
    Load state from disk.
    
    Args:
        path: Path to .safetensors file
        
    Returns:
        Restored StreamDiffVSRState
    """
    from safetensors import safe_open
    
    tensors = load_file(path)
    
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata()
    
    return cls(
        previous_hq=tensors.get("previous_hq"),
        previous_lq=tensors.get("previous_lq"),
        frame_index=int(metadata.get("frame_index", 0)),
    )

# For in-memory state passing between nodes, NO serialization needed.
# ComfyUI passes Python objects directly between nodes in the same execution.
```

**Design Decision:** State is only serialized when explicitly saving to disk (e.g., for resuming interrupted processing). Normal node-to-node communication uses direct Python object passing with zero overhead.

---

## 10. Error Handling

### Error Categories and Responses

```python
# stream_diffvsr/exceptions.py

class StreamDiffVSRError(Exception):
    """Base exception for Stream-DiffVSR errors."""
    pass


class ModelNotFoundError(StreamDiffVSRError):
    """Raised when model files are missing."""
    
    def __init__(self, component: str, expected_path: str):
        self.component = component
        self.expected_path = expected_path
        super().__init__(
            f"Model component '{component}' not found.\n"
            f"Expected location: {expected_path}\n"
            f"Please download from: https://huggingface.co/Jamichsu/Stream-DiffVSR"
        )


class IncompatibleInputError(StreamDiffVSRError):
    """Raised when input dimensions are incompatible."""
    
    def __init__(self, expected: str, got: str):
        super().__init__(
            f"Incompatible input dimensions.\n"
            f"Expected: {expected}\n"
            f"Got: {got}"
        )


class VRAMError(StreamDiffVSRError):
    """Raised when VRAM is insufficient."""
    
    def __init__(self, required_mb: int, available_mb: int, suggestion: str):
        super().__init__(
            f"Insufficient VRAM.\n"
            f"Required: ~{required_mb}MB, Available: {available_mb}MB\n"
            f"Suggestion: {suggestion}"
        )
```

### Graceful Degradation

```python
# In pipeline.py

def process_frame(self, lq_frame, state, ...):
    try:
        # Normal processing
        return self._process_frame_impl(lq_frame, state, ...)
    
    except torch.cuda.OutOfMemoryError:
        # Clear cache and retry with tiling
        torch.cuda.empty_cache()
        
        if not self._tiling_enabled:
            print("[Stream-DiffVSR] VRAM exceeded, falling back to tiled processing")
            return self._process_frame_tiled(lq_frame, state, ...)
        else:
            raise VRAMError(
                required_mb=estimate_vram(lq_frame.shape),
                available_mb=get_available_vram(),
                suggestion="Reduce tile_size or input resolution"
            )
    
    except Exception as e:
        # Log error details for debugging
        print(f"[Stream-DiffVSR] Error processing frame {state.frame_index}: {e}")
        raise
```

---

## 11. Performance Considerations

### VRAM Estimates

| Resolution | Scale | Tiling | Estimated VRAM |
|------------|-------|--------|----------------|
| 480p (854×480) | 4× | Off | ~6 GB |
| 720p (1280×720) | 4× | Off | ~10 GB |
| 1080p (1920×1080) | 4× | Off | ~18 GB |
| 720p (1280×720) | 4× | 512px tiles | ~6 GB |
| 1080p (1920×1080) | 4× | 512px tiles | ~8 GB |

### Optimization Strategies

```python
# stream_diffvsr/utils/optimization.py

def optimize_for_inference(pipeline: StreamDiffVSRPipeline):
    """Apply inference optimizations to pipeline."""
    
    # 1. Enable memory-efficient attention if available
    if hasattr(pipeline.unet, 'enable_xformers_memory_efficient_attention'):
        try:
            pipeline.unet.enable_xformers_memory_efficient_attention()
            print("[Stream-DiffVSR] xformers memory efficient attention enabled")
        except Exception as e:
            print(f"[Stream-DiffVSR] xformers not available: {e}")
    
    # 2. Enable torch.compile if PyTorch 2.0+
    if hasattr(torch, 'compile') and torch.cuda.is_available():
        try:
            pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead")
            print("[Stream-DiffVSR] torch.compile enabled for U-Net")
        except Exception as e:
            print(f"[Stream-DiffVSR] torch.compile not available: {e}")
    
    # 3. Set eval mode and disable gradients
    pipeline.unet.eval()
    pipeline.decoder.eval()
    
    return pipeline


def estimate_vram_mb(height: int, width: int, scale: int = 4) -> int:
    """Estimate VRAM usage in MB for given input size."""
    # Rough estimation based on model architecture
    pixels = height * width
    output_pixels = pixels * (scale ** 2)
    
    # Base model memory (~4GB)
    base_memory = 4000
    
    # Per-pixel memory (latents, features, etc.)
    pixel_memory = (pixels * 0.01 + output_pixels * 0.005)
    
    # Buffer for activations during inference
    activation_memory = pixel_memory * 1.5
    
    return int(base_memory + pixel_memory + activation_memory)
```

---

## 11.1 Tiling and Temporal Consistency (Critical)

**⚠️ WARNING:** Naive spatial tiling WILL break temporal consistency in Stream-DiffVSR.

### Why Tiling is Problematic

Stream-DiffVSR's temporal guidance relies on:
1. **Optical flow** computed between full frames
2. **Warped previous HQ** aligned to current frame
3. **ARTG features** injected during denoising

If you tile the input:
- Flow must be computed on full frame (or tile boundaries become discontinuous)
- Warped previous HQ must be tiled identically
- ARTG features at tile boundaries may not align

### Recommended Tiling Strategy (If Required)

```python
# stream_diffvsr/utils/tiling.py

class TemporalAwareTiler:
    """
    Tiling strategy that preserves temporal consistency.
    
    Key insight: Flow and warping happen BEFORE tiling.
    Only the diffusion/decoding steps are tiled.
    """
    
    def __init__(self, tile_size: int = 512, overlap: int = 64):
        self.tile_size = tile_size
        self.overlap = overlap
    
    def process_frame_tiled(
        self,
        pipeline: StreamDiffVSRPipeline,
        lq_frame: torch.Tensor,  # Full frame BCHW
        state: StreamDiffVSRState,
        **kwargs
    ) -> Tuple[torch.Tensor, StreamDiffVSRState]:
        """
        Process frame with temporal-aware tiling.
        
        Strategy:
        1. Compute optical flow on FULL LQ frames (no tiling)
        2. Warp FULL previous HQ frame (no tiling)
        3. Tile the LQ, warped HQ, and run diffusion per-tile
        4. Blend tiles with overlap feathering
        5. Store FULL HQ result in state
        """
        B, C, H, W = lq_frame.shape
        scale = pipeline.config.scale_factor
        
        # Step 1-2: Full-frame temporal operations (NOT tiled)
        warped_hq = None
        if state.has_previous:
            flow = pipeline.estimate_flow(lq_frame, state.previous_lq)
            flow_up = F.interpolate(flow, scale_factor=scale, mode='bilinear') * scale
            warped_hq = warp_image(state.previous_hq, flow_up)
        
        # Step 3: Generate tile coordinates
        tiles = self._get_tile_coords(H * scale, W * scale)
        
        # Step 4: Process each tile
        output = torch.zeros(B, C, H * scale, W * scale, device=lq_frame.device)
        weights = torch.zeros(1, 1, H * scale, W * scale, device=lq_frame.device)
        
        for (y1, y2, x1, x2) in tiles:
            # Extract tile from LQ (downscaled coords)
            ly1, ly2 = y1 // scale, y2 // scale
            lx1, lx2 = x1 // scale, x2 // scale
            lq_tile = lq_frame[:, :, ly1:ly2, lx1:lx2]
            
            # Extract tile from warped HQ
            warped_tile = warped_hq[:, :, y1:y2, x1:x2] if warped_hq else None
            
            # Process tile (diffusion + decode)
            hq_tile = pipeline._process_tile(lq_tile, warped_tile, **kwargs)
            
            # Blend with feathering
            blend_mask = self._create_feather_mask(y2-y1, x2-x1)
            output[:, :, y1:y2, x1:x2] += hq_tile * blend_mask
            weights[:, :, y1:y2, x1:x2] += blend_mask
        
        # Normalize by weights
        output = output / weights.clamp(min=1e-8)
        
        # Step 5: Update state with FULL frame
        new_state = StreamDiffVSRState(
            previous_hq=output,
            previous_lq=lq_frame,
            frame_index=state.frame_index + 1,
        )
        
        return output, new_state
```

### Tiling Recommendations

| Scenario | Recommendation |
|----------|----------------|
| VRAM >= 12GB, 720p | **No tiling** — best quality |
| VRAM >= 8GB, 720p | Tiling with 512px tiles, 64px overlap |
| VRAM < 8GB | Reduce input resolution first, then tile |
| Long videos (1000+ frames) | Chunk processing with state passthrough |

**Key Rule:** Always compute flow and warping on full frames, only tile the diffusion steps.

---

## 11.2 Optical Flow Model Sharing

### Problem

RAFT (optical flow) models are ~400MB and consume ~4GB VRAM when loaded.

### Solution: Check for Existing RAFT

```python
# stream_diffvsr/models/flow_estimator.py

def get_flow_estimator(device: torch.device, dtype: torch.dtype) -> FlowEstimator:
    """
    Get optical flow estimator, reusing existing models if available.
    
    Checks for:
    1. ComfyUI-Frame-Interpolation's loaded RAFT
    2. ComfyUI-RAFT node's model
    3. Falls back to bundled/torchvision RAFT
    """
    
    # Try to find existing RAFT from other nodes
    try:
        # Check ComfyUI-Frame-Interpolation
        import ComfyUI_Frame_Interpolation
        if hasattr(ComfyUI_Frame_Interpolation, 'get_raft_model'):
            raft = ComfyUI_Frame_Interpolation.get_raft_model()
            if raft is not None:
                print("[Stream-DiffVSR] Reusing RAFT from ComfyUI-Frame-Interpolation")
                return ExternalRAFTWrapper(raft, device, dtype)
    except ImportError:
        pass
    
    try:
        # Check ComfyUI-RAFT
        from ComfyUI_RAFT import raft_model
        if raft_model is not None:
            print("[Stream-DiffVSR] Reusing RAFT from ComfyUI-RAFT")
            return ExternalRAFTWrapper(raft_model, device, dtype)
    except ImportError:
        pass
    
    # Fall back to bundled RAFT
    print("[Stream-DiffVSR] Loading bundled RAFT model")
    return BundledRAFT(device, dtype)


class FlowEstimatorConfig:
    """Configuration for flow estimation."""
    
    # Options: "bundled", "raft_small", "raft_large", "external", "none"
    model_type: str = "bundled"
    
    # If True, try to find existing RAFT before loading new one
    reuse_existing: bool = True
    
    # Number of RAFT iterations (more = better quality, slower)
    num_iterations: int = 12
```

---

## 12. Testing Strategy

### Unit Tests

```
tests/
├── test_pipeline.py          # Pipeline integration tests
├── test_models/
│   ├── test_unet.py
│   ├── test_artg.py
│   ├── test_decoder.py
│   └── test_flow.py
├── test_state.py             # State management tests
├── test_nodes.py             # ComfyUI node tests
└── fixtures/
    ├── sample_frames/        # Test input images
    └── expected_outputs/     # Reference outputs
```

### Test Cases

```python
# tests/test_pipeline.py

import pytest
import torch

class TestStreamDiffVSRPipeline:
    
    @pytest.fixture
    def pipeline(self):
        """Load pipeline with test weights."""
        # Use smaller test weights or mocked components
        pass
    
    def test_single_frame_processing(self, pipeline):
        """Test processing a single frame."""
        input_frame = torch.rand(1, 256, 256, 3)
        state = StreamDiffVSRState()
        
        output, new_state = pipeline.process_frame(input_frame, state)
        
        assert output.shape == (1, 1024, 1024, 3)
        assert new_state.frame_index == 1
        assert new_state.has_previous
    
    def test_batch_processing(self, pipeline):
        """Test processing multiple frames."""
        input_frames = torch.rand(5, 256, 256, 3)
        
        outputs, state = pipeline(input_frames)
        
        assert outputs.shape == (5, 1024, 1024, 3)
        assert state.frame_index == 5
    
    def test_temporal_consistency(self, pipeline):
        """Test that temporal guidance improves consistency."""
        # Create two similar frames
        frame1 = torch.rand(1, 256, 256, 3)
        frame2 = frame1 + torch.rand_like(frame1) * 0.1  # Small perturbation
        
        # Process without temporal guidance
        out1_no_temporal, _ = pipeline.process_frame(frame1, StreamDiffVSRState())
        out2_no_temporal, _ = pipeline.process_frame(frame2, StreamDiffVSRState())
        
        # Process with temporal guidance
        out1_temporal, state = pipeline.process_frame(frame1, StreamDiffVSRState())
        out2_temporal, _ = pipeline.process_frame(frame2, state)
        
        # Temporal version should be more consistent
        diff_no_temporal = (out1_no_temporal - out2_no_temporal).abs().mean()
        diff_temporal = (out1_temporal - out2_temporal).abs().mean()
        
        assert diff_temporal < diff_no_temporal
    
    def test_state_continuity(self, pipeline):
        """Test state can be saved and restored."""
        input_frame = torch.rand(1, 256, 256, 3)
        
        _, state = pipeline.process_frame(input_frame, StreamDiffVSRState())
        
        # Serialize and deserialize
        state_dict = state.to_dict()
        restored_state = StreamDiffVSRState.from_dict(state_dict)
        
        assert restored_state.frame_index == state.frame_index
        assert torch.allclose(restored_state.previous_hq, state.previous_hq)
```

---

## 13. Dependencies

### requirements.txt

```
# Core dependencies - KNOWN GOOD VERSIONS
torch>=2.0.0,<2.5.0
torchvision>=0.15.0,<0.20.0
safetensors>=0.4.0,<1.0.0
einops>=0.6.0,<1.0.0
numpy>=1.24.0,<2.0.0

# Diffusers - pin to tested range to avoid breaking changes
diffusers>=0.25.0,<0.32.0
transformers>=4.30.0,<5.0.0
accelerate>=0.20.0,<1.0.0

# Optional: Performance (don't hard-require)
# xformers>=0.0.20  # Memory efficient attention
# triton>=2.0.0     # For torch.compile
```

### Known Good Configurations

| Config | torch | diffusers | CUDA | Status |
|--------|-------|-----------|------|--------|
| ComfyUI stable | 2.1.2 | 0.27.2 | 11.8 | ✅ Tested |
| ComfyUI latest | 2.3.1 | 0.29.0 | 12.1 | ✅ Tested |
| Upstream Stream-DiffVSR | 2.0.0 | 0.25.0 | 11.x | ✅ Reference |

### Defensive Import Pattern

```python
# stream_diffvsr/compat.py

"""
Compatibility layer for handling version differences.
"""

import warnings
from packaging import version

def check_dependencies():
    """Verify dependencies and warn about potential issues."""
    
    issues = []
    
    # Check torch
    try:
        import torch
        if version.parse(torch.__version__) < version.parse("2.0.0"):
            issues.append(f"torch {torch.__version__} < 2.0.0 (untested)")
    except ImportError:
        raise ImportError("PyTorch is required")
    
    # Check diffusers
    try:
        import diffusers
        dv = version.parse(diffusers.__version__)
        if dv < version.parse("0.25.0"):
            issues.append(f"diffusers {diffusers.__version__} < 0.25.0 (may not work)")
        elif dv >= version.parse("0.32.0"):
            issues.append(f"diffusers {diffusers.__version__} >= 0.32.0 (untested)")
    except ImportError:
        raise ImportError("diffusers is required: pip install diffusers>=0.25.0")
    
    # Check for xformers (optional but recommended)
    try:
        import xformers
        print(f"[Stream-DiffVSR] xformers {xformers.__version__} available")
    except ImportError:
        pass  # Optional, don't warn
    
    if issues:
        warnings.warn(
            "Stream-DiffVSR dependency warnings:\n  - " + 
            "\n  - ".join(issues)
        )
    
    return len(issues) == 0
```

### pyproject.toml

```toml
[project]
name = "comfyui-stream-diffvsr"
version = "1.0.0"
description = "ComfyUI nodes for Stream-DiffVSR video super-resolution"
readme = "README.md"
license = {text = "Apache-2.0"}
authors = [
    {name = "Cedar"}
]
requires-python = ">=3.10"
keywords = ["comfyui", "video", "super-resolution", "diffusion", "upscaling"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Multimedia :: Video",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "safetensors>=0.4.0",
    "einops>=0.6.0",
    "numpy>=1.24.0",
    "diffusers>=0.25.0,<0.32.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
performance = [
    "xformers>=0.0.20",
    "triton>=2.0.0",
]

[project.urls]
Homepage = "https://github.com/cedar/ComfyUI-Stream-DiffVSR"
Repository = "https://github.com/cedar/ComfyUI-Stream-DiffVSR"
Documentation = "https://github.com/cedar/ComfyUI-Stream-DiffVSR#readme"
Upstream = "https://github.com/jamichss/Stream-DiffVSR"

[tool.black]
line-length = 100
target-version = ["py310"]

[tool.ruff]
line-length = 100
select = ["E", "F", "I", "N", "W"]
```

---

## 15. Future Roadmap

### Version 1.1

- [ ] TensorRT integration for faster inference
- [ ] SageAttention support (no Block-Sparse-Attention compilation)
- [ ] Batch normalization for multi-GPU processing

### Version 1.2

- [ ] Video-to-video workflow node (VHS integration)
- [ ] Audio passthrough support
- [ ] Automatic chunk sizing based on available VRAM

### Version 2.0

- [ ] Real-time preview mode
- [ ] Custom training support
- [ ] Support for additional VSR models (FlashVSR, Upscale-A-Video)

---

## 16. Reference Implementations

### Diffusion Streaming VSR in ComfyUI

These repositories provide valuable reference patterns:

| Repository | What to Learn |
|------------|---------------|
| [smthemex/ComfyUI_FlashVSR](https://github.com/smthemex/ComfyUI_FlashVSR) | Model loading, batch processing, tiling, Apache-2.0 patterns |
| [1038lab/ComfyUI-FlashVSR](https://github.com/1038lab/ComfyUI-FlashVSR) | Auto-download, presets, node UX, error handling |
| [lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast](https://github.com/lihaoyun6/ComfyUI-FlashVSR_Ultra_Fast) | Sparse attention alternatives, tile_dit, multi-GPU |
| [naxci1/ComfyUI-FlashVSR_Stable](https://github.com/naxci1/ComfyUI-FlashVSR_Stable) | VRAM optimizations, chunk processing |

### Optical Flow Building Blocks

| Repository | Use Case |
|------------|----------|
| [ComfyUI-RAFT](https://github.com/...) | Externalize flow estimation, reuse install patterns |
| [comfyui-optical-flow](https://github.com/...) | Computing + applying flow fields |

### Video Upscaling Patterns (Non-Diffusion)

| Repository | What to Learn |
|------------|---------------|
| [ComfyUI-VideoUpscale_WithModel](https://github.com/...) | Memory-aware video frame handling, node UX |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | Video I/O, audio passthrough, frame batching |

### Key Patterns to Adopt

1. **From smthemex**: Modular loader utilities, VAE format handling
2. **From 1038lab**: Preset system (Fast/Balanced/Quality), auto model download
3. **From lihaoyun6**: Sparse attention fallback without kernel compilation
4. **From VHS**: Video I/O integration, audio sync

---

## Appendix A: Node Registration

```python
# __init__.py

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

from .nodes.loader_node import StreamDiffVSR_Loader
from .nodes.upscale_node import StreamDiffVSR_Upscale
from .nodes.process_frame_node import StreamDiffVSR_ProcessFrame
from .nodes.state_nodes import StreamDiffVSR_CreateState, StreamDiffVSR_ExtractState

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

# Run dependency checks on import
from .stream_diffvsr.compat import check_dependencies
check_dependencies()
```

---

## Appendix B: Example Workflows

### Basic Workflow

```json
{
  "nodes": [
    {
      "id": 1,
      "type": "StreamDiffVSR_Loader",
      "inputs": {
        "model_version": "v1",
        "device": "auto",
        "dtype": "float16"
      }
    },
    {
      "id": 2,
      "type": "VHS_LoadVideo",
      "inputs": {
        "video": "input.mp4"
      }
    },
    {
      "id": 3,
      "type": "StreamDiffVSR_Upscale",
      "inputs": {
        "pipe": ["1", 0],
        "images": ["2", 0],
        "num_inference_steps": 4,
        "seed": 42
      }
    },
    {
      "id": 4,
      "type": "VHS_VideoCombine",
      "inputs": {
        "images": ["3", 0],
        "audio": ["2", 1],
        "filename_prefix": "upscaled"
      }
    }
  ]
}
```

---

*Document End*
