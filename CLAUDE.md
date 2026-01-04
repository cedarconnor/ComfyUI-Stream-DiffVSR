# CLAUDE.md - Claude Code Context for ComfyUI-Stream-DiffVSR

## Project Overview

ComfyUI custom node pack wrapping **Stream-DiffVSR** for low-latency video super-resolution with auto-regressive temporal guidance.

**Upstream repo:** https://github.com/jamichss/Stream-DiffVSR  
**License:** Apache-2.0 (NOT MIT - this is critical)  
**Design doc:** See `ComfyUI-Stream-DiffVSR-Design-Doc.md`

## Quick Reference

```
Stream-DiffVSR = 4-step diffusion VSR with temporal feedback
- Each frame uses previous HQ output (warped by optical flow) for guidance
- ARTG module injects temporal features INTO U-Net decoder mid-network
- Cannot use ComfyUI's KSampler - we own the entire denoising loop
```

## Critical Architecture Constraints

### 1. ARTG Injection (Why We Bypass ComfyUI's Sampler)

Stream-DiffVSR requires injecting temporal features at specific U-Net decoder layers during denoising. This is NOT a conditioning input - it's mid-network feature injection.

```python
# WRONG - Cannot use ComfyUI's sampler infrastructure
model = comfy.sd.load_checkpoint(...)
samples = nodes.KSampler(model, ...)

# CORRECT - Self-contained pipeline owns the loop
for t in timesteps:
    temporal_features = artg(warped_hq, z_lq)  # Compute guidance
    noise_pred = unet(z_t, t, cond, temporal_features)  # Inject here
    z_t = scheduler.step(noise_pred, t, z_t)
```

### 2. Batch Dimension = Time (Frame Order)

ComfyUI IMAGE tensors use batch dim for frames. Process sequentially:

```python
# images.shape = (N, H, W, C) where N = frame count
for i in range(N):
    frame = images[i:i+1]  # Keep batch dim
    hq, state = process_frame(frame, state)  # State carries temporal info
```

### 3. State Contains Large Tensors

```python
@dataclass
class StreamDiffVSRState:
    previous_hq: torch.Tensor  # (1, C, H*4, W*4) BCHW float32 [0,1]
    previous_lq: torch.Tensor  # (1, C, H, W) BCHW float32 [0,1]
    frame_index: int
```

**NEVER serialize to Python lists.** Use safetensors for disk, direct object passing for node-to-node.

### 4. Tensor Layout Convention

| Location | Format | Range |
|----------|--------|-------|
| ComfyUI node I/O | BHWC | [0, 1] |
| Internal pipeline | BCHW | [0, 1] or [-1, 1] |
| State storage | BCHW | [0, 1] |
| VAE latents | BCHW | scaled by config.scaling_factor |

Convert at node boundaries only:
```python
# Node input → pipeline
x = image.permute(0, 3, 1, 2)  # BHWC → BCHW

# Pipeline output → node
out = result.permute(0, 2, 3, 1)  # BCHW → BHWC
```

## Common Pitfalls (Do Not Make These Mistakes)

### ❌ Hardcoded VAE Scaling
```python
# WRONG
latent = vae.encode(x) * 0.18215

# CORRECT
latent = vae.encode(x).latent_dist.mode()
latent = latent * vae.config.scaling_factor
```

### ❌ Naive Tiling
Tiling breaks temporal consistency because flow/warping must be computed on full frames.

```python
# WRONG - tile everything
for tile in tiles:
    flow_tile = estimate_flow(lq_tile, prev_lq_tile)  # Discontinuous!

# CORRECT - flow on full frame, tile only diffusion
flow = estimate_flow(lq_full, prev_lq_full)
warped_hq = warp(prev_hq_full, flow)
for tile in tiles:
    process_diffusion_tile(...)
```

### ❌ State Serialization via Lists
```python
# WRONG - explodes memory, loses dtype
def to_dict(self):
    return {"previous_hq": self.previous_hq.numpy().tolist()}

# CORRECT - use safetensors for disk persistence
from safetensors.torch import save_file
save_file({"previous_hq": self.previous_hq}, path)
```

### ❌ Claiming MIT License
Upstream is Apache-2.0. Include NOTICE file, attribute authors.

## File Structure

```
ComfyUI-Stream-DiffVSR/
├── __init__.py                 # Node registration + version check
├── CLAUDE.md                   # This file
├── LICENSE                     # Apache-2.0
├── NOTICE                      # Attribution to upstream authors
├── README.md
├── requirements.txt
├── pyproject.toml
│
├── nodes/
│   ├── __init__.py
│   ├── loader_node.py          # StreamDiffVSR_Loader
│   ├── upscale_node.py         # StreamDiffVSR_Upscale (main)
│   ├── process_frame_node.py   # StreamDiffVSR_ProcessFrame (advanced)
│   └── state_nodes.py          # CreateState, ExtractState
│
├── stream_diffvsr/
│   ├── __init__.py
│   ├── pipeline.py             # Main inference pipeline
│   ├── state.py                # StreamDiffVSRState dataclass
│   ├── compat.py               # Version checks, defensive imports
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── unet.py             # Distilled U-Net wrapper
│   │   ├── artg.py             # Auto-Regressive Temporal Guidance
│   │   ├── temporal_decoder.py # VAE decoder with TPM
│   │   └── flow_estimator.py   # RAFT wrapper
│   │
│   ├── schedulers/
│   │   └── ddim_4step.py       # 4-step DDIM
│   │
│   └── utils/
│       ├── image_utils.py      # Tensor conversions
│       ├── flow_utils.py       # Warp operations
│       ├── tiling.py           # Temporal-aware tiling
│       └── device_utils.py     # Device/dtype management
│
└── example_workflows/
    ├── basic_upscale.json
    └── vhs_integration.json
```

## Node Signatures

### StreamDiffVSR_Loader
```python
RETURN_TYPES = ("STREAM_DIFFVSR_PIPE",)
# Loads all model components, returns pipeline object
```

### StreamDiffVSR_Upscale
```python
INPUT_TYPES = {
    "required": {
        "pipe": ("STREAM_DIFFVSR_PIPE",),
        "images": ("IMAGE",),  # Batch = frames in order
    },
    "optional": {
        "state": ("STREAM_DIFFVSR_STATE",),
        "num_inference_steps": ("INT", {"default": 4}),
        "seed": ("INT", {"default": 0}),
    }
}
RETURN_TYPES = ("IMAGE", "STREAM_DIFFVSR_STATE")
```

### StreamDiffVSR_ProcessFrame (Advanced)
```python
# Single frame processing with explicit state I/O
# For custom loops, VHS integration
```

## Model Files Location

```
ComfyUI/models/StreamDiffVSR/
├── unet/
│   └── diffusion_pytorch_model.safetensors
├── artg/
│   └── artg_module.safetensors
├── temporal_decoder/
│   └── temporal_decoder.safetensors
├── vae/
│   └── vae_tiny.safetensors
└── flow/
    └── raft_small.pth  (optional - can reuse existing)
```

## Development Commands

```bash
# Install in dev mode
cd ComfyUI/custom_nodes
git clone <repo> ComfyUI-Stream-DiffVSR
cd ComfyUI-Stream-DiffVSR
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Format code
black . --line-length 100
ruff check . --fix

# Check types
mypy stream_diffvsr/ --ignore-missing-imports
```

## Testing Checklist

- [ ] Single frame processing (no temporal guidance)
- [ ] Multi-frame batch (temporal guidance active)
- [ ] State continuity across chunks
- [ ] VRAM usage at 720p, 1080p
- [ ] Tiling with temporal consistency preserved
- [ ] Different dtypes (fp16, bf16, fp32)
- [ ] Error handling for missing models
- [ ] Compatibility with VHS video loader

## Dependencies (Known Good)

```
torch>=2.0.0,<2.5.0
diffusers>=0.25.0,<0.32.0
safetensors>=0.4.0
einops>=0.6.0
```

## Reference Implementations

Study these for patterns:
- `smthemex/ComfyUI_FlashVSR` - Model loading, Apache-2.0 patterns
- `1038lab/ComfyUI-FlashVSR` - Auto-download, presets, error handling
- `Kosinkadink/ComfyUI-VideoHelperSuite` - Video I/O, audio sync

## When Implementing

1. **Start with upstream inference.py** - Vendor the minimal forward pass
2. **Get correctness first** - No tiling, no optimizations
3. **Add state management** - Test temporal consistency
4. **Then optimize** - Tiling, xformers, torch.compile
5. **Finally, polish UX** - Presets, auto-download, error messages

## Key Upstream Files to Study

```
Stream-DiffVSR/
├── inference.py           # Entry point - copy this logic
├── pipeline/              # Pipeline class
├── scheduler/             # DDIM implementation
├── temporal_autoencoder/  # Decoder with TPM
└── util/                  # Flow, warping utilities
```

## Questions to Resolve During Implementation

1. Does upstream use `latent_dist.sample()` or `.mode()` for VAE encode?
2. What's the exact ARTG injection point in the U-Net?
3. Does flow estimation use RAFT or another model?
4. What's the VAE scaling factor in the shipped config?
5. How does the temporal decoder handle the first frame (no previous)?
