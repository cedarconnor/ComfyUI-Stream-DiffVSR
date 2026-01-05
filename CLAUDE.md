# CLAUDE.md - Claude Code Context for ComfyUI-Stream-DiffVSR

## Project Overview

ComfyUI custom node pack wrapping **Stream-DiffVSR** for low-latency video super-resolution with auto-regressive temporal guidance.

**Upstream repo:** https://github.com/jamichss/Stream-DiffVSR  
**HuggingFace:** https://huggingface.co/Jamichsu/Stream-DiffVSR  
**Design doc:** See `ComfyUI-Stream-DiffVSR-Design-Doc.md`

## Quick Reference

```
Stream-DiffVSR = 4-step diffusion VSR with temporal feedback
- Each frame uses previous HQ output (warped by optical flow) for guidance
- ControlNet injects temporal features INTO U-Net during denoising
- Temporal VAE decoder fuses features via TPM (Temporal Processor Module)
- Cannot use ComfyUI's KSampler - we own the entire denoising loop
```

## Actual Architecture (Post-Analysis)

### Key Components

| Component | Implementation | Source |
|-----------|---------------|--------|
| **Temporal Guidance** | `ControlNetModel` (diffusers) | Takes warped previous HQ as conditioning |
| **U-Net** | `UNet2DConditionModel` (diffusers) | SD x4 Upscaler base, distilled for 4-step |
| **Temporal VAE** | `TemporalAutoencoderTiny` | Custom VAE with TPM for temporal fusion |
| **Flow Estimator** | `raft_large` (torchvision) | RAFT-Large with automatic tiled fallback on OOM |
| **Scheduler** | `DDIMScheduler` (diffusers) | Standard DDIM, 4 steps default |

### Inference Flow (from upstream)

```python
# 1. Upscale LQ frames 4x with bicubic (BEFORE flow estimation)
upscaled_images = [F.interpolate(img, scale_factor=4, mode='bicubic') for img in lq_images]

# 2. Compute optical flow between upscaled frames
flows = compute_flows(of_model, upscaled_images)

# 3. For each frame:
for num_image, image in enumerate(images):
    # Warp previous HQ to current frame
    if num_image > 0:
        warped_prev = flow_warp(prev_hq, flows[num_image - 1])
        # Extract features for temporal decoder
        temporal_features = vae.encode(warped_prev, return_features_only=True)[::-1]
    
    # Denoising loop with ControlNet
    for t in timesteps:
        # ControlNet produces residuals from warped previous frame
        if num_image > 0:
            down_samples, mid_sample = controlnet(latents, t, warped_prev)
        else:
            down_samples, mid_sample = None, None
        
        # U-Net with ControlNet injection
        noise_pred = unet(latents, t, 
                         down_block_additional_residuals=down_samples,
                         mid_block_additional_residual=mid_sample)
        latents = scheduler.step(noise_pred, t, latents)
    
    # Decode with temporal features
    hq = vae.decode(latents, temporal_features=temporal_features)
    prev_hq = hq  # Store for next frame
```

## Critical Architecture Constraints

### 1. ControlNet Injection (Why We Bypass ComfyUI's Sampler)

Stream-DiffVSR uses **ControlNet** to inject temporal guidance. The warped previous HQ frame is the ControlNet conditioning image.

```python
# WRONG - Cannot use ComfyUI's sampler infrastructure
model = comfy.sd.load_checkpoint(...)
samples = nodes.KSampler(model, ...)

# CORRECT - Self-contained pipeline owns the loop
for t in timesteps:
    # ControlNet processes warped previous frame
    down_samples, mid_sample = controlnet(
        latents, t, encoder_hidden_states, controlnet_cond=warped_prev_hq
    )
    # U-Net receives ControlNet outputs as additional residuals
    noise_pred = unet(latents, t, encoder_hidden_states,
                      down_block_additional_residuals=down_samples,
                      mid_block_additional_residual=mid_sample)
    latents = scheduler.step(noise_pred, t, latents)
```

### 2. Temporal VAE with TPM

The decoder uses Temporal Processor Modules (TPM) to fuse current decode features with features extracted from the warped previous frame:

```python
# Encoder extracts multi-scale features from warped previous HQ
layer_features = vae.encode(warped_prev_hq, return_features_only=True)
temporal_features = layer_features[::-1]  # Reverse for decoder order

# Decoder uses TPM to fuse at each scale
hq_image = vae.decode(latents, temporal_features=temporal_features)
```

### 3. Flow on UPSCALED Images

**Critical:** Flow is computed on bicubic-upscaled 4x images, NOT the original LQ resolution:

```python
# WRONG - flow on LQ resolution
flow = of_model(current_lq, prev_lq)

# CORRECT - flow on upscaled images
current_up = F.interpolate(current_lq, scale_factor=4, mode='bicubic')
prev_up = F.interpolate(prev_lq, scale_factor=4, mode='bicubic')
flow = of_model(current_up, prev_up)
```

### 4. Batch Dimension = Time (Frame Order)

ComfyUI IMAGE tensors use batch dim for frames. Process sequentially:

```python
# images.shape = (N, H, W, C) where N = frame count
for i in range(N):
    frame = images[i:i+1]  # Keep batch dim
    hq, state = process_frame(frame, state)  # State carries temporal info
```

### 5. State Contains Large Tensors

```python
@dataclass
class StreamDiffVSRState:
    previous_hq: torch.Tensor  # (1, C, H*4, W*4) BCHW float32 [-1,1]
    frame_index: int
```

**NEVER serialize to Python lists.** Use safetensors for disk, direct object passing for node-to-node.

### 6. Tensor Layout Convention

| Location | Format | Range |
|----------|--------|-------|
| ComfyUI node I/O | BHWC | [0, 1] |
| Internal pipeline | BCHW | [-1, 1] |
| State storage | BCHW | [-1, 1] |
| VAE latents | BCHW | magnitude/shift scaled |

Convert at node boundaries only:
```python
# Node input → pipeline
x = image.permute(0, 3, 1, 2)  # BHWC → BCHW
x = x * 2.0 - 1.0  # [0,1] → [-1,1]

# Pipeline output → node
out = (result + 1.0) / 2.0  # [-1,1] → [0,1]
out = out.permute(0, 2, 3, 1)  # BCHW → BHWC
```

## Common Pitfalls (Do Not Make These Mistakes)

### ❌ Using ARTG terminology
```python
# WRONG - outdated terminology
artg_features = artg(warped_hq, z_lq)

# CORRECT - it's ControlNet
down_samples, mid_sample = controlnet(latents, t, controlnet_cond=warped_hq)
```

### ❌ Hardcoded VAE Scaling
```python
# WRONG - SD scaling factor doesn't apply here
latent = vae.encode(x) * 0.18215

# CORRECT - TemporalAutoencoderTiny uses scaling_factor=1.0
# with latent_magnitude and latent_shift for scaling
latent = vae.encode(x)  # No additional scaling
```

### ❌ Flow on LQ Resolution
```python
# WRONG - flow on original LQ
flow = estimate_flow(lq_current, lq_prev)

# CORRECT - bicubic upscale first, then compute flow
upscaled = F.interpolate(lq, scale_factor=4, mode='bicubic')
flow = estimate_flow(upscaled_current, upscaled_prev)
```

### ❌ Naive Tiling
Tiling breaks temporal consistency because flow/warping must be computed on full frames.

```python
# WRONG - tile everything including flow
for tile in tiles:
    flow_tile = estimate_flow(lq_tile, prev_lq_tile)  # Discontinuous!

# CORRECT - flow auto-tiles with overlap blending on OOM
# The FlowEstimator automatically handles this:
try:
    flow = estimate_flow(lq_full, prev_lq_full)  # Try full frame
except OOM:
    flow = estimate_flow_tiled(lq_full, prev_lq_full)  # Auto-fallback
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

## File Structure

```
ComfyUI-Stream-DiffVSR/
├── __init__.py                 # Node registration + version check
├── CLAUDE.md                   # This file
├── LICENSE                     # Apache-2.0
├── README.md
├── requirements.txt
├── pyproject.toml

├── nodes/
│   ├── __init__.py
│   ├── loader_node.py          # StreamDiffVSR_Loader
│   ├── upscale_node.py         # StreamDiffVSR_Upscale (batch processing)
│   ├── video_upscale_node.py   # StreamDiffVSR_UpscaleVideo (all-in-one)
│   └── state_nodes.py          # CreateState, ExtractState

├── stream_diffvsr/
│   ├── __init__.py
│   ├── pipeline.py             # Main inference pipeline
│   ├── state.py                # StreamDiffVSRState dataclass
│   ├── compat.py               # Version checks, defensive imports
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py           # Model loading from HuggingFace
│   │   ├── unet.py             # UNet2DConditionModel wrapper
│   │   ├── controlnet.py       # ControlNet for temporal guidance
│   │   ├── temporal_vae.py     # TemporalAutoencoderTiny wrapper
│   │   └── flow_estimator.py   # RAFT-Large with auto-tiling
│   │
│   ├── schedulers/
│   │   └── ddim.py             # DDIMScheduler (from diffusers)
│   │
│   └── utils/
│       ├── image_utils.py      # Tensor conversions
│       ├── flow_utils.py       # Warp operations
│       ├── tiling.py           # Temporal-aware tiling utilities
│       └── device_utils.py     # Device/dtype management

└── example_workflows/
    ├── video_upscale.json      # Simple all-in-one workflow
    └── single_image_upscale.json
```

## Node Signatures

### StreamDiffVSR_Loader
```python
RETURN_TYPES = ("STREAM_DIFFVSR_PIPE",)
# Loads all model components from HuggingFace, returns pipeline object
```

### StreamDiffVSR_UpscaleVideo (Recommended)
```python
INPUT_TYPES = {
    "required": {
        "pipe": ("STREAM_DIFFVSR_PIPE",),
        "video_path": ("STRING",),
    },
    "optional": {
        "output_path": ("STRING", {"default": ""}),
        "frames_per_batch": ("INT", {"default": 16}),
        "start_frame": ("INT", {"default": 0}),
        "end_frame": ("INT", {"default": -1}),  # -1 = all
        "num_inference_steps": ("INT", {"default": 4}),
        "seed": ("INT", {"default": 0}),
    }
}
RETURN_TYPES = ("STRING",)  # Output video path
OUTPUT_NODE = True
# All-in-one video processing with automatic chunking
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
# For VHS integration or manual batch workflows
```

## Model Loading

### Option 1: Manual Download (Preferred)

Download models from HuggingFace and place in ComfyUI models directory:

```
ComfyUI/models/StreamDiffVSR/v1/
├── controlnet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.safetensors
└── scheduler/
    └── scheduler_config.json
```

**Download from:** https://huggingface.co/Jamichsu/Stream-DiffVSR

### Option 2: Auto-Download (Fallback)

If local models not found, the loader automatically downloads from HuggingFace.
Models are cached in `~/.cache/huggingface/`.

```python
from diffusers import ControlNetModel, UNet2DConditionModel, DDIMScheduler
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

# Load from HuggingFace (if local not found)
model_id = "Jamichsu/Stream-DiffVSR"

controlnet = ControlNetModel.from_pretrained(model_id, subfolder="controlnet")
unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
vae = TemporalAutoencoderTiny.from_pretrained(model_id, subfolder="vae")
scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")

# RAFT from torchvision (always auto-downloaded)
of_model = raft_large(weights=Raft_Large_Weights.DEFAULT)
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
- [ ] Multi-frame batch (temporal guidance via ControlNet)
- [ ] State continuity across chunks
- [ ] VRAM usage at 720p, 1080p
- [ ] Tiling with temporal consistency preserved
- [ ] Different dtypes (fp16, bf16, fp32)
- [ ] Error handling for missing models
- [ ] Compatibility with VHS video loader

## Dependencies (Known Good)

```
torch>=2.0.0
torchvision>=0.15.0
diffusers>=0.30.0,<0.32.0
transformers>=4.30.0,<5.0.0
safetensors>=0.4.0
accelerate>=0.20.0
```

## Reference Implementations

Study these for patterns:
- `smthemex/ComfyUI_FlashVSR` - Model loading, Apache-2.0 patterns
- `1038lab/ComfyUI-FlashVSR` - Auto-download, presets, error handling
- `Kosinkadink/ComfyUI-VideoHelperSuite` - Video I/O, audio sync

## Architecture Questions (RESOLVED)

| Question | Answer |
|----------|--------|
| VAE encode method? | Direct encode, no latent_dist (AutoEncoderTiny) |
| ARTG injection point? | **ControlNet** via down_block/mid_block residuals |
| Flow model? | RAFT-Large from torchvision |
| VAE scaling factor? | 1.0 (uses latent_magnitude/shift instead) |
| First frame handling? | No ControlNet conditioning, no temporal features |
