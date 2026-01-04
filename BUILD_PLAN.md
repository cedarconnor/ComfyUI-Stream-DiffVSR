# ComfyUI-Stream-DiffVSR Build Plan

**Status:** Ready to Implement  
**Based on:** Upstream analysis of https://github.com/jamichss/Stream-DiffVSR  
**Last Updated:** January 2026

---

## Summary

This plan implements a ComfyUI custom node pack wrapping **Stream-DiffVSR** for 4x video super-resolution with temporal guidance. 

### Key Architecture (Post-Analysis)

The upstream implementation uses **standard diffusers components**, not custom modules:

| Component | Actual Implementation |
|-----------|----------------------|
| "ARTG" | **ControlNet** - warped previous HQ as conditioning |
| U-Net | **UNet2DConditionModel** from diffusers |
| Temporal VAE | **TemporalAutoencoderTiny** (custom, but straightforward) |
| Flow | **RAFT-Large** from torchvision |
| Scheduler | **DDIMScheduler** from diffusers |

### Critical Design Decisions

| Decision | Rationale |
|----------|-----------|
| Use ControlNet directly | Matches upstream exactly, simpler than custom ARTG |
| HuggingFace loading | Auto-download from `Jamichsu/Stream-DiffVSR` |
| Self-contained pipeline | ControlNet injects at U-Net residuals during denoising |
| Batch = time convention | Process frames sequentially for temporal guidance |
| BCHW state tensors | Match internal format, convert at I/O boundaries |
| Apache-2.0 license | Match upstream, include NOTICE file |

---

## Phase 1: Project Structure Updates

### 1.1 Rename/Restructure Model Files

```
stream_diffvsr/models/
├── __init__.py           # Updated exports
├── loader.py             # HuggingFace model loading
├── controlnet.py         # ControlNet wrapper (replaces artg.py)
├── unet.py               # UNet2DConditionModel wrapper
├── temporal_vae.py       # TemporalAutoencoderTiny (replaces temporal_decoder.py)
└── flow_estimator.py     # RAFT-Large from torchvision
```

### 1.2 Delete Obsolete Files

- [ ] Delete `stream_diffvsr/models/artg.py` (replaced by controlnet.py)
- [ ] Rename concepts throughout documentation

### 1.3 Update Dependencies

```
# requirements.txt
torch>=2.0.0,<2.5.0
torchvision>=0.15.0,<0.20.0
diffusers>=0.25.0,<0.32.0
transformers>=4.30.0,<5.0.0
safetensors>=0.4.0,<1.0.0
accelerate>=0.20.0,<1.0.0
```

---

## Phase 2: Utility Implementation

### 2.1 Flow Utilities

**File:** `stream_diffvsr/utils/flow_utils.py`

Implement from upstream:
```python
def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp image/features with optical flow.
    
    Args:
        x: (B, C, H, W) tensor
        flow: (B, 2, H, W) or (B, H, W, 2) tensor
    
    Returns:
        Warped tensor (B, C, H, W)
    """
    # Handle flow format
    if flow.dim() == 4 and flow.shape[1] == 2:
        flow = flow.permute(0, 2, 3, 1)  # (B, 2, H, W) -> (B, H, W, 2)
    
    # Create sampling grid
    _, _, H, W = x.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack((grid_x, grid_y), 2).float().to(x.device)
    vgrid = grid + flow
    
    # Normalize to [-1, 1]
    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    
    return F.grid_sample(x, vgrid_scaled, mode=interp_mode, 
                         padding_mode=padding_mode, align_corners=False)
```

### 2.2 Image Utilities

**File:** `stream_diffvsr/utils/image_utils.py`

- BHWC ↔ BCHW conversion
- [0,1] ↔ [-1,1] normalization
- Bicubic 4x upscaling for flow computation

---

## Phase 3: Model Wrappers

### 3.1 ControlNet (Temporal Guidance)

**File:** `stream_diffvsr/models/controlnet.py`

```python
from diffusers import ControlNetModel

class TemporalControlNet:
    """ControlNet for temporal guidance from warped previous frame."""
    
    @classmethod
    def from_pretrained(cls, model_id, subfolder="controlnet", **kwargs):
        return ControlNetModel.from_pretrained(
            model_id, subfolder=subfolder, **kwargs
        )
```

### 3.2 U-Net

**File:** `stream_diffvsr/models/unet.py`

```python
from diffusers import UNet2DConditionModel

class StreamDiffVSRUNet:
    """U-Net wrapper with ControlNet residual injection."""
    
    @classmethod
    def from_pretrained(cls, model_id, subfolder="unet", **kwargs):
        return UNet2DConditionModel.from_pretrained(
            model_id, subfolder=subfolder, **kwargs
        )
```

### 3.3 Temporal VAE

**File:** `stream_diffvsr/models/temporal_vae.py`

Need to vendor `TemporalAutoencoderTiny` from upstream or create adapter:

```python
class TemporalVAE:
    """Temporal-aware VAE with TPM for feature fusion."""
    
    def encode(self, x, return_features_only=False):
        """Encode image, optionally returning layer features for TPM."""
        pass
    
    def decode(self, latents, temporal_features=None):
        """Decode with optional temporal feature fusion via TPM."""
        pass
    
    def reset_temporal_condition(self):
        """Reset TPM state between frames."""
        pass
```

### 3.4 Flow Estimator

**File:** `stream_diffvsr/models/flow_estimator.py`

```python
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

class FlowEstimator:
    """RAFT-Large optical flow estimator."""
    
    def __init__(self, device='cuda'):
        self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
        self.model = self.model.to(device).eval()
        self.model.requires_grad_(False)
    
    def __call__(self, target, source):
        """Estimate flow from source to target."""
        flows = self.model(target, source)
        return flows[-1].permute(0, 2, 3, 1)  # (B, H, W, 2)
```

---

## Phase 4: Scheduler

**File:** `stream_diffvsr/schedulers/ddim.py`

Use diffusers DDIMScheduler directly:

```python
from diffusers import DDIMScheduler

def create_scheduler(model_id):
    return DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
```

---

## Phase 5: Pipeline Implementation

**File:** `stream_diffvsr/pipeline.py`

Major changes from stub:

```python
class StreamDiffVSRPipeline:
    def __init__(self, unet, controlnet, vae, scheduler, flow_estimator, ...):
        self.unet = unet
        self.controlnet = controlnet  # Replaces ARTG
        self.vae = vae  # TemporalAutoencoderTiny
        self.scheduler = scheduler  # DDIMScheduler
        self.flow_estimator = flow_estimator  # RAFT-Large
    
    def compute_flows(self, images):
        """Compute forward flows between UPSCALED frames."""
        # images are already bicubic 4x upscaled
        flows = []
        for i in range(1, len(images)):
            flow = self.flow_estimator(images[i], images[i-1])
            flows.append(flow)
        return flows
    
    def process_frame(self, lq_frame, state, ...):
        """Process single frame with ControlNet temporal guidance."""
        
        # 1. Bicubic upscale LQ to target resolution
        lq_upscaled = F.interpolate(lq_frame, scale_factor=4, mode='bicubic')
        
        # 2. Get temporal conditioning if not first frame
        if state.has_previous:
            # Warp previous HQ to current frame
            warped_prev = flow_warp(state.previous_hq, state.flow)
            
            # Extract TPM features
            temporal_features = self.vae.encode(
                warped_prev, return_features_only=True
            )[::-1]
        else:
            warped_prev = None
            temporal_features = None
        
        # 3. Denoising loop with ControlNet
        latents = self.prepare_latents(...)
        
        for t in self.scheduler.timesteps:
            # ControlNet (temporal guidance)
            if warped_prev is not None:
                down_samples, mid_sample = self.controlnet(
                    latents, t,
                    encoder_hidden_states=prompt_embeds,
                    controlnet_cond=warped_prev,
                    return_dict=False
                )
            else:
                down_samples, mid_sample = None, None
            
            # U-Net with ControlNet injection
            noise_pred = self.unet(
                latents, t,
                encoder_hidden_states=prompt_embeds,
                down_block_additional_residuals=down_samples,
                mid_block_additional_residual=mid_sample,
                return_dict=False
            )[0]
            
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        
        # 4. Decode with temporal features
        hq_frame = self.vae.decode(latents, temporal_features=temporal_features)
        
        # 5. Update state
        new_state = StreamDiffVSRState(
            previous_hq=hq_frame,
            frame_index=state.frame_index + 1
        )
        
        return hq_frame, new_state
```

---

## Phase 6: Model Loading

**File:** `stream_diffvsr/models/loader.py`

```python
MODEL_ID = "Jamichsu/Stream-DiffVSR"

def load_pipeline(device='cuda', dtype=torch.float16):
    """Load complete pipeline from HuggingFace."""
    
    controlnet = ControlNetModel.from_pretrained(
        MODEL_ID, subfolder="controlnet", torch_dtype=dtype
    )
    unet = UNet2DConditionModel.from_pretrained(
        MODEL_ID, subfolder="unet", torch_dtype=dtype
    )
    vae = TemporalAutoencoderTiny.from_pretrained(
        MODEL_ID, subfolder="vae", torch_dtype=dtype
    )
    scheduler = DDIMScheduler.from_pretrained(
        MODEL_ID, subfolder="scheduler"
    )
    flow_estimator = FlowEstimator(device=device)
    
    return StreamDiffVSRPipeline(
        unet=unet.to(device),
        controlnet=controlnet.to(device),
        vae=vae.to(device),
        scheduler=scheduler,
        flow_estimator=flow_estimator,
        device=device,
        dtype=dtype
    )
```

---

## Phase 7: Node Updates

### 7.1 Loader Node

Update to use HuggingFace loading:

```python
class StreamDiffVSR_Loader:
    def load_model(self, model_version, device, dtype):
        from stream_diffvsr.models.loader import load_pipeline
        return (load_pipeline(device=device, dtype=dtype),)
```

### 7.2 Upscale Node

Minor updates to match new pipeline API.

---

## Implementation Order

### Day 1: Utilities & Infrastructure
1. Update `requirements.txt` with correct deps
2. Implement `flow_utils.py` with flow_warp
3. Update `image_utils.py` with bicubic upscaling
4. Delete `artg.py`, create `controlnet.py` (thin wrapper)

### Day 2: Model Wrappers
1. Update `unet.py` to wrap UNet2DConditionModel
2. Create `temporal_vae.py` (vendor from upstream)
3. Update `flow_estimator.py` for RAFT-Large
4. Update `ddim.py` to use diffusers DDIMScheduler

### Day 3: Pipeline & Integration
1. Rewrite `pipeline.py` with ControlNet flow
2. Update `loader.py` for HuggingFace loading
3. Update loader_node.py
4. Test basic loading

### Day 4: Testing & Validation
1. Single frame test
2. Multi-frame temporal consistency test
3. VRAM usage validation
4. Compare output to upstream inference.py

---

## Verification Plan

### Automated Tests

```bash
# Test imports
python -c "from stream_diffvsr.pipeline import StreamDiffVSRPipeline"

# Test model loading
python -c "
from stream_diffvsr.models.loader import load_pipeline
pipe = load_pipeline()
print('Pipeline loaded successfully')
"

# Test single frame
python -c "
import torch
from stream_diffvsr.models.loader import load_pipeline
pipe = load_pipeline()
test_input = torch.randn(1, 180, 320, 3)  # 180p test
output, state = pipe(test_input)
assert output.shape == (1, 720, 1280, 3), f'Wrong shape: {output.shape}'
print('Single frame test passed')
"
```

### Manual Verification

1. Load workflow in ComfyUI
2. Process test video (3-5 frames)
3. Compare output quality to upstream
4. Verify temporal consistency (no flickering)

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| TemporalAutoencoderTiny not in diffusers | Vendor from upstream with minimal changes |
| HuggingFace model format differs | Test loading early, adapt as needed |
| VRAM usage higher than expected | Implement tiling as Phase 2 optimization |
| diffusers version incompatibility | Pin versions in requirements.txt |

---

*Plan created: January 2026*
