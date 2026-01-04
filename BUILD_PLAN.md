# ComfyUI-Stream-DiffVSR Build Plan

**Status:** Ready to Implement
**Based on:** Design Doc v1.1 + CLAUDE.md

---

## Summary

This plan implements a ComfyUI custom node pack wrapping **Stream-DiffVSR** for 4x video super-resolution with temporal guidance. The core insight is that we **cannot** use ComfyUI's standard sampler infrastructure - the ARTG module requires mid-network feature injection during denoising.

### Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Self-contained pipeline | ARTG injects at U-Net decoder layers mid-denoising |
| Wrap upstream inference | Fast MVP, guaranteed correctness vs clean diffusers |
| Batch = time convention | Process frames sequentially for temporal guidance |
| BCHW state tensors | Avoid repeated permutations, convert at I/O boundaries |
| Apache-2.0 license | Match upstream, include NOTICE file |

---

## Phase 1: Project Skeleton & Infrastructure

### 1.1 Create Directory Structure

```
ComfyUI-Stream-DiffVSR/
├── __init__.py                 # Node registration + version check
├── LICENSE                     # Apache-2.0 (copy full text)
├── NOTICE                      # Attribution to upstream authors
├── README.md                   # User documentation
├── requirements.txt            # Dependencies
├── pyproject.toml              # Package metadata
│
├── nodes/
│   ├── __init__.py
│   ├── loader_node.py          # StreamDiffVSR_Loader
│   ├── upscale_node.py         # StreamDiffVSR_Upscale
│   ├── process_frame_node.py   # StreamDiffVSR_ProcessFrame
│   └── state_nodes.py          # CreateState, ExtractState
│
├── stream_diffvsr/
│   ├── __init__.py
│   ├── pipeline.py             # Main inference pipeline
│   ├── state.py                # StreamDiffVSRState dataclass
│   ├── compat.py               # Version checks, defensive imports
│   ├── exceptions.py           # Custom exceptions
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── loader.py           # Model file discovery/loading
│   │   ├── unet.py             # Distilled U-Net wrapper
│   │   ├── artg.py             # Auto-Regressive Temporal Guidance
│   │   ├── temporal_decoder.py # VAE decoder with TPM
│   │   └── flow_estimator.py   # RAFT wrapper
│   │
│   ├── schedulers/
│   │   ├── __init__.py
│   │   └── ddim_4step.py       # 4-step DDIM
│   │
│   └── utils/
│       ├── __init__.py
│       ├── image_utils.py      # Tensor conversions
│       ├── flow_utils.py       # Warp operations
│       ├── tiling.py           # Temporal-aware tiling
│       └── device_utils.py     # Device/dtype management
│
├── example_workflows/
│   ├── basic_upscale.json
│   └── vhs_integration.json
│
└── tests/
    ├── __init__.py
    ├── test_state.py
    ├── test_pipeline.py
    └── fixtures/
```

### 1.2 Create Base Files

**Files to create:**
- [ ] `LICENSE` - Apache-2.0 full text
- [ ] `NOTICE` - Attribution (Stream-DiffVSR authors, year 2025)
- [ ] `requirements.txt` - Pin to known-good versions
- [ ] `pyproject.toml` - Package metadata
- [ ] `stream_diffvsr/compat.py` - Version checks
- [ ] `stream_diffvsr/exceptions.py` - Custom exceptions

### 1.3 Dependencies

```
# requirements.txt
torch>=2.0.0,<2.5.0
torchvision>=0.15.0,<0.20.0
safetensors>=0.4.0,<1.0.0
einops>=0.6.0,<1.0.0
numpy>=1.24.0,<2.0.0
diffusers>=0.25.0,<0.32.0
transformers>=4.30.0,<5.0.0
accelerate>=0.20.0,<1.0.0
```

---

## Phase 2: Core Pipeline Implementation

### 2.1 Study Upstream Code (CRITICAL FIRST STEP)

**Before writing any code**, fetch and study the upstream repo:

```bash
git clone https://github.com/jamichss/Stream-DiffVSR /tmp/Stream-DiffVSR
```

**Questions to answer from upstream:**
1. Does VAE use `latent_dist.sample()` or `.mode()` for encode?
2. Exact ARTG injection point in U-Net decoder layers?
3. Flow estimation model (RAFT variant, config)?
4. VAE scaling factor in shipped config?
5. How does temporal decoder handle first frame (no previous)?
6. Scheduler implementation details (DDIM 4-step)?

**Key files to study:**
- `inference.py` - Entry point, copy this logic
- `pipeline/` - Pipeline class structure
- `scheduler/` - DDIM implementation
- `temporal_autoencoder/` - Decoder with TPM
- `util/` - Flow, warping utilities

### 2.2 Implement State Management

**File:** `stream_diffvsr/state.py`

```python
@dataclass
class StreamDiffVSRState:
    previous_hq: Optional[torch.Tensor] = None  # BCHW, [0,1]
    previous_lq: Optional[torch.Tensor] = None  # BCHW, [0,1]
    frame_index: int = 0
    metadata: dict = field(default_factory=dict)
```

**Key features:**
- BCHW tensor format (model-native)
- `has_previous` property for conditional logic
- `save_state()` / `load_state()` using safetensors (NOT Python lists)
- `clone()` and `to_device()` methods

### 2.3 Implement Utility Functions

**Files:**
- `stream_diffvsr/utils/image_utils.py`
  - BHWC ↔ BCHW conversion
  - [0,1] ↔ [-1,1] normalization
  - Tensor dtype handling

- `stream_diffvsr/utils/flow_utils.py`
  - `warp_image(image, flow)` - Backward warping
  - Flow upscaling for HQ resolution

- `stream_diffvsr/utils/device_utils.py`
  - Device detection (cuda/cpu/auto)
  - Dtype handling (fp16/bf16/fp32)

### 2.4 Implement Model Wrappers

**Files (ordered by dependency):**

1. `stream_diffvsr/models/loader.py`
   - `get_model_path()` - Find StreamDiffVSR folder
   - `validate_model_files()` - Check required components exist
   - `load_component()` - Load safetensors/pth files

2. `stream_diffvsr/models/flow_estimator.py`
   - `FlowEstimator` class wrapping RAFT
   - Check for existing RAFT from other nodes first
   - Fallback to bundled/torchvision RAFT

3. `stream_diffvsr/models/artg.py`
   - `ARTGModule` wrapping upstream ARTG
   - `encode_temporal(warped_hq, z_lq)` method
   - Returns features for U-Net injection

4. `stream_diffvsr/models/unet.py`
   - `StreamDiffVSRUNet` wrapper
   - Forward pass accepts `temporal_features` parameter
   - Inject ARTG features at correct decoder layers

5. `stream_diffvsr/models/temporal_decoder.py`
   - `TemporalAwareDecoder` with TPM
   - Forward pass accepts `warped_previous` and `lq_features`

### 2.5 Implement Scheduler

**File:** `stream_diffvsr/schedulers/ddim_4step.py`

- Copy/adapt upstream DDIM implementation
- `set_timesteps(num_steps)` method
- `step(noise_pred, t, latents)` method
- Default to 4 steps as optimized by distillation

### 2.6 Implement Main Pipeline

**File:** `stream_diffvsr/pipeline.py`

```python
class StreamDiffVSRPipeline:
    def __init__(self, unet, artg, decoder, vae_encoder,
                 flow_estimator, scheduler, config, device, dtype):
        ...

    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """BHWC [0,1] -> latent BCHW"""
        # Use config.scaling_factor, NOT hardcoded 0.18215

    def estimate_flow(self, current_lq, previous_lq) -> torch.Tensor:
        """Estimate optical flow between frames"""

    def process_frame(self, lq_frame, state, num_steps=4, seed=0):
        """
        Single frame processing with ARTG injection.

        1. Convert BHWC -> BCHW
        2. Encode LQ to latent
        3. If has_previous: compute flow, warp HQ, get ARTG features
        4. Denoising loop with ARTG injection at each step
        5. Temporal decoder with warped_previous
        6. Convert BCHW -> BHWC, update state
        """

    def __call__(self, images, state=None, num_steps=4, seed=0):
        """Batch processing - loop over frames internally"""
```

**Critical implementation notes:**
- VAE scaling: Use `vae.config.scaling_factor`, not 0.18215
- ARTG features inject INTO U-Net decoder, not as conditioning
- First frame: No temporal guidance (warped_hq=None, temporal_features=None)

---

## Phase 3: ComfyUI Nodes

### 3.1 Node Registration

**File:** `__init__.py`

```python
NODE_CLASS_MAPPINGS = {
    "StreamDiffVSR_Loader": StreamDiffVSR_Loader,
    "StreamDiffVSR_Upscale": StreamDiffVSR_Upscale,
    "StreamDiffVSR_ProcessFrame": StreamDiffVSR_ProcessFrame,
    "StreamDiffVSR_CreateState": StreamDiffVSR_CreateState,
    "StreamDiffVSR_ExtractState": StreamDiffVSR_ExtractState,
}
```

### 3.2 Loader Node

**File:** `nodes/loader_node.py`

```python
class StreamDiffVSR_Loader:
    CATEGORY = "StreamDiffVSR"
    RETURN_TYPES = ("STREAM_DIFFVSR_PIPE",)
    FUNCTION = "load_model"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["v1"],),
                "device": (["cuda", "cpu", "auto"],),
                "dtype": (["float16", "bfloat16", "float32"],),
            }
        }
```

**Implementation:**
1. Get model path via folder_paths
2. Validate all components exist
3. Load each component (unet, artg, decoder, vae, flow)
4. Create scheduler
5. Construct and return StreamDiffVSRPipeline

### 3.3 Main Upscale Node

**File:** `nodes/upscale_node.py`

```python
class StreamDiffVSR_Upscale:
    CATEGORY = "StreamDiffVSR"
    RETURN_TYPES = ("IMAGE", "STREAM_DIFFVSR_STATE")
    FUNCTION = "upscale"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": ("STREAM_DIFFVSR_PIPE",),
                "images": ("IMAGE",),  # Batch = frames in order
            },
            "optional": {
                "state": ("STREAM_DIFFVSR_STATE",),
                "num_inference_steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0}),
                "enable_tiling": ("BOOLEAN", {"default": False}),
                "tile_size": ("INT", {"default": 512}),
                "tile_overlap": ("INT", {"default": 64}),
            }
        }
```

**Implementation:**
1. Initialize state if None
2. Loop over batch dimension (frames)
3. Call `pipeline.process_frame()` for each
4. Stack results back to batch
5. Return (images, final_state)

### 3.4 Advanced Nodes

**File:** `nodes/process_frame_node.py`
- Single frame with explicit state I/O
- For custom loops, VHS integration

**File:** `nodes/state_nodes.py`
- `StreamDiffVSR_CreateState` - Initialize empty state
- `StreamDiffVSR_ExtractState` - Get previous HQ from state

---

## Phase 4: Testing & Validation

### 4.1 Unit Tests

- [ ] `test_state.py` - State creation, serialization, clone
- [ ] `test_pipeline.py` - Single frame, multi-frame, temporal consistency
- [ ] `test_nodes.py` - Node input validation, output shapes

### 4.2 Integration Testing Checklist

- [ ] Single frame processing (no temporal guidance)
- [ ] Multi-frame batch (temporal guidance active)
- [ ] State continuity across chunks
- [ ] VRAM usage at 720p, 1080p
- [ ] Tiling with temporal consistency preserved
- [ ] Different dtypes (fp16, bf16, fp32)
- [ ] Error handling for missing models
- [ ] Compatibility with VHS video loader

### 4.3 Quality Validation

- Compare output quality to upstream inference.py
- Verify temporal consistency (smooth motion)
- Check for tiling artifacts at boundaries

---

## Implementation Order

### Step 1: Infrastructure (Day 1)
1. Create directory structure
2. Add LICENSE, NOTICE, requirements.txt, pyproject.toml
3. Implement `compat.py` and `exceptions.py`
4. Create empty `__init__.py` files

### Step 2: Study Upstream (Day 1-2)
1. Clone and study upstream repo
2. Document answers to architecture questions
3. Identify exact code to vendor/wrap

### Step 3: Utilities & State (Day 2)
1. Implement `state.py`
2. Implement `utils/image_utils.py`
3. Implement `utils/flow_utils.py`
4. Implement `utils/device_utils.py`

### Step 4: Models (Day 3-4)
1. Implement `models/loader.py`
2. Implement `models/flow_estimator.py`
3. Implement `models/artg.py`
4. Implement `models/unet.py`
5. Implement `models/temporal_decoder.py`

### Step 5: Pipeline (Day 4-5)
1. Implement `schedulers/ddim_4step.py`
2. Implement `pipeline.py`
3. Test pipeline standalone (without ComfyUI)

### Step 6: Nodes (Day 5-6)
1. Implement loader node
2. Implement upscale node
3. Implement advanced nodes
4. Wire up `__init__.py` registration

### Step 7: Testing (Day 6-7)
1. Unit tests
2. Integration testing with ComfyUI
3. Quality comparison to upstream

### Step 8: Polish (Day 7+)
1. README documentation
2. Example workflows
3. Error message improvements
4. Optional: Tiling support

---

## Critical Pitfalls to Avoid

| Mistake | Correct Approach |
|---------|------------------|
| Hardcoded VAE scaling 0.18215 | Use `vae.config.scaling_factor` |
| State tensors as Python lists | Use safetensors for serialization |
| Naive spatial tiling | Flow/warp on full frame, tile only diffusion |
| Using ComfyUI's KSampler | Self-contained pipeline owns denoising loop |
| Claiming MIT license | Use Apache-2.0, include NOTICE file |
| BHWC tensors in state | BCHW format, convert at I/O boundaries |

---

## Open Questions for Upstream Investigation

1. **VAE encode method:** `latent_dist.sample()` vs `.mode()`?
2. **ARTG injection layers:** Which decoder blocks receive features?
3. **Flow model:** RAFT-small? RAFT-large? Custom weights?
4. **First frame handling:** Zero temporal features? Skip ARTG?
5. **Scheduler sigmas:** What timestep schedule does 4-step use?
6. **TPM in decoder:** How are multi-scale features fused?

---

*Plan created: January 2026*
