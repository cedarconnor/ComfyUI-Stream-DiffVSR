# ComfyUI-Stream-DiffVSR

ComfyUI custom nodes for **Stream-DiffVSR** video super-resolution with temporal consistency.

## Features

- **4x Video Upscaling** - High-quality 4x super-resolution using diffusion
- **Temporal Consistency** - ControlNet-based temporal guidance prevents flickering
- **4-Step Inference** - Distilled model optimized for fast 4-step denoising
- **State Management** - Process long videos in chunks with seamless continuity
- **Auto-Download** - Models automatically downloaded from HuggingFace

## Installation

### 1. Install custom nodes:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cedarconnor/ComfyUI-Stream-DiffVSR
cd ComfyUI-Stream-DiffVSR
pip install -r requirements.txt
```

### 2. Download models (Choose one option):

**Option A: Manual Download (Preferred)**

Download from [Jamichsu/Stream-DiffVSR](https://huggingface.co/Jamichsu/Stream-DiffVSR) and place in:
```
ComfyUI/models/StreamDiffVSR/v1/
├── controlnet/
├── unet/
├── vae/
└── scheduler/
```

**Option B: Auto-Download (Fallback)**

If local models are not found, they will be automatically downloaded from HuggingFace on first use. Models are cached in `~/.cache/huggingface/`.

### 3. Restart ComfyUI

## Nodes

### StreamDiffVSR_Loader
Load all model components from HuggingFace. Returns a pipeline object for inference.

- **model_version**: Model version (default: v1)
- **device**: cuda / cpu / auto
- **dtype**: float16 / bfloat16 / float32

### StreamDiffVSR_Upscale
Main upscaling node. Processes batches of frames with temporal guidance.

- **pipe**: Pipeline from loader
- **images**: Input frames (batch = frame sequence)
- **state**: (optional) Previous state for continuation
- **num_inference_steps**: Denoising steps (default: 4)
- **seed**: Random seed

### StreamDiffVSR_ProcessFrame (Advanced)
Process single frames with explicit state I/O. For custom loops.

### StreamDiffVSR_CreateState (Advanced)
Create empty state or pre-populate for seamless continuation.

### StreamDiffVSR_ExtractState (Advanced)
Extract previous frame and metadata from state.

## Usage

### Basic Workflow
```
[Load Video] → [StreamDiffVSR_Loader] → [StreamDiffVSR_Upscale] → [Save Video]
                      ↓
               [Pipeline]
```

### Processing Long Videos
For videos that don't fit in VRAM, process in chunks:

1. First chunk: No state input
2. Subsequent chunks: Connect state output → state input

This maintains temporal consistency across chunk boundaries.

## VRAM Requirements

| Resolution | Tiling | Estimated VRAM |
|------------|--------|----------------|
| 480p | Off | ~6 GB |
| 720p | Off | ~10 GB |
| 1080p | Off | ~18 GB |
| 720p | 512px | ~6 GB |

## Technical Details

Stream-DiffVSR uses:
- **ControlNet** for temporal guidance (warped previous HQ frame)
- **UNet2DConditionModel** from SD x4 Upscaler (distilled for 4-step)
- **TemporalAutoencoderTiny** with TPM for temporal feature fusion
- **RAFT-Large** optical flow for motion alignment

The batch dimension represents frame order. Frame N uses frame N-1's
upscaled output (warped by optical flow) for temporal guidance via ControlNet.

### Architecture

```
LQ Frame (t) ──────────────────────────────────────────────────────┐
     │                                                              │
     ├── Bicubic 4x ──── Optical Flow ◄──── Previous HQ (t-1)      │
     │                        │                     │               │
     │                        ▼                     │               │
     │                   Flow Warp ─────────────────┘               │
     │                        │                                     │
     │                        ▼                                     │
     │              Warped Previous HQ                              │
     │                   │         │                                │
     │                   │         ├── VAE Encode ──► TPM Features  │
     │                   │         │                       │        │
     │                   ▼         ▼                       │        │
     │              ControlNet ──► U-Net ◄─────────────────┘        │
     │                              │                               │
     │                              ▼                               │
     │                      Denoised Latents                        │
     │                              │                               │
     └────────────────────────────► VAE Decode (with TPM) ──► HQ Frame (t)
                                                                    │
                                               Store for t+1 ◄──────┘
```

## License

Apache-2.0 (matching upstream Stream-DiffVSR)

## Credits

- **Stream-DiffVSR**: [jamichss/Stream-DiffVSR](https://github.com/jamichss/Stream-DiffVSR)
- **HuggingFace Model**: [Jamichsu/Stream-DiffVSR](https://huggingface.co/Jamichsu/Stream-DiffVSR)
- **ComfyUI Integration**: Cedar

## Status

⚠️ **Development**: Model wrappers are being implemented based on upstream architecture.

Current progress:
- [x] Project structure and ComfyUI integration
- [x] Documentation and planning
- [ ] ControlNet temporal guidance implementation
- [ ] Temporal VAE with TPM
- [ ] Full pipeline integration
- [ ] Testing and validation
