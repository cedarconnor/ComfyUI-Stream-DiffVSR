# ComfyUI-Stream-DiffVSR

ComfyUI custom nodes for **Stream-DiffVSR** video super-resolution with temporal consistency.

## Features

- **4x Video Upscaling** - High-quality 4x super-resolution using diffusion
- **Temporal Consistency** - Auto-regressive temporal guidance prevents flickering
- **4-Step Inference** - Distilled model optimized for fast 4-step denoising
- **State Management** - Process long videos in chunks with seamless continuity

## Installation

1. Clone into ComfyUI custom nodes:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cedarconnor/ComfyUI-Stream-DiffVSR
cd ComfyUI-Stream-DiffVSR
pip install -r requirements.txt
```

2. Download model weights from [HuggingFace](https://huggingface.co/Jamichsu/Stream-DiffVSR) and place in:
```
ComfyUI/models/StreamDiffVSR/v1/
├── unet/
│   └── diffusion_pytorch_model.safetensors
├── artg/
│   └── artg_module.safetensors
├── temporal_decoder/
│   └── temporal_decoder.safetensors
├── vae/
│   └── vae_tiny.safetensors
└── flow/  (optional)
    └── raft_small.pth
```

3. Restart ComfyUI

## Nodes

### StreamDiffVSR_Loader
Load all model components. Returns a pipeline object for inference.

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
- **Distilled U-Net** initialized from SD x4 Upscaler
- **ARTG Module** for temporal feature injection
- **Temporal-Aware Decoder** with TPM for consistency
- **RAFT** optical flow for motion alignment

The batch dimension represents frame order. Frame N uses frame N-1's
upscaled output (warped by optical flow) for temporal guidance.

## License

Apache-2.0 (matching upstream Stream-DiffVSR)

## Credits

- **Stream-DiffVSR**: [jamichss/Stream-DiffVSR](https://github.com/jamichss/Stream-DiffVSR)
- **ComfyUI Integration**: Cedar

## Status

⚠️ **Development**: Model wrappers are stubs pending upstream architecture study.

The project structure and ComfyUI integration is complete. Full functionality
requires implementing model forward passes based on upstream code.
