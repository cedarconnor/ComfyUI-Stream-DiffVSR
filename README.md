Very experimental.................................................................

# ComfyUI-Stream-DiffVSR

ComfyUI custom nodes for **Stream-DiffVSR** video super-resolution with temporal consistency.

![Stream-DiffVSR](https://img.shields.io/badge/Stream--DiffVSR-v1.1-blue) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

## Features

- **4x Video Upscaling** - High-quality 4x super-resolution using diffusion
- **Temporal Consistency** - ControlNet-based temporal guidance prevents flickering
- **All-in-One Video Node** - Upscale entire videos with automatic chunking
- **4-Step Inference** - Distilled model optimized for fast 4-step denoising
- **Auto-Download** - Models automatically downloaded from HuggingFace
- **Automatic Tiling** - Optical flow auto-tiles on OOM for high-resolution inputs

## Installation

### 1. Install custom nodes
Navigate to your ComfyUI custom nodes directory and clone this repository:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/cedarconnor/ComfyUI-Stream-DiffVSR
cd ComfyUI-Stream-DiffVSR
pip install -r requirements.txt
```

### 2. Download models
You have two options for model setup:

**Option A: Auto-Download (Recommended)**
The nodes will automatically download models from HuggingFace to your cache (`~/.cache/huggingface/`) upon first run.

**Option B: Manual Download**
Download the model files from [Jamichsu/Stream-DiffVSR](https://huggingface.co/Jamichsu/Stream-DiffVSR) and place in:

```
ComfyUI/models/StreamDiffVSR/v1/
├── controlnet/
├── unet/
├── vae/
└── scheduler/
```

### 3. Restart ComfyUI
Restart your ComfyUI server to load the new nodes.

## Nodes Reference

### StreamDiffVSR_Loader
Initializes the model pipeline.

- **model_version**: Select model version (default: `v1`).
- **device**: Compute device (`cuda`, `cpu`, or `auto`).
- **dtype**: Precision (`float16` for GPUs, `float32` for CPU).

### StreamDiffVSR_UpscaleVideo ⭐ (Recommended)
**All-in-one node for processing videos (File or Tensor).**

This node supports two modes:
1. **File Mode**: Reads/writes directly to disk (Low RAM, infinite length).
2. **Tensor Mode**: Compatible with standard ComfyUI `Load Video`/`Save Video`.

**Features:**
- Automatic chunking (manages VRAM)
- Temporal state continuity
- Supports both file paths and IMAGE tensors

**Inputs:**
- **pipe**: Connection from `StreamDiffVSR_Loader`
- **video_path** (Optional): Path to input file (File Mode)
- **images** (Optional): Input frames tensor (Tensor Mode)

**Parameters:**
- **output_path**: Output path (File Mode only)
- **frames_per_batch**: Frames to process per chunk (default: `16`). 
- **start_frame/end_frame**: Frame range to process
- **num_inference_steps**: Denoising steps (default: `4`)

> **Note:** For Tensor Mode, ensure you have enough system RAM to hold the entire upscaled video. For very long videos, use File Mode.

### StreamDiffVSR_Upscale
Upscale a batch of frames from memory (IMAGE tensor input).

Use this when:
- You're loading frames from VHS or other video loaders
- You need more control over frame loading
- You're integrating with other ComfyUI video workflows

**Inputs:**
- **pipe**: Connection from `StreamDiffVSR_Loader`
- **images**: Batch of frames (BHWC format)
- **state**: Optional state for temporal continuity

### State Management (Advanced)
For advanced workflows that need to chain multiple upscale operations:

- **StreamDiffVSR_CreateState**: Create empty state
- **StreamDiffVSR_ExtractState**: Extract frames from state for inspection

## Example Workflows

### Simple Video Upscale (Recommended)
```
[StreamDiffVSR_Loader] ──► [StreamDiffVSR_UpscaleVideo]
                                    │
                                    └── video_path: "input.mp4"
                                    └── output_path: "" (auto-generates)
                                    └── frames_per_batch: 16
```

### With VHS Integration
```
[Load Video (VHS)] ──► [StreamDiffVSR_Upscale] ──► [Video Combine (VHS)]
        ▲                       ▲
        │                       │
[StreamDiffVSR_Loader] ─────────┘
```

## VRAM Requirements

| Input Resolution | Batch Size | Est. VRAM |
|------------------|------------|-----------|
| 480p             | 16 frames  | ~6 GB     |
| 720p             | 8 frames   | ~10 GB    |
| 1080p            | 4 frames   | ~18 GB    |

> **Tip:** If you run out of VRAM, reduce `frames_per_batch`. The optical flow will automatically tile on OOM.

## Technical Details

Stream-DiffVSR uses a recurrent architecture:
1. **RAFT Optical Flow**: Aligns the previous high-quality (HQ) output to the current frame
2. **ControlNet**: Uses the warped previous HQ frame as a condition to guide diffusion
3. **TPM VAE**: A temporal VAE fuses features from the previous timestep

## Troubleshooting

- **Out of Memory (OOM)**: Reduce `frames_per_batch` in UpscaleVideo or batch size in Upscale node
- **Missing ffmpeg**: Ensure `ffmpeg` and `ffprobe` are installed and in your PATH
- **diffusers Import Error**: Run `pip install diffusers>=0.30.0,<0.32.0`
- **Flickering**: Ensure state is properly passed between batches (UpscaleVideo handles this automatically)

## License

Apache-2.0
