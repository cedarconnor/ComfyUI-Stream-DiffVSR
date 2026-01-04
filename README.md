# ComfyUI-Stream-DiffVSR

ComfyUI custom nodes for **Stream-DiffVSR** video super-resolution with temporal consistency.

![Stream-DiffVSR](https://img.shields.io/badge/Stream--DiffVSR-v1-blue) ![License](https://img.shields.io/badge/License-Apache%202.0-green)

## Features

- **4x Video Upscaling** - High-quality 4x super-resolution using diffusion
- **Temporal Consistency** - ControlNet-based temporal guidance prevents flickering
- **4-Step Inference** - Distilled model optimized for fast 4-step denoising
- **State Management** - Process long videos in chunks with seamless continuity
- **Auto-Download** - Models automatically downloaded from HuggingFace

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

**Option A: Manual Download (Recommended)**
Download the model files from [Jamichsu/Stream-DiffVSR](https://huggingface.co/Jamichsu/Stream-DiffVSR) and verify the structure matches exactly:

```
ComfyUI/models/StreamDiffVSR/v1/
├── controlnet/
├── unet/
├── vae/
└── scheduler/
```

**Option B: Auto-Download**
If you skip manual download, the nodes will automatically download models from HuggingFace to your cache (`~/.cache/huggingface/`) upon first run. This requires internet access.

### 3. Restart ComfyUI
Restart your ComfyUI server to load the new nodes.

## Nodes Reference

### 1. StreamDiffVSR_Loader
Initializes the model pipeline.

- **model_version**: Select model version (default: `v1`).
- **device**: Compute device (`cuda`, `cpu`, or `auto`).
- **dtype**: Precision (`float16` for GPUs, `float32` for CPU).
    - *Tip*: Use `float16` to save VRAM on supported GPUs.
- **use_local_models**: Attempt to load from `ComfyUI/models/StreamDiffVSR/` first.
- **use_huggingface**: Download from HuggingFace if local load fails.

### 2. StreamDiffVSR_Upscale
The main node for processing video frames.

**Inputs:**
- **pipe**: Connection from `StreamDiffVSR_Loader`.
- **images**: Batch of frames to upscale (BHWC format).
- **state**: (Optional) Connection from a previous `StreamDiffVSR_Upscale` or `StreamDiffVSR_CreateState` node. Required for temporal continuity between chunks.

**Parameters:**
- **num_inference_steps**: Denoising steps (default: `4`).
- **seed**: Seed for noise generation.
- **guidance_scale**: CFG scale. default `0.0` (disabled) is recommended for this model.
- **controlnet_scale**: Strength of temporal guidance (default: `1.0`). Lower values reduce consistency but may increase sharpness.
- **enable_tiling**: (Experimental) Enable to reduce VRAM usage for high-res inputs.
- **tile_size**: Size of tiles (default: `512`).
- **tile_overlap**: Pixels of overlap between tiles (default: `64`).

### 3. State Management (Advanced)
Nodes for handling long-video processing where you need to break the video into batches.

**StreamDiffVSR_CreateState**
Creates a new state object.
- **initial_frame/initial_lq_frame**: Optionally pre-populate state to continue from a specific frame (e.g. from a separate workflow).

**StreamDiffVSR_ExtractState**
- Extracts the `previous_hq` and `previous_lq_upscaled` frames from a state object for inspection or saving.

## Workflows

### Basic Workflow
For short videos that fit in VRAM (e.g., < 24 frames at 1080p output).

1. **Load Video** -> **StreamDiffVSR_Upscale** (images input)
2. **StreamDiffVSR_Loader** -> **StreamDiffVSR_Upscale** (pipe input)
3. **StreamDiffVSR_Upscale** (images output) -> **Save Video**

### Long Video Workflow (Chunked)
To process long videos without OOM errors, process in batches.

1. **Batch 1**:
    - **Load Video (Frames 0-15)** -> `Upscale Node 1`
    - `Upscale Node 1` (state output) -> Connect to next node...

2. **Batch 2**:
    - **Load Video (Frames 16-31)** -> `Upscale Node 2`
    - Connect `Upscale Node 1` (state) -> `Upscale Node 2` (state input)

*Note: In ComfyUI, you typically use a loop or a custom script to handle this sequentially.*

## VRAM Estimation

| Input Resolution | Tiling | Est. VRAM |
|------------------|--------|-----------|
| 480p             | Off    | ~6 GB     |
| 720p             | Off    | ~10 GB    |
| 1080p            | Off    | ~18 GB    |
| 720p             | 512px  | ~6 GB     |

## Technical Details

Stream-DiffVSR uses a recurrent architecture:
1. **RAFT Optical Flow**: Aligns the previous high-quality (HQ) output to the current frame.
2. **ControlNet**: Uses the warped previous HQ frame as a condition to guide the diffusion process for the current frame.
3. **TPM VAE**: A temporal VAE fuses features from the previous timestep to decode the final image.

## Troubleshooting

- **Out of Memory (OOM)**: Be sure `enable_tiling` is checked if you are hitting limits, or reduce batch size.
- **Missing Dependencies**: Ensure `torchvision` is installed for RAFT optical flow support.
- **Flickering**: Ensure `state` is correctly passed between batches. If `state` is disconnected, each batch starts fresh without temporal knowledge.

## License

Apache-2.0
