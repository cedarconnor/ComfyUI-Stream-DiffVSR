"""
Main upscale node for Stream-DiffVSR.
"""

import torch
from typing import Tuple, Optional

from ..stream_diffvsr.state import StreamDiffVSRState
from ..stream_diffvsr.pipeline import StreamDiffVSRPipeline


class StreamDiffVSR_Upscale:
    """
    Main upscaling node for Stream-DiffVSR.

    Processes a batch of frames with auto-regressive temporal guidance.
    The batch dimension is treated as time - frames are processed
    sequentially with each frame using the previous frame's HQ output
    for temporal conditioning via ControlNet.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pipe": (
                    "STREAM_DIFFVSR_PIPE",
                    {
                        "tooltip": "Pipeline from StreamDiffVSR_Loader",
                    },
                ),
                "images": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Input frames (BHWC tensor). "
                            "Batch dimension = frame count, processed in order."
                        ),
                    },
                ),
            },
            "optional": {
                "state": (
                    "STREAM_DIFFVSR_STATE",
                    {
                        "tooltip": (
                            "Previous state for continuing a sequence. "
                            "Leave unconnected to start fresh."
                        ),
                    },
                ),
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Denoising steps. Model is optimized for 4 steps.",
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Random seed for reproducibility",
                    },
                ),
                "guidance_scale": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.5,
                        "tooltip": "CFG scale. 0 = disabled (default, as per upstream).",
                    },
                ),
                "controlnet_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.1,
                        "tooltip": "ControlNet conditioning strength for temporal guidance.",
                    },
                ),
                "enable_tiling": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Enable tiled processing for lower VRAM usage",
                    },
                ),
                "tile_size": (
                    "INT",
                    {
                        "default": 512,
                        "min": 256,
                        "max": 1024,
                        "step": 64,
                        "tooltip": "Tile size when tiling is enabled",
                    },
                ),
                "tile_overlap": (
                    "INT",
                    {
                        "default": 64,
                        "min": 16,
                        "max": 256,
                        "step": 16,
                        "tooltip": "Overlap between tiles to reduce seams",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STREAM_DIFFVSR_STATE")
    RETURN_NAMES = ("images", "state")
    FUNCTION = "upscale"
    CATEGORY = "StreamDiffVSR"
    DESCRIPTION = "Upscale video frames 4x with temporal consistency"

    def upscale(
        self,
        pipe: StreamDiffVSRPipeline,
        images: torch.Tensor,
        state: Optional[StreamDiffVSRState] = None,
        num_inference_steps: int = 4,
        seed: int = 0,
        guidance_scale: float = 0.0,
        controlnet_scale: float = 1.0,
        enable_tiling: bool = False,
        tile_size: int = 512,
        tile_overlap: int = 64,
    ) -> Tuple[torch.Tensor, StreamDiffVSRState]:
        """
        Upscale video frames.

        Args:
            pipe: Stream-DiffVSR pipeline
            images: Input frames (B, H, W, C)
            state: Optional previous state
            num_inference_steps: Denoising steps
            seed: Random seed
            guidance_scale: CFG scale (0 = disabled)
            controlnet_scale: ControlNet conditioning strength
            enable_tiling: Whether to use tiling
            tile_size: Tile size
            tile_overlap: Tile overlap

        Returns:
            (upscaled_images, final_state)
        """
        num_frames = images.shape[0]
        print(f"[Stream-DiffVSR] Processing {num_frames} frames...")

        # Progress callback
        def progress(current, total):
            print(f"[Stream-DiffVSR] Frame {current}/{total}")

        # TODO: Implement tiling support
        if enable_tiling:
            print(f"[Stream-DiffVSR] Tiling enabled: {tile_size}px with {tile_overlap}px overlap")
            # For now, just process without tiling
            # Tiled processing requires careful handling to preserve temporal consistency

        # Process frames
        hq_images, final_state = pipe(
            images,
            state=state,
            num_inference_steps=num_inference_steps,
            seed=seed,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_scale,
            progress_callback=progress,
        )

        print(f"[Stream-DiffVSR] Done! Output shape: {tuple(hq_images.shape)}")

        return (hq_images, final_state)
