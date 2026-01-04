"""
Single frame processing node for advanced use cases.
"""

import torch
from typing import Tuple

from ..stream_diffvsr.state import StreamDiffVSRState
from ..stream_diffvsr.pipeline import StreamDiffVSRPipeline


class StreamDiffVSR_ProcessFrame:
    """
    Process a single frame with explicit state input/output.

    Use this node for:
    - Custom frame processing loops
    - Integration with other temporal nodes (e.g., VHS)
    - Fine-grained control over state management
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
                "image": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Single frame (B=1). "
                            "If batch > 1, only first frame is processed."
                        ),
                    },
                ),
                "state": (
                    "STREAM_DIFFVSR_STATE",
                    {
                        "tooltip": "State from previous frame (or CreateState node)",
                    },
                ),
            },
            "optional": {
                "num_inference_steps": (
                    "INT",
                    {
                        "default": 4,
                        "min": 1,
                        "max": 50,
                        "step": 1,
                    },
                ),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "STREAM_DIFFVSR_STATE")
    RETURN_NAMES = ("image", "state")
    FUNCTION = "process_frame"
    CATEGORY = "StreamDiffVSR/Advanced"
    DESCRIPTION = "Process a single frame with explicit state management"

    def process_frame(
        self,
        pipe: StreamDiffVSRPipeline,
        image: torch.Tensor,
        state: StreamDiffVSRState,
        num_inference_steps: int = 4,
        seed: int = 0,
    ) -> Tuple[torch.Tensor, StreamDiffVSRState]:
        """
        Process a single frame.

        Args:
            pipe: Stream-DiffVSR pipeline
            image: Single input frame (B, H, W, C)
            state: Current state
            num_inference_steps: Denoising steps
            seed: Random seed

        Returns:
            (upscaled_frame, new_state)
        """
        # Take only first frame if batch > 1
        if image.shape[0] > 1:
            image = image[0:1]

        # Process frame
        hq_frame, new_state = pipe.process_frame(
            image,
            state,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        return (hq_frame, new_state)
