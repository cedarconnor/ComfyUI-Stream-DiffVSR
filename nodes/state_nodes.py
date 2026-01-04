"""
State management nodes for Stream-DiffVSR.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Optional

from ..stream_diffvsr.state import StreamDiffVSRState
from ..stream_diffvsr.utils.image_utils import bhwc_to_bchw, bchw_to_bhwc, normalize_to_neg1_1, denormalize_from_neg1_1


class StreamDiffVSR_CreateState:
    """
    Create an empty state for starting a new sequence.

    The first frame will be processed without temporal guidance
    when using this empty state.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "initial_frame": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional: Pre-populate state with an already-upscaled frame. "
                            "Use this to seamlessly continue from a previously processed video."
                        ),
                    },
                ),
                "initial_lq_frame": (
                    "IMAGE",
                    {
                        "tooltip": (
                            "Optional: The LQ frame corresponding to initial_frame. "
                            "Required for optical flow if initial_frame is provided."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STREAM_DIFFVSR_STATE",)
    RETURN_NAMES = ("state",)
    FUNCTION = "create_state"
    CATEGORY = "StreamDiffVSR/Advanced"
    DESCRIPTION = "Create empty state for starting a new sequence"

    def create_state(
        self,
        initial_frame: Optional[torch.Tensor] = None,
        initial_lq_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[StreamDiffVSRState]:
        """
        Create a new state, optionally pre-populated.

        Args:
            initial_frame: Optional HQ frame to use as previous
            initial_lq_frame: Optional LQ frame for flow estimation

        Returns:
            New state
        """
        if initial_frame is not None:
            # Convert to BCHW for state storage
            hq_bchw = bhwc_to_bchw(initial_frame)
            # Normalize HQ to [-1, 1] range as expected by pipeline
            hq_bchw = normalize_to_neg1_1(hq_bchw)

            lq_upscaled_bchw = None
            if initial_lq_frame is not None:
                lq_bchw = bhwc_to_bchw(initial_lq_frame)
                # Upscale LQ to 4x to match previous_lq_upscaled format
                lq_upscaled_bchw = F.interpolate(
                    lq_bchw, scale_factor=4, mode='bicubic', align_corners=False
                )

            state = StreamDiffVSRState(
                previous_hq=hq_bchw,
                previous_lq_upscaled=lq_upscaled_bchw,
                frame_index=1,  # Start at 1 since we have a "previous" frame
            )
        else:
            state = StreamDiffVSRState()

        return (state,)


class StreamDiffVSR_ExtractState:
    """
    Extract information from a state object.

    Useful for debugging or saving the previous HQ frame.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "state": (
                    "STREAM_DIFFVSR_STATE",
                    {
                        "tooltip": "State to extract from",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "BOOLEAN")
    RETURN_NAMES = ("previous_hq", "previous_lq_upscaled", "frame_index", "has_previous")
    FUNCTION = "extract_state"
    CATEGORY = "StreamDiffVSR/Advanced"
    DESCRIPTION = "Extract previous frame and info from state"

    def extract_state(
        self,
        state: StreamDiffVSRState,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, bool]:
        """
        Extract state contents.

        Args:
            state: State to extract from

        Returns:
            (previous_hq, previous_lq, frame_index, has_previous)
        """
        # Convert from BCHW to BHWC for ComfyUI
        previous_hq = None
        if state.previous_hq is not None:
            # Denormalize from [-1, 1] to [0, 1] for ComfyUI
            hq_denorm = denormalize_from_neg1_1(state.previous_hq)
            previous_hq = bchw_to_bhwc(hq_denorm)

        previous_lq_upscaled = None
        if state.previous_lq_upscaled is not None:
            previous_lq_upscaled = bchw_to_bhwc(state.previous_lq_upscaled)

        # Create placeholder images if None
        if previous_hq is None:
            previous_hq = torch.zeros(1, 64, 64, 3)
        if previous_lq_upscaled is None:
            previous_lq_upscaled = torch.zeros(1, 64, 64, 3)  # 4x upscaled placeholder

        return (previous_hq, previous_lq_upscaled, state.frame_index, state.has_previous)
