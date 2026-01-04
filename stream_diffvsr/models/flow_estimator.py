"""
Optical flow estimation for Stream-DiffVSR.

Provides flow estimation between frames for temporal alignment.
Supports RAFT and can reuse existing models from other ComfyUI nodes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlowEstimator(nn.Module):
    """
    Base class for optical flow estimation.

    Estimates flow from previous frame to current frame.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        prev_frame: torch.Tensor,
        curr_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate optical flow.

        Args:
            prev_frame: Previous frame (B, C, H, W), normalized to [0, 1]
            curr_frame: Current frame (B, C, H, W), normalized to [0, 1]

        Returns:
            Flow field (B, 2, H, W) where:
                flow[:, 0] = horizontal displacement (x)
                flow[:, 1] = vertical displacement (y)
        """
        raise NotImplementedError


class RAFTFlowEstimator(FlowEstimator):
    """
    RAFT-based optical flow estimator.

    Uses RAFT (Recurrent All-Pairs Field Transforms) for accurate
    optical flow estimation.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
        num_iterations: int = 12,
    ):
        """
        Initialize RAFT flow estimator.

        Args:
            model_path: Path to RAFT weights (optional, uses torchvision if None)
            device: Target device
            dtype: Data type
            num_iterations: Number of RAFT iterations (more = better, slower)
        """
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.num_iterations = num_iterations
        self._model = None
        self._model_path = model_path

    def _load_model(self):
        """Lazy load the RAFT model."""
        if self._model is not None:
            return

        try:
            # Try torchvision RAFT first (simpler, no extra dependencies)
            from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

            self._model = raft_small(weights=Raft_Small_Weights.DEFAULT)
            self._model = self._model.to(self.device)
            self._model.eval()
            self._use_torchvision = True
            print("[Stream-DiffVSR] Using torchvision RAFT-small for optical flow")

        except ImportError:
            # Fall back to loading from weights file
            if self._model_path:
                # TODO: Load RAFT from weights file
                raise NotImplementedError(
                    "Custom RAFT weights loading not yet implemented. "
                    "Please install torchvision>=0.14.0 for RAFT support."
                )
            else:
                raise ImportError(
                    "Optical flow requires torchvision>=0.14.0 with RAFT support. "
                    "Install with: pip install torchvision>=0.14.0"
                )

    @torch.inference_mode()
    def forward(
        self,
        prev_frame: torch.Tensor,
        curr_frame: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate optical flow using RAFT.

        Args:
            prev_frame: Previous frame (B, C, H, W), [0, 1] range
            curr_frame: Current frame (B, C, H, W), [0, 1] range

        Returns:
            Flow field (B, 2, H, W)
        """
        self._load_model()

        # Convert to expected format for torchvision RAFT
        # RAFT expects uint8 or float in [0, 255] range
        prev = (prev_frame * 255.0).to(self.device)
        curr = (curr_frame * 255.0).to(self.device)

        # Ensure float32 for RAFT
        prev = prev.float()
        curr = curr.float()

        # Run RAFT
        flow_predictions = self._model(prev, curr)

        # Return the final flow prediction
        # torchvision RAFT returns list of flows at different iterations
        return flow_predictions[-1]


class ZeroFlowEstimator(FlowEstimator):
    """
    Dummy flow estimator that returns zero flow.

    Used when optical flow is disabled or unavailable.
    Results in no temporal alignment (each frame processed independently).
    """

    def forward(
        self,
        prev_frame: torch.Tensor,
        curr_frame: torch.Tensor,
    ) -> torch.Tensor:
        """Return zero flow field."""
        B, C, H, W = curr_frame.shape
        return torch.zeros(B, 2, H, W, device=curr_frame.device, dtype=curr_frame.dtype)


def get_flow_estimator(
    device: torch.device,
    dtype: torch.dtype,
    model_path: Optional[str] = None,
    reuse_existing: bool = True,
) -> FlowEstimator:
    """
    Get optical flow estimator, reusing existing models if available.

    Checks for:
    1. ComfyUI-Frame-Interpolation's loaded RAFT
    2. ComfyUI-RAFT node's model
    3. Falls back to bundled/torchvision RAFT

    Args:
        device: Target device
        dtype: Data type
        model_path: Optional path to RAFT weights
        reuse_existing: Whether to try reusing existing models

    Returns:
        Flow estimator instance
    """
    if reuse_existing:
        # Try to find existing RAFT from other nodes
        try:
            # Check ComfyUI-Frame-Interpolation
            import ComfyUI_Frame_Interpolation

            if hasattr(ComfyUI_Frame_Interpolation, "get_raft_model"):
                raft = ComfyUI_Frame_Interpolation.get_raft_model()
                if raft is not None:
                    print("[Stream-DiffVSR] Reusing RAFT from ComfyUI-Frame-Interpolation")
                    # TODO: Wrap external RAFT
        except ImportError:
            pass

        try:
            # Check ComfyUI-RAFT
            from ComfyUI_RAFT import raft_model

            if raft_model is not None:
                print("[Stream-DiffVSR] Reusing RAFT from ComfyUI-RAFT")
                # TODO: Wrap external RAFT
        except ImportError:
            pass

    # Fall back to our own RAFT
    try:
        return RAFTFlowEstimator(
            model_path=model_path,
            device=device,
            dtype=dtype,
        )
    except ImportError as e:
        print(f"[Stream-DiffVSR] RAFT not available: {e}")
        print("[Stream-DiffVSR] Falling back to zero flow (no temporal guidance)")
        return ZeroFlowEstimator()
