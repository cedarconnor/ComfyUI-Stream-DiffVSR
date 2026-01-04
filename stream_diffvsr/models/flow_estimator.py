"""
Optical flow estimation for Stream-DiffVSR.

Uses RAFT-Large from torchvision for high-quality optical flow estimation
between consecutive frames. The flow is used to warp the previous HQ frame
to align with the current frame for temporal guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlowEstimator(nn.Module):
    """
    RAFT-Large optical flow estimator.

    Estimates optical flow between consecutive frames for temporal alignment.
    The flow is computed on bicubic-upscaled 4x images (not LQ resolution).
    """

    def __init__(
        self,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Initialize RAFT-Large flow estimator.

        Args:
            device: Target device (default: cuda if available)
            dtype: Model dtype (RAFT works best in float32)
        """
        super().__init__()
        
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.dtype = dtype

        # Load RAFT-Large from torchvision
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            self.model = self.model.to(self.device).eval()
            self.model.requires_grad_(False)
        except ImportError:
            raise ImportError(
                "torchvision with optical flow support is required. "
                "Install with: pip install torchvision>=0.15.0"
            )

    @torch.inference_mode()
    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        rescale_factor: int = 1,
    ) -> torch.Tensor:
        """
        Estimate optical flow from source to target.

        Args:
            target: Target frame (B, C, H, W) - current frame
            source: Source frame (B, C, H, W) - previous frame
            rescale_factor: Optional downscale factor for flow computation

        Returns:
            Flow field (B, H, W, 2) in pixel coordinates
            
        Note:
            Flow convention: flow[y, x] = (dx, dy) where
            source[y + dy, x + dx] ≈ target[y, x]
        """
        # Ensure inputs are on correct device
        target = target.to(self.device, self.dtype)
        source = source.to(self.device, self.dtype)

        # RAFT expects [0, 255] range, convert from [-1, 1] or [0, 1]
        if target.min() < 0:
            # [-1, 1] range
            target_255 = ((target + 1) / 2 * 255).clamp(0, 255)
            source_255 = ((source + 1) / 2 * 255).clamp(0, 255)
        elif target.max() <= 1:
            # [0, 1] range
            target_255 = (target * 255).clamp(0, 255)
            source_255 = (source * 255).clamp(0, 255)
        else:
            # Already in [0, 255]
            target_255 = target
            source_255 = source

        # Compute flow (returns list of flow predictions, use last one)
        flows = self.model(target_255, source_255)
        flow = flows[-1]  # Best prediction

        # Optionally rescale flow
        if rescale_factor != 1:
            flow = F.interpolate(
                flow / rescale_factor,  # Use float division, not floor division
                scale_factor=1 / rescale_factor,
                mode='bilinear',
                align_corners=False,
            )

        # Convert to (B, H, W, 2) format
        flow = flow.permute(0, 2, 3, 1)

        return flow

    def to(self, device=None, dtype=None, *args, **kwargs):
        """Move model to device."""
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        if dtype is not None:
            self.dtype = dtype
        return self


class ZeroFlowEstimator(nn.Module):
    """
    Dummy flow estimator that returns zero flow.

    Used when optical flow estimation is disabled or as fallback.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        rescale_factor: int = 1,
    ) -> torch.Tensor:
        """Return zero flow field."""
        B, C, H, W = target.shape
        return torch.zeros(B, H, W, 2, device=target.device, dtype=target.dtype)

    def to(self, *args, **kwargs):
        """No-op for compatibility."""
        return self


def compute_flows(
    flow_estimator: FlowEstimator,
    images: list,
    rescale_factor: int = 1,
) -> list:
    """
    Compute forward optical flows for a sequence of images.

    Args:
        flow_estimator: FlowEstimator instance
        images: List of image tensors (B, C, H, W)
        rescale_factor: Optional downscale factor

    Returns:
        List of flow tensors, one for each consecutive pair
        flows[i] warps images[i] to images[i+1]
    """
    print('[Stream-DiffVSR] Computing optical flows...')
    
    forward_flows = []
    for i in range(1, len(images)):
        prev_image = images[i - 1]
        cur_image = images[i]
        flow = flow_estimator(cur_image, prev_image, rescale_factor=rescale_factor)
        forward_flows.append(flow)
    
    return forward_flows


def get_flow_estimator(
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    use_raft: bool = True,
) -> FlowEstimator:
    """
    Get optical flow estimator.

    First checks if RAFT is available from torchvision.
    Falls back to ZeroFlowEstimator if not available.

    Args:
        device: Target device
        dtype: Model dtype (RAFT works best in float32)
        use_raft: Whether to use RAFT (if False, returns ZeroFlowEstimator)

    Returns:
        FlowEstimator or ZeroFlowEstimator instance
    """
    if not use_raft:
        print("[Stream-DiffVSR] Flow estimation disabled")
        return ZeroFlowEstimator()

    try:
        estimator = FlowEstimator(device=device, dtype=dtype)
        print("[Stream-DiffVSR] Using RAFT-Large for optical flow")
        return estimator
    except ImportError as e:
        print(f"[Stream-DiffVSR] RAFT not available: {e}")
        print("[Stream-DiffVSR] Falling back to zero flow (no temporal guidance)")
        return ZeroFlowEstimator()
