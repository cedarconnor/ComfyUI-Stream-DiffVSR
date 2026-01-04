"""
Optical flow utilities for Stream-DiffVSR.

Provides image warping using flow fields for temporal alignment.
"""

import torch
import torch.nn.functional as F
from typing import Tuple


def create_meshgrid(
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create meshgrid for image coordinates.

    Args:
        height: Image height
        width: Image width
        device: Target device
        dtype: Data type

    Returns:
        (grid_x, grid_y): Coordinate grids of shape (H, W)
    """
    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return grid_x, grid_y


def warp_image(
    image: torch.Tensor,
    flow: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> torch.Tensor:
    """
    Warp image using optical flow (backward warping).

    Given flow from frame A to frame B, this warps frame A to align with B.
    Flow convention: flow[y, x] = (dx, dy) where the pixel at (x, y) in the
    output comes from (x + dx, y + dy) in the input.

    Args:
        image: Input image tensor (B, C, H, W)
        flow: Flow field (B, 2, H, W) where flow[:, 0] is horizontal (x)
              and flow[:, 1] is vertical (y) displacement
        mode: Interpolation mode ("bilinear" or "nearest")
        padding_mode: Padding mode ("zeros", "border", or "reflection")

    Returns:
        Warped image (B, C, H, W)
    """
    B, C, H, W = image.shape
    _, flow_channels, flow_H, flow_W = flow.shape

    if flow_channels != 2:
        raise ValueError(f"Flow must have 2 channels, got {flow_channels}")

    # Resize flow if dimensions don't match
    if flow_H != H or flow_W != W:
        flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=False)
        # Scale flow values proportionally
        flow = flow * torch.tensor(
            [W / flow_W, H / flow_H],
            device=flow.device,
            dtype=flow.dtype,
        ).view(1, 2, 1, 1)

    # Create sampling grid
    grid_x, grid_y = create_meshgrid(H, W, image.device, image.dtype)

    # Add flow to grid coordinates
    # flow[:, 0] is horizontal (x), flow[:, 1] is vertical (y)
    sample_x = grid_x.unsqueeze(0) + flow[:, 0]
    sample_y = grid_y.unsqueeze(0) + flow[:, 1]

    # Normalize to [-1, 1] for grid_sample
    sample_x = 2.0 * sample_x / (W - 1) - 1.0
    sample_y = 2.0 * sample_y / (H - 1) - 1.0

    # Stack into grid format (B, H, W, 2)
    grid = torch.stack([sample_x, sample_y], dim=-1)

    # Warp image
    warped = F.grid_sample(
        image,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )

    return warped


def upscale_flow(
    flow: torch.Tensor,
    scale_factor: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Upscale flow field and adjust values for new resolution.

    Args:
        flow: Flow field (B, 2, H, W)
        scale_factor: Upscale factor (e.g., 4 for 4x upscaling)
        mode: Interpolation mode

    Returns:
        Upscaled flow (B, 2, H*scale, W*scale)
    """
    upscaled = F.interpolate(
        flow,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=False,
    )
    # Scale flow values by the same factor
    upscaled = upscaled * scale_factor
    return upscaled


def compute_occlusion_mask(
    flow_forward: torch.Tensor,
    flow_backward: torch.Tensor,
    threshold: float = 1.0,
) -> torch.Tensor:
    """
    Compute occlusion mask using forward-backward consistency check.

    Pixels where forward and backward flows are inconsistent are likely
    occluded and should not use temporal guidance.

    Args:
        flow_forward: Forward flow (B, 2, H, W)
        flow_backward: Backward flow (B, 2, H, W)
        threshold: Consistency threshold in pixels

    Returns:
        Occlusion mask (B, 1, H, W) where 1 = occluded, 0 = visible
    """
    # Warp backward flow using forward flow
    warped_backward = warp_image(flow_backward, flow_forward)

    # Check consistency: forward + warped_backward should be ~0
    consistency = flow_forward + warped_backward

    # Compute magnitude of inconsistency
    inconsistency = torch.sqrt(
        consistency[:, 0:1] ** 2 + consistency[:, 1:2] ** 2
    )

    # Threshold to get occlusion mask
    occlusion = (inconsistency > threshold).float()

    return occlusion
