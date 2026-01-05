"""
Optical flow utilities for Stream-DiffVSR.

Provides image warping using flow fields for temporal alignment.
"""

import torch
import torch.nn.functional as F
from typing import Tuple, Union


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


def flow_warp(
    x: torch.Tensor,
    flow: torch.Tensor,
    interp_mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Warp an image or feature map with optical flow.
    
    This is the upstream implementation from Stream-DiffVSR.
    Handles both (B, 2, H, W) and (B, H, W, 2) flow formats.

    Args:
        x: Input tensor (B, C, H, W)
        flow: Flow field (B, 2, H, W) or (B, H, W, 2)
              Flow convention: flow[y, x] = (dx, dy)
        interp_mode: 'nearest' or 'bilinear'
        padding_mode: 'zeros', 'border', or 'reflection'

    Returns:
        Warped tensor (B, C, H, W)
    """
    # Handle flow format
    if flow.dim() == 4 and flow.shape[1] == 2:
        # (B, 2, H, W) -> (B, H, W, 2)
        flow = flow.permute(0, 2, 3, 1)

    # Cast flow to match input dtype (RAFT outputs float32, input may be float16)
    flow = flow.to(dtype=x.dtype, device=x.device)

    assert x.size()[-2:] == flow.size()[1:3], \
        f"Image size {x.size()[-2:]} doesn't match flow size {flow.size()[1:3]}"
    
    _, _, H, W = x.size()
    
    # Create mesh grid
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, H, device=x.device, dtype=x.dtype),
        torch.arange(0, W, device=x.device, dtype=x.dtype),
        indexing="ij"
    )
    grid = torch.stack((grid_x, grid_y), dim=2)  # (H, W, 2)
    grid = grid.unsqueeze(0).expand(flow.shape[0], -1, -1, -1)  # (B, H, W, 2)
    
    # Add flow to grid
    vgrid = grid + flow
    
    # Normalize to [-1, 1] for grid_sample
    vgrid_x = 2.0 * vgrid[..., 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    
    # Sample
    output = F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=False,
    )
    
    return output


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
    return flow_warp(image, flow, interp_mode=mode, padding_mode=padding_mode)


def upscale_flow(
    flow: torch.Tensor,
    scale_factor: int,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Upscale flow field and adjust values for new resolution.

    Args:
        flow: Flow field (B, 2, H, W) or (B, H, W, 2)
        scale_factor: Upscale factor (e.g., 4 for 4x upscaling)
        mode: Interpolation mode

    Returns:
        Upscaled flow in same format as input
    """
    # Handle (B, H, W, 2) format
    if flow.dim() == 4 and flow.shape[-1] == 2:
        flow_bchw = flow.permute(0, 3, 1, 2)  # (B, H, W, 2) -> (B, 2, H, W)
        upscaled = F.interpolate(
            flow_bchw,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )
        upscaled = upscaled * scale_factor
        return upscaled.permute(0, 2, 3, 1)  # Back to (B, H, W, 2)
    else:
        upscaled = F.interpolate(
            flow,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=False,
        )
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
