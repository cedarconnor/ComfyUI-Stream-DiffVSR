"""
Image tensor utilities for Stream-DiffVSR.

Handles conversions between ComfyUI format (BHWC) and model format (BCHW),
as well as normalization between different value ranges.
"""

import torch
from typing import Tuple


def bhwc_to_bchw(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from ComfyUI format to model format.

    Args:
        tensor: (B, H, W, C) tensor

    Returns:
        (B, C, H, W) tensor
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (BHWC), got {tensor.ndim}D")
    return tensor.permute(0, 3, 1, 2).contiguous()


def bchw_to_bhwc(tensor: torch.Tensor) -> torch.Tensor:
    """
    Convert tensor from model format to ComfyUI format.

    Args:
        tensor: (B, C, H, W) tensor

    Returns:
        (B, H, W, C) tensor
    """
    if tensor.ndim != 4:
        raise ValueError(f"Expected 4D tensor (BCHW), got {tensor.ndim}D")
    return tensor.permute(0, 2, 3, 1).contiguous()


def normalize_to_neg1_1(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize tensor from [0, 1] to [-1, 1] range.

    Args:
        tensor: Tensor in [0, 1] range

    Returns:
        Tensor in [-1, 1] range
    """
    return tensor * 2.0 - 1.0


def denormalize_from_neg1_1(tensor: torch.Tensor) -> torch.Tensor:
    """
    Denormalize tensor from [-1, 1] to [0, 1] range.

    Args:
        tensor: Tensor in [-1, 1] range

    Returns:
        Tensor in [0, 1] range
    """
    return (tensor + 1.0) / 2.0


def ensure_range_01(tensor: torch.Tensor) -> torch.Tensor:
    """
    Clamp tensor to [0, 1] range.

    Args:
        tensor: Input tensor

    Returns:
        Tensor clamped to [0, 1]
    """
    return torch.clamp(tensor, 0.0, 1.0)


def get_image_size(tensor: torch.Tensor, format: str = "BHWC") -> Tuple[int, int]:
    """
    Get height and width from image tensor.

    Args:
        tensor: Image tensor
        format: "BHWC" or "BCHW"

    Returns:
        (height, width) tuple
    """
    if format == "BHWC":
        return tensor.shape[1], tensor.shape[2]
    elif format == "BCHW":
        return tensor.shape[2], tensor.shape[3]
    else:
        raise ValueError(f"Unknown format: {format}")


def resize_to_multiple(
    tensor: torch.Tensor,
    multiple: int = 8,
    mode: str = "bilinear",
    format: str = "BCHW",
) -> torch.Tensor:
    """
    Resize tensor so height and width are multiples of given value.

    This is often required for VAE encoding (multiple of 8) or
    other model requirements.

    Args:
        tensor: Input tensor
        multiple: Target multiple (default 8 for VAE)
        mode: Interpolation mode
        format: "BHWC" or "BCHW"

    Returns:
        Resized tensor (same format as input)
    """
    import torch.nn.functional as F

    if format == "BHWC":
        tensor = bhwc_to_bchw(tensor)
        was_bhwc = True
    else:
        was_bhwc = False

    _, _, h, w = tensor.shape
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple

    if new_h != h or new_w != w:
        tensor = F.interpolate(tensor, size=(new_h, new_w), mode=mode, align_corners=False)

    if was_bhwc:
        tensor = bchw_to_bhwc(tensor)

    return tensor
