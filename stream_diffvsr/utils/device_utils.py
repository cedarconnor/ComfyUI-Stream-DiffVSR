"""
Device and dtype utilities for Stream-DiffVSR.
"""

import torch
from typing import Union


def get_device(device_str: str = "auto") -> torch.device:
    """
    Get torch device from string specification.

    Args:
        device_str: One of:
            - "auto": Select CUDA if available, else CPU
            - "cuda": Use default CUDA device
            - "cuda:N": Use specific CUDA device
            - "cpu": Use CPU

    Returns:
        torch.device: Target device
    """
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)


def get_dtype(dtype_str: str = "float16") -> torch.dtype:
    """
    Get torch dtype from string specification.

    Args:
        dtype_str: One of "float16", "bfloat16", "float32"

    Returns:
        torch.dtype: Target dtype
    """
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
    }

    dtype_str_lower = dtype_str.lower()
    if dtype_str_lower not in dtype_map:
        raise ValueError(
            f"Unknown dtype: {dtype_str}. "
            f"Supported: {list(dtype_map.keys())}"
        )

    return dtype_map[dtype_str_lower]


def get_optimal_dtype(device: torch.device) -> torch.dtype:
    """
    Determine optimal dtype based on device capabilities.

    Args:
        device: Target device

    Returns:
        Recommended dtype for inference
    """
    if device.type == "cpu":
        return torch.float32

    if device.type == "cuda":
        # Check compute capability for bfloat16 support
        capability = torch.cuda.get_device_capability(device)
        if capability >= (8, 0):  # Ampere and newer
            return torch.bfloat16
        else:
            return torch.float16

    # Default fallback
    return torch.float32


def get_available_vram_mb(device: Union[torch.device, int] = 0) -> int:
    """
    Get available VRAM in megabytes.

    Args:
        device: CUDA device or device index

    Returns:
        Available VRAM in MB, or 0 if not applicable
    """
    if not torch.cuda.is_available():
        return 0

    if isinstance(device, torch.device):
        if device.type != "cuda":
            return 0
        device_idx = device.index or 0
    else:
        device_idx = device

    try:
        free, total = torch.cuda.mem_get_info(device_idx)
        return free // (1024 * 1024)
    except Exception:
        return 0


def estimate_vram_mb(height: int, width: int, scale: int = 4) -> int:
    """
    Estimate VRAM usage in MB for given input size.

    This is a rough estimation based on model architecture.

    Args:
        height: Input height
        width: Input width
        scale: Upscale factor

    Returns:
        Estimated VRAM usage in MB
    """
    pixels = height * width
    output_pixels = pixels * (scale ** 2)

    # Base model memory (~4GB for all components)
    base_memory = 4000

    # Per-pixel memory for latents, features, activations
    # This is a rough estimate
    pixel_memory = pixels * 0.01 + output_pixels * 0.005

    # Buffer for activations during inference (1.5x multiplier)
    activation_memory = pixel_memory * 1.5

    return int(base_memory + pixel_memory + activation_memory)


def clear_cuda_cache():
    """Clear CUDA cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
