"""
Compatibility layer for handling version differences.

Provides defensive imports and version checks to ensure Stream-DiffVSR
works across different ComfyUI and dependency configurations.
"""

import sys
import warnings
from typing import Tuple

# Minimum supported versions
MIN_PYTHON = (3, 10)
MIN_TORCH = (2, 0, 0)
MIN_DIFFUSERS = (0, 25, 0)
MAX_DIFFUSERS = (0, 32, 0)  # Exclusive upper bound


def parse_version(version_str: str) -> Tuple[int, ...]:
    """Parse version string to tuple of integers."""
    # Handle versions like "2.1.0+cu118" or "0.25.0.dev0"
    clean = version_str.split("+")[0].split(".dev")[0]
    parts = []
    for part in clean.split(".")[:3]:
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def check_dependencies() -> bool:
    """
    Verify dependencies and warn about potential issues.

    Returns:
        True if all checks pass, False if there are warnings
    """
    issues = []

    # Check Python version
    if sys.version_info < MIN_PYTHON:
        issues.append(
            f"Python {sys.version_info.major}.{sys.version_info.minor} < "
            f"{MIN_PYTHON[0]}.{MIN_PYTHON[1]} (unsupported)"
        )

    # Check PyTorch
    try:
        import torch

        torch_version = parse_version(torch.__version__)
        if torch_version < MIN_TORCH:
            issues.append(f"torch {torch.__version__} < 2.0.0 (untested, may not work)")
        else:
            # Check CUDA availability
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda or "unknown"
                device_name = torch.cuda.get_device_name(0)
                print(f"[Stream-DiffVSR] CUDA {cuda_version} available: {device_name}")
            else:
                print("[Stream-DiffVSR] CUDA not available, using CPU (slow)")
    except ImportError:
        raise ImportError(
            "PyTorch is required for Stream-DiffVSR. "
            "Install with: pip install torch>=2.0.0"
        )

    # Check diffusers
    try:
        import diffusers

        diffusers_version = parse_version(diffusers.__version__)
        if diffusers_version < MIN_DIFFUSERS:
            issues.append(
                f"diffusers {diffusers.__version__} < 0.25.0 (may not work)"
            )
        elif diffusers_version >= MAX_DIFFUSERS:
            issues.append(
                f"diffusers {diffusers.__version__} >= 0.32.0 (untested)"
            )
    except ImportError:
        raise ImportError(
            "diffusers is required for Stream-DiffVSR. "
            "Install with: pip install diffusers>=0.25.0,<0.32.0"
        )

    # Check safetensors
    try:
        import safetensors
    except ImportError:
        raise ImportError(
            "safetensors is required for Stream-DiffVSR. "
            "Install with: pip install safetensors>=0.4.0"
        )

    # Check einops
    try:
        import einops
    except ImportError:
        raise ImportError(
            "einops is required for Stream-DiffVSR. "
            "Install with: pip install einops>=0.6.0"
        )

    # Check for optional xformers (recommended but not required)
    try:
        import xformers

        print(f"[Stream-DiffVSR] xformers {xformers.__version__} available")
    except ImportError:
        pass  # Optional, don't warn

    # Report issues
    if issues:
        warnings.warn(
            "Stream-DiffVSR dependency warnings:\n  - " + "\n  - ".join(issues),
            UserWarning,
            stacklevel=2,
        )

    return len(issues) == 0


def get_optimal_dtype():
    """
    Determine optimal dtype based on available hardware.

    Returns:
        torch.dtype: Recommended dtype for inference
    """
    import torch

    if not torch.cuda.is_available():
        return torch.float32

    # Check compute capability for bfloat16 support
    capability = torch.cuda.get_device_capability()
    if capability >= (8, 0):  # Ampere and newer
        return torch.bfloat16
    else:
        return torch.float16


def get_device(device_str: str = "auto"):
    """
    Get torch device from string specification.

    Args:
        device_str: "auto", "cuda", "cpu", or specific like "cuda:0"

    Returns:
        torch.device: Target device
    """
    import torch

    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_str)
