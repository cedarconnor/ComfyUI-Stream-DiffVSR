"""
Scheduler utilities for Stream-DiffVSR.

Stream-DiffVSR uses a distilled model optimized for 4 denoising steps.
This module uses diffusers DDIMScheduler for compatibility.
"""

import torch
from typing import Optional, Union


def create_scheduler(
    model_path_or_id: Optional[str] = None,
    subfolder: str = "scheduler",
    num_train_timesteps: int = 1000,
    beta_start: float = 0.00085,
    beta_end: float = 0.012,
    beta_schedule: str = "scaled_linear",
    prediction_type: str = "epsilon",
):
    """
    Create DDIM scheduler for Stream-DiffVSR.

    If model_path_or_id is provided, loads scheduler config from there.
    Otherwise creates with default parameters.

    Args:
        model_path_or_id: Path to model or HuggingFace model ID
        subfolder: Subfolder containing scheduler config
        num_train_timesteps: Number of training timesteps
        beta_start: Starting beta value
        beta_end: Ending beta value
        beta_schedule: Schedule type
        prediction_type: "epsilon" or "v_prediction"

    Returns:
        DDIMScheduler instance
    """
    try:
        from diffusers import DDIMScheduler
    except ImportError:
        raise ImportError(
            "diffusers is required for scheduler. "
            "Install with: pip install diffusers>=0.25.0"
        )

    if model_path_or_id is not None:
        try:
            scheduler = DDIMScheduler.from_pretrained(
                model_path_or_id, subfolder=subfolder
            )
            return scheduler
        except Exception:
            # Fall back to default config
            pass

    # Create with default config
    scheduler = DDIMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        beta_schedule=beta_schedule,
        clip_sample=False,
        set_alpha_to_one=False,
        prediction_type=prediction_type,
    )

    return scheduler


# Backwards compatibility alias
DDIM4StepScheduler = None


def get_scheduler(*args, **kwargs):
    """Alias for create_scheduler for backwards compatibility."""
    return create_scheduler(*args, **kwargs)
