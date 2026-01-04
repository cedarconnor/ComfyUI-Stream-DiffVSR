"""
4-step DDIM scheduler for Stream-DiffVSR.

Stream-DiffVSR uses a distilled model optimized for 4 denoising steps.
This scheduler implements the DDIM sampling with appropriate timesteps.

NOTE: The exact timestep schedule should be verified against upstream.
"""

import torch
import numpy as np
from typing import Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class SchedulerOutput:
    """Output of scheduler step."""

    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


class DDIM4StepScheduler:
    """
    DDIM scheduler optimized for 4-step inference.

    Implements deterministic DDIM sampling with configurable timesteps.
    The model is distilled for 4 steps, but other step counts are supported.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        clip_sample: bool = False,
        set_alpha_to_one: bool = False,
        prediction_type: str = "epsilon",
    ):
        """
        Initialize scheduler.

        Args:
            num_train_timesteps: Number of training timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            beta_schedule: Schedule type ("linear" or "scaled_linear")
            clip_sample: Whether to clip predicted samples
            set_alpha_to_one: Whether to set final alpha to 1
            prediction_type: "epsilon" or "v_prediction"
        """
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type
        self.clip_sample = clip_sample

        # Compute betas
        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float64)
        elif beta_schedule == "scaled_linear":
            # Scaled linear schedule (used by most SD models)
            betas = np.linspace(
                beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float64
            ) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = torch.from_numpy(betas).float()
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        # Final alpha
        self.final_alpha_cumprod = (
            torch.tensor(1.0) if set_alpha_to_one else self.alphas_cumprod[0]
        )

        # Initial noise sigma
        self.init_noise_sigma = 1.0

        # Will be set by set_timesteps
        self.timesteps = None
        self.num_inference_steps = None

    def set_timesteps(
        self,
        num_inference_steps: int = 4,
        device: Union[str, torch.device] = "cpu",
    ):
        """
        Set timesteps for inference.

        Args:
            num_inference_steps: Number of denoising steps (default 4)
            device: Target device
        """
        self.num_inference_steps = num_inference_steps

        # Compute timesteps - evenly spaced
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (np.arange(0, num_inference_steps) * step_ratio).round()[::-1]
        timesteps = timesteps.copy().astype(np.int64)

        self.timesteps = torch.from_numpy(timesteps).to(device)

        # Move alphas to device
        self.alphas_cumprod = self.alphas_cumprod.to(device)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> SchedulerOutput:
        """
        Perform one DDIM step.

        Args:
            model_output: Predicted noise from U-Net
            timestep: Current timestep
            sample: Current noisy sample
            eta: DDIM eta parameter (0 = deterministic)
            generator: Optional random generator

        Returns:
            SchedulerOutput with denoised sample
        """
        # Get current and previous timestep indices
        timestep_idx = (self.timesteps == timestep).nonzero(as_tuple=True)[0]

        if timestep_idx.numel() == 0:
            # Timestep not found, use closest
            timestep_idx = torch.argmin(torch.abs(self.timesteps - timestep))

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]

        if timestep_idx + 1 < len(self.timesteps):
            prev_timestep = self.timesteps[timestep_idx + 1]
            alpha_prod_t_prev = self.alphas_cumprod[prev_timestep]
        else:
            alpha_prod_t_prev = self.final_alpha_cumprod

        beta_prod_t = 1 - alpha_prod_t

        # Compute predicted original sample
        if self.prediction_type == "epsilon":
            pred_original_sample = (
                sample - beta_prod_t**0.5 * model_output
            ) / alpha_prod_t**0.5
        elif self.prediction_type == "v_prediction":
            pred_original_sample = (
                alpha_prod_t**0.5 * sample - beta_prod_t**0.5 * model_output
            )
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")

        # Clip if requested
        if self.clip_sample:
            pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute coefficients
        pred_epsilon = model_output  # For epsilon prediction

        # Compute previous sample
        alpha_prod_t_prev_sqrt = alpha_prod_t_prev**0.5
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        # DDIM formula
        pred_sample_direction = beta_prod_t_prev**0.5 * pred_epsilon

        prev_sample = (
            alpha_prod_t_prev_sqrt * pred_original_sample + pred_sample_direction
        )

        # Add noise if eta > 0 (stochastic DDIM)
        if eta > 0:
            variance = (
                (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
            )
            std = eta * variance**0.5

            noise = torch.randn(
                sample.shape,
                generator=generator,
                device=sample.device,
                dtype=sample.dtype,
            )
            prev_sample = prev_sample + std * noise

        return SchedulerOutput(
            prev_sample=prev_sample,
            pred_original_sample=pred_original_sample,
        )

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples for given timesteps.

        Args:
            original_samples: Clean samples
            noise: Noise to add
            timesteps: Timesteps for noise level

        Returns:
            Noisy samples
        """
        alphas_cumprod = self.alphas_cumprod.to(
            device=original_samples.device, dtype=original_samples.dtype
        )

        sqrt_alpha_prod = alphas_cumprod[timesteps] ** 0.5
        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timesteps]) ** 0.5

        # Reshape for broadcasting
        while len(sqrt_alpha_prod.shape) < len(original_samples.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        noisy_samples = (
            sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        )

        return noisy_samples
