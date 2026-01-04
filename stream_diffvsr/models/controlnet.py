"""
ControlNet wrapper for Stream-DiffVSR temporal guidance.

Stream-DiffVSR uses ControlNet to inject temporal information from the
warped previous HQ frame into the U-Net during denoising. This is the
"Auto-Regressive Temporal Guidance" (ARTG) mentioned in the paper.

The ControlNet takes the flow-warped previous HQ frame as conditioning
and outputs residuals that are added to the U-Net's down/mid blocks.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, Union, List


class TemporalControlNet(nn.Module):
    """
    ControlNet for temporal guidance in Stream-DiffVSR.

    Wraps diffusers ControlNetModel to provide temporal conditioning
    from warped previous HQ frames during the denoising process.
    
    The ControlNet is conditioned on the warped previous HQ frame and
    produces residuals that are injected into the U-Net decoder.
    """

    def __init__(
        self,
        controlnet: nn.Module,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize ControlNet wrapper.

        Args:
            controlnet: The underlying ControlNetModel from diffusers
            config: Optional configuration overrides
        """
        super().__init__()
        self.controlnet = controlnet
        self.config = config or {}

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        guess_mode: bool = False,
        return_dict: bool = False,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Forward pass to compute temporal guidance residuals.

        Args:
            sample: Noisy latent (B, C, H, W)
            timestep: Current denoising timestep
            encoder_hidden_states: Text/prompt embeddings
            controlnet_cond: Warped previous HQ frame (B, 3, H*4, W*4)
                            This is the temporal conditioning input
            conditioning_scale: Scale factor for ControlNet outputs
            guess_mode: Whether to use guess mode (no text conditioning)
            return_dict: Whether to return as dict

        Returns:
            Tuple of:
                - down_block_res_samples: List of residuals for U-Net down blocks
                - mid_block_res_sample: Residual for U-Net mid block
        """
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            guess_mode=guess_mode,
            return_dict=False,
        )

        return down_block_res_samples, mid_block_res_sample

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        subfolder: Optional[str] = "controlnet",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "TemporalControlNet":
        """
        Load ControlNet from pretrained weights.

        Supports loading from:
        1. Local path (e.g., ComfyUI/models/StreamDiffVSR/v1/controlnet/)
        2. HuggingFace model ID (e.g., "Jamichsu/Stream-DiffVSR")

        Args:
            pretrained_path: Local path or HuggingFace model ID
            subfolder: Subfolder containing ControlNet (for HF format)
            torch_dtype: Model dtype (default: float16)
            **kwargs: Additional arguments for from_pretrained

        Returns:
            TemporalControlNet instance
        """
        try:
            from diffusers import ControlNetModel
        except ImportError:
            raise ImportError(
                "diffusers is required for ControlNet. "
                "Install with: pip install diffusers>=0.25.0"
            )

        controlnet = ControlNetModel.from_pretrained(
            pretrained_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        return cls(controlnet=controlnet)

    @classmethod
    def from_local(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "TemporalControlNet":
        """
        Load ControlNet from local model files.

        Args:
            model_path: Path to ControlNet directory containing config.json
                       and model weights (e.g., diffusion_pytorch_model.safetensors)
            torch_dtype: Model dtype
            **kwargs: Additional arguments

        Returns:
            TemporalControlNet instance
        """
        try:
            from diffusers import ControlNetModel
        except ImportError:
            raise ImportError(
                "diffusers is required for ControlNet. "
                "Install with: pip install diffusers>=0.25.0"
            )

        controlnet = ControlNetModel.from_pretrained(
            model_path,
            subfolder=None,  # No subfolder for direct path
            torch_dtype=torch_dtype,
            **kwargs,
        )

        return cls(controlnet=controlnet)

    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        self.controlnet = self.controlnet.to(*args, **kwargs)
        return self
