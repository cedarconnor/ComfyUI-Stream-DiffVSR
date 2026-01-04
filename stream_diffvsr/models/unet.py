"""
U-Net wrapper for Stream-DiffVSR.

Wraps diffusers UNet2DConditionModel with ControlNet residual injection support.
The U-Net is based on Stable Diffusion x4 Upscaler, distilled for 4-step inference.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union


class StreamDiffVSRUNet(nn.Module):
    """
    U-Net for Stream-DiffVSR denoising.

    This is a wrapper around diffusers UNet2DConditionModel that handles
    ControlNet residual injection for temporal guidance.

    The model accepts:
    - Noisy latents concatenated with LQ image latents
    - Text/prompt embeddings (can be empty)
    - ControlNet residuals (down_block_additional_residuals, mid_block_additional_residual)
    """

    def __init__(
        self,
        unet: nn.Module,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize U-Net wrapper.

        Args:
            unet: The underlying UNet2DConditionModel from diffusers
            config: Optional configuration overrides
        """
        super().__init__()
        self.unet = unet
        self._config = config or {}

    @property
    def config(self):
        """Return the underlying U-Net config."""
        return self.unet.config

    def forward(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, int],
        encoder_hidden_states: torch.Tensor,
        down_block_additional_residuals: Optional[List[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass with optional ControlNet residual injection.

        Args:
            sample: Noisy latent, possibly concatenated with LQ (B, C, H, W)
            timestep: Current denoising timestep
            encoder_hidden_states: Text/prompt embeddings
            down_block_additional_residuals: ControlNet down block residuals
                                            (for temporal guidance)
            mid_block_additional_residual: ControlNet mid block residual
            return_dict: Whether to return as dict
            **kwargs: Additional arguments passed to U-Net

        Returns:
            Predicted noise (B, C, H, W)
        """
        output = self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=down_block_additional_residuals,
            mid_block_additional_residual=mid_block_additional_residual,
            return_dict=return_dict,
            **kwargs,
        )

        if return_dict:
            return output
        else:
            # Return just the sample when not using return_dict
            return output[0] if isinstance(output, tuple) else output

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        subfolder: Optional[str] = "unet",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "StreamDiffVSRUNet":
        """
        Load U-Net from pretrained weights.

        Supports loading from:
        1. Local path (e.g., ComfyUI/models/StreamDiffVSR/v1/unet/)
        2. HuggingFace model ID (e.g., "Jamichsu/Stream-DiffVSR")

        Args:
            pretrained_path: Local path or HuggingFace model ID
            subfolder: Subfolder containing U-Net (for HF format)
            torch_dtype: Model dtype (default: float16)
            **kwargs: Additional arguments for from_pretrained

        Returns:
            StreamDiffVSRUNet instance
        """
        try:
            from diffusers import UNet2DConditionModel
        except ImportError:
            raise ImportError(
                "diffusers is required for U-Net. "
                "Install with: pip install diffusers>=0.25.0"
            )

        unet = UNet2DConditionModel.from_pretrained(
            pretrained_path,
            subfolder=subfolder,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        return cls(unet=unet)

    @classmethod
    def from_local(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "StreamDiffVSRUNet":
        """
        Load U-Net from local model files.

        Args:
            model_path: Path to U-Net directory containing config.json
                       and model weights
            torch_dtype: Model dtype
            **kwargs: Additional arguments

        Returns:
            StreamDiffVSRUNet instance
        """
        try:
            from diffusers import UNet2DConditionModel
        except ImportError:
            raise ImportError(
                "diffusers is required for U-Net. "
                "Install with: pip install diffusers>=0.25.0"
            )

        unet = UNet2DConditionModel.from_pretrained(
            model_path,
            subfolder=None,
            torch_dtype=torch_dtype,
            **kwargs,
        )

        return cls(unet=unet)

    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        self.unet = self.unet.to(*args, **kwargs)
        return self

    def enable_xformers_memory_efficient_attention(self):
        """Enable xformers for memory-efficient attention."""
        if hasattr(self.unet, 'enable_xformers_memory_efficient_attention'):
            self.unet.enable_xformers_memory_efficient_attention()

    def eval(self):
        """Set model to eval mode."""
        self.unet.eval()
        return self

    def train(self, mode: bool = True):
        """Set model training mode."""
        self.unet.train(mode)
        return self
