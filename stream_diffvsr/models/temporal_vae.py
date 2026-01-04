"""
Temporal-aware VAE for Stream-DiffVSR.

This module wraps the TemporalAutoencoderTiny from upstream, which extends
AutoEncoderTiny with Temporal Processor Modules (TPM) for temporal consistency.

The TPM fuses features from the warped previous HQ frame with current decode
features at multiple scales.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Union, Tuple


class TemporalProcessorModule(nn.Module):
    """
    Temporal Processor Module (TPM).

    Fuses features from the current decode path with features extracted
    from the warped previous HQ frame at each decoder scale.
    
    This is a placeholder - the actual TPM is part of TemporalAutoencoderTiny.
    """

    def __init__(self, channels: int, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.channels = channels
        self.config = config or {}
        # Actual implementation in TemporalAutoencoderTiny

    def forward(
        self,
        current_features: torch.Tensor,
        temporal_features: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse current and temporal features."""
        raise NotImplementedError("Use TemporalAutoencoderTiny directly")


class TemporalVAE(nn.Module):
    """
    Temporal-aware VAE for Stream-DiffVSR.

    Wraps TemporalAutoencoderTiny from upstream with a simplified interface.
    
    Key features:
    - encode(): Optionally returns layer features for TPM
    - decode(): Accepts temporal features for TPM fusion
    - reset_temporal_condition(): Clears TPM state between sequences
    
    The VAE uses scaling_factor=1.0 with latent_magnitude/latent_shift
    instead of the typical 0.18215 scaling.
    """

    def __init__(
        self,
        vae: nn.Module,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize Temporal VAE wrapper.

        Args:
            vae: The underlying TemporalAutoencoderTiny
            config: Optional configuration overrides
        """
        super().__init__()
        self.vae = vae
        self._config = config or {}

    @property
    def config(self):
        """Return the underlying VAE config."""
        return self.vae.config

    @property
    def scaling_factor(self) -> float:
        """VAE scaling factor (typically 1.0 for AutoEncoderTiny)."""
        return getattr(self.vae.config, 'scaling_factor', 1.0)

    def encode(
        self,
        x: torch.Tensor,
        return_features_only: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Encode image to latent space.

        Args:
            x: Input image (B, C, H, W) in [-1, 1] range
            return_features_only: If True, return only layer features
                                 for TPM (used for warped previous HQ)

        Returns:
            If return_features_only:
                List of layer features (for TPM)
            Else:
                Latent tensor (B, 4, H/8, W/8)
        """
        if return_features_only:
            # Extract layer features for temporal conditioning
            return self.vae.encode(x, return_features_only=True)
        else:
            # Standard encode
            output = self.vae.encode(x)
            if hasattr(output, 'latents'):
                return output.latents
            return output

    def decode(
        self,
        latents: torch.Tensor,
        temporal_features: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Decode latents to image with optional temporal fusion.

        Args:
            latents: Denoised latents (B, 4, H/8, W/8)
            temporal_features: Optional list of features from warped previous
                              HQ frame, used by TPM for temporal consistency.
                              Should be in reversed order (from encode output).

        Returns:
            Decoded image (B, 3, H, W) in [-1, 1] range
        """
        output = self.vae.decode(
            latents,
            temporal_features=temporal_features,
        )
        
        if hasattr(output, 'sample'):
            return output.sample
        return output[0] if isinstance(output, tuple) else output

    def reset_temporal_condition(self):
        """Reset TPM state between sequences or frames."""
        if hasattr(self.vae, 'reset_temporal_condition'):
            self.vae.reset_temporal_condition()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        subfolder: Optional[str] = "vae",
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "TemporalVAE":
        """
        Load Temporal VAE from pretrained weights.

        Supports loading from:
        1. Local path (e.g., ComfyUI/models/StreamDiffVSR/v1/vae/)
        2. HuggingFace model ID (e.g., "Jamichsu/Stream-DiffVSR")

        Args:
            pretrained_path: Local path or HuggingFace model ID
            subfolder: Subfolder containing VAE (for HF format)
            torch_dtype: Model dtype (default: float16)
            **kwargs: Additional arguments for from_pretrained

        Returns:
            TemporalVAE instance
        """
        # Try to import vendored TemporalAutoencoderTiny
        try:
            from ..vendored.temporal_autoencoder import TemporalAutoencoderTiny
            vae = TemporalAutoencoderTiny.from_pretrained(
                pretrained_path,
                subfolder=subfolder,
                torch_dtype=torch_dtype,
                **kwargs,
            )
        except (ImportError, Exception) as e:
            # Fallback: try diffusers AutoencoderTiny
            # Note: This won't have TPM support
            try:
                from diffusers import AutoencoderTiny
                print(
                    "[Stream-DiffVSR] Warning: TemporalAutoencoderTiny not available, "
                    f"using AutoencoderTiny (no TPM support). Reason: {e}"
                )
                vae = AutoencoderTiny.from_pretrained(
                    pretrained_path,
                    subfolder=subfolder,
                    torch_dtype=torch_dtype,
                    **kwargs,
                )
            except ImportError:
                raise ImportError(
                    "Could not load Temporal VAE. "
                    "Install diffusers>=0.25.0 or vendor TemporalAutoencoderTiny."
                )

        return cls(vae=vae)

    @classmethod
    def from_local(
        cls,
        model_path: str,
        torch_dtype: torch.dtype = torch.float16,
        **kwargs,
    ) -> "TemporalVAE":
        """
        Load Temporal VAE from local model files.

        Args:
            model_path: Path to VAE directory containing config and weights
            torch_dtype: Model dtype
            **kwargs: Additional arguments

        Returns:
            TemporalVAE instance
        """
        return cls.from_pretrained(
            model_path,
            subfolder=None,
            torch_dtype=torch_dtype,
            **kwargs,
        )

    def to(self, *args, **kwargs):
        """Move model to device/dtype."""
        self.vae = self.vae.to(*args, **kwargs)
        return self

    def eval(self):
        """Set model to eval mode."""
        self.vae.eval()
        return self

    def enable_slicing(self):
        """Enable sliced VAE decoding for memory efficiency."""
        if hasattr(self.vae, 'enable_slicing'):
            self.vae.enable_slicing()

    def enable_tiling(self):
        """Enable tiled VAE decoding for large images."""
        if hasattr(self.vae, 'enable_tiling'):
            self.vae.enable_tiling()
