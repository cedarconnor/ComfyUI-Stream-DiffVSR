"""
Temporal Autoencoder Tiny.

Vendored from: https://github.com/jamichss/Stream-DiffVSR
License: Apache-2.0

A tiny distilled VAE model for encoding images into latents and decoding 
latent representations into images, with temporal processing for video
frame consistency.
"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.utils import BaseOutput
from diffusers.utils.accelerate_utils import apply_forward_hook
from diffusers.models.modeling_utils import ModelMixin

from .vae import DecoderOutput, TemporalDecoderTiny, EncoderTiny
from .blocks import TemporalAutoencoderTinyBlock


@dataclass
class TemporalAutoencoderTinyOutput(BaseOutput):
    """
    Output of TemporalAutoencoderTiny encoding method.

    Args:
        latents (torch.Tensor): Encoded outputs of the Encoder.
    """
    latents: torch.Tensor


class TemporalAutoencoderTiny(ModelMixin, ConfigMixin):
    """
    A tiny distilled VAE model with temporal processing for video.
    
    This is a modified version of AutoencoderTiny that includes Temporal
    Processor Modules (TPM) for maintaining consistency across video frames.

    The encoder extracts features at multiple scales, which are then passed
    to the decoder where TPM blocks fuse them with the current decode path.

    Parameters:
        in_channels (int, defaults to 3): Number of input channels.
        out_channels (int, defaults to 3): Number of output channels.
        encoder_block_out_channels (Tuple[int], defaults to (64, 64, 64, 64)):
            Output channels for each encoder block.
        decoder_block_out_channels (Tuple[int], defaults to (64, 64, 64, 64)):
            Output channels for each decoder block.
        act_fn (str, defaults to "relu"): Activation function.
        latent_channels (int, defaults to 4): Number of latent channels.
        upsampling_scaling_factor (int, defaults to 2): Upsampling factor.
        num_encoder_blocks (Tuple[int], defaults to (1, 3, 3, 3)):
            Number of blocks at each encoder stage.
        num_decoder_blocks (Tuple[int], defaults to (3, 3, 3, 1)):
            Number of blocks at each decoder stage.
        latent_magnitude (float, defaults to 3.0): Latent scaling magnitude.
        latent_shift (float, defaults to 0.5): Latent shift value.
        scaling_factor (float, defaults to 1.0): VAE scaling factor.
        force_upcast (bool, defaults to False): Force float32 computation.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        encoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        decoder_block_out_channels: Tuple[int, ...] = (64, 64, 64, 64),
        act_fn: str = "relu",
        upsample_fn: str = "nearest",
        latent_channels: int = 4,
        upsampling_scaling_factor: int = 2,
        num_encoder_blocks: Tuple[int, ...] = (1, 3, 3, 3),
        num_decoder_blocks: Tuple[int, ...] = (3, 3, 3, 1),
        latent_magnitude: int = 3,
        latent_shift: float = 0.5,
        force_upcast: bool = False,
        scaling_factor: float = 1.0,
        shift_factor: float = 0.0,
    ):
        super().__init__()

        if len(encoder_block_out_channels) != len(num_encoder_blocks):
            raise ValueError("`encoder_block_out_channels` should have the same length as `num_encoder_blocks`.")
        if len(decoder_block_out_channels) != len(num_decoder_blocks):
            raise ValueError("`decoder_block_out_channels` should have the same length as `num_decoder_blocks`.")

        self.encoder = EncoderTiny(
            in_channels=in_channels,
            out_channels=latent_channels,
            num_blocks=num_encoder_blocks,
            block_out_channels=encoder_block_out_channels,
            act_fn=act_fn,
        )
        # Freeze encoder weights
        self.encoder.requires_grad_(False)

        self.decoder = TemporalDecoderTiny(
            in_channels=latent_channels,
            out_channels=out_channels,
            num_blocks=num_decoder_blocks,
            block_out_channels=decoder_block_out_channels,
            upsampling_scaling_factor=upsampling_scaling_factor,
            act_fn=act_fn,
            upsample_fn=upsample_fn,
        )
        # Freeze decoder weights except TPM-related
        self.decoder.requires_grad_(False)
        
        # Enable gradients for temporal processing parameters
        for name, param in self.decoder.named_parameters():
            if "alpha" in name or "temporal_processor" in name:
                param.requires_grad_(True)

        self.latent_magnitude = latent_magnitude
        self.latent_shift = latent_shift
        self.scaling_factor = scaling_factor

        self.use_slicing = False
        self.use_tiling = False

        # Tiling parameters
        self.spatial_scale_factor = 2 ** out_channels
        self.tile_overlap_factor = 0.125
        self.tile_sample_min_size = 512
        self.tile_latent_min_size = self.tile_sample_min_size // self.spatial_scale_factor

        self.register_to_config(block_out_channels=decoder_block_out_channels)
        self.register_to_config(force_upcast=False)

    def reset_temporal_condition(self):
        """Reset temporal memory in all blocks."""
        for module in self.encoder.layers:
            if isinstance(module, TemporalAutoencoderTinyBlock):
                module.reset_temporal()
        for module in self.decoder.layers:
            if isinstance(module, TemporalAutoencoderTinyBlock):
                module.reset_temporal()

    def _set_gradient_checkpointing(self, module, value: bool = False) -> None:
        if isinstance(module, (EncoderTiny, TemporalDecoderTiny)):
            module.gradient_checkpointing = value

    def scale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """raw latents -> [0, 1]"""
        return x.div(2 * self.latent_magnitude).add(self.latent_shift).clamp(0, 1)

    def unscale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """[0, 1] -> raw latents"""
        return x.sub(self.latent_shift).mul(2 * self.latent_magnitude)

    def enable_slicing(self) -> None:
        """Enable sliced VAE decoding for memory reduction."""
        self.use_slicing = True

    def disable_slicing(self) -> None:
        """Disable sliced VAE decoding."""
        self.use_slicing = False

    def enable_tiling(self, use_tiling: bool = True) -> None:
        """Enable tiled VAE decoding for larger images."""
        self.use_tiling = use_tiling

    def disable_tiling(self) -> None:
        """Disable tiled VAE decoding."""
        self.enable_tiling(False)

    @apply_forward_hook
    def encode(
        self, 
        x: torch.Tensor, 
        return_dict: bool = True, 
        return_layers_features: bool = True, 
        return_features_only: bool = False
    ) -> Union[TemporalAutoencoderTinyOutput, Tuple[torch.Tensor], List[torch.Tensor]]:
        """
        Encode images to latent space.
        
        Args:
            x: Input images (B, C, H, W) in [-1, 1] range.
            return_dict: Whether to return as dataclass.
            return_layers_features: Whether to return intermediate features.
            return_features_only: Only return features, not latents.
            
        Returns:
            Latent tensor or features depending on arguments.
        """
        layer_features = [] if return_layers_features else None

        if self.use_slicing and x.shape[0] > 1:
            output = [self.encoder(x_slice) for x_slice in x.split(1)]
            output = torch.cat(output)
        elif return_layers_features:
            current_features = x
            for module in self.encoder.layers:
                current_features = module(current_features)
                if isinstance(module, TemporalAutoencoderTinyBlock):
                    layer_features.append(current_features)

            if return_features_only:
                return layer_features

            output = self.encoder(x)
        else:
            output = self.encoder(x)

        if not return_dict:
            return (output,), layer_features

        return TemporalAutoencoderTinyOutput(latents=output)

    @apply_forward_hook
    def decode(
        self, 
        x: torch.Tensor, 
        temporal_features: Optional[List[torch.Tensor]] = None,
        generator: Optional[torch.Generator] = None, 
        return_dict: bool = True
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """
        Decode latents to images with optional temporal feature fusion.
        
        Args:
            x: Latent tensor (B, C, h, w).
            temporal_features: Features from previous frame encoding (reversed order).
            generator: Random generator (unused, for API compatibility).
            return_dict: Whether to return as dataclass.
            
        Returns:
            Decoded images (B, C, H, W) in [-1, 1] range.
        """
        if self.use_slicing and x.shape[0] > 1:
            output = [self.decoder(x_slice) for x_slice in x.split(1)]
            output = torch.cat(output)
        elif temporal_features is not None:
            # Set previous features for temporal processing
            block_idx = 0
            for module in self.decoder.layers:
                if isinstance(module, TemporalAutoencoderTinyBlock):
                    if block_idx < len(temporal_features):
                        module.prev_features = temporal_features[block_idx]
                    block_idx += 1
            output = self.decoder(x)
        else:
            output = self.decoder(x)

        if not return_dict:
            return (output,)

        return DecoderOutput(sample=output)

    def forward(
        self,
        sample: torch.Tensor,
        previous_sample: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[DecoderOutput, Tuple[torch.Tensor]]:
        """
        Full forward pass: encode then decode with temporal conditioning.
        
        Args:
            sample: Input latents to decode.
            previous_sample: Previous frame for temporal features.
            return_dict: Whether to return as dataclass.
            
        Returns:
            Decoded images.
        """
        layer_features = None

        if previous_sample is not None:
            _, layer_features = self.encode(previous_sample, return_dict=False)
            temporal_features = layer_features[::-1] if layer_features else None
        else:
            temporal_features = None

        dec = self.decode(sample, temporal_features=temporal_features, return_dict=False)[0]

        if not return_dict:
            return (dec,)

        return DecoderOutput(sample=dec)
