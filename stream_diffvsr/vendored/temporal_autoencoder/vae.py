"""
VAE Components for Temporal Autoencoder.

Vendored from: https://github.com/jamichss/Stream-DiffVSR
License: Apache-2.0

This module contains the EncoderTiny and TemporalDecoderTiny classes
used by TemporalAutoencoderTiny.
"""

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from diffusers.utils import BaseOutput
from diffusers.models.activations import get_activation
from .blocks import TemporalAutoencoderTinyBlock


@dataclass
class DecoderOutput(BaseOutput):
    """
    Output of decoding method.

    Args:
        sample (torch.Tensor): Shape (batch_size, num_channels, height, width).
            The decoded output sample from the last layer of the model.
    """
    sample: torch.Tensor


class EncoderTiny(nn.Module):
    """
    The EncoderTiny layer is a simpler version of the Encoder layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        num_blocks (Tuple[int, ...]): Each value represents a Conv2d layer
            followed by that many TemporalAutoencoderTinyBlocks.
        block_out_channels (Tuple[int, ...]): Output channels for each block.
        act_fn (str): The activation function to use.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        act_fn: str,
    ):
        super().__init__()

        layers = []
        for i, num_block in enumerate(num_blocks):
            num_channels = block_out_channels[i]

            if i == 0:
                layers.append(nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1))
            else:
                layers.append(
                    nn.Conv2d(
                        num_channels,
                        num_channels,
                        kernel_size=3,
                        padding=1,
                        stride=2,
                        bias=False,
                    )
                )

            for _ in range(num_block):
                layers.append(TemporalAutoencoderTinyBlock(num_channels, num_channels, act_fn))

        layers.append(nn.Conv2d(block_out_channels[-1], out_channels, kernel_size=3, padding=1))

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Scales input from [-1, 1] to [0, 1] to match TAESD convention."""
        x = self.layers(x.add(1).div(2))
        return x


class TemporalDecoderTiny(nn.Module):
    """
    The DecoderTiny layer with temporal processing support.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        num_blocks (Tuple[int, ...]): Each value represents a Conv2d layer
            followed by that many TemporalAutoencoderTinyBlocks.
        block_out_channels (Tuple[int, ...]): Output channels for each block.
        upsampling_scaling_factor (int): The scaling factor for upsampling.
        act_fn (str): The activation function to use.
        upsample_fn (str): Upsampling mode ('nearest', 'bilinear', etc.)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: Tuple[int, ...],
        block_out_channels: Tuple[int, ...],
        upsampling_scaling_factor: int,
        act_fn: str,
        upsample_fn: str,
    ):
        super().__init__()

        layers = [
            nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, padding=1),
            get_activation(act_fn),
        ]

        for i, num_block in enumerate(num_blocks):
            is_final_block = i == (len(num_blocks) - 1)
            num_channels = block_out_channels[i]

            for _ in range(num_block):
                block = TemporalAutoencoderTinyBlock(num_channels, num_channels, act_fn)
                layers.append(block)

            if not is_final_block:
                layers.append(nn.Upsample(scale_factor=upsampling_scaling_factor, mode=upsample_fn))

            conv_out_channel = num_channels if not is_final_block else out_channels
            layers.append(
                nn.Conv2d(
                    num_channels,
                    conv_out_channel,
                    kernel_size=3,
                    padding=1,
                    bias=is_final_block,
                )
            )

        self.layers = nn.Sequential(*layers)
        self.gradient_checkpointing = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Clamps input and scales output from [0, 1] to [-1, 1]."""
        # Clamp input latents
        x = torch.tanh(x / 3) * 3
        
        x = self.layers(x)
        
        # Scale from [0, 1] to [-1, 1] to match diffusers convention
        return x.mul(2).sub(1)
