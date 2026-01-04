"""
Temporal Autoencoder Blocks.

Vendored from: https://github.com/jamichss/Stream-DiffVSR
License: Apache-2.0

This module contains the TemporalAutoencoderTinyBlock which adds temporal
processing to the standard AutoencoderTinyBlock, enabling temporal consistency
across video frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from diffusers.models.activations import get_activation


class TemporalAutoencoderTinyBlock(nn.Module):
    """
    Tiny Autoencoder block with temporal processing support.
    
    Used in TemporalAutoencoderTiny. It is a mini residual module consisting 
    of plain conv + ReLU blocks with an additional temporal processor for
    fusing features from previous frames.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        act_fn (str): The activation function to use.

    Returns:
        torch.Tensor: A tensor with the same shape as the input tensor, but 
        with the number of channels equal to `out_channels`.
    """

    def __init__(self, in_channels: int, out_channels: int, act_fn: str):
        super().__init__()
        act_fn_module = get_activation(act_fn)
        
        # Main convolution path
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            act_fn_module,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            act_fn_module,
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )
        
        # Skip connection
        self.skip = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.fuse = nn.ReLU()

        # Temporal processing layers
        self.prev_features: Optional[torch.Tensor] = None
        
        # Learnable blending parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))
        
        # 1D convolution for temporal processing
        self.temporal_processor = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            get_activation(act_fn),
            nn.Conv1d(out_channels, out_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional temporal feature fusion."""
        current_features = self.conv(x)

        # Apply temporal processing if previous features are available
        if self.prev_features is not None:
            B, C, H, W = current_features.shape
            
            # Fixed pooling kernel size
            pool_kernel = (4, 4)

            # Downsample for efficient temporal processing
            avg_pool = nn.AvgPool2d(kernel_size=pool_kernel, stride=pool_kernel)
            current_pooled = avg_pool(current_features)
            prev_pooled = avg_pool(self.prev_features)

            # Concatenate temporal features
            temporal_input = torch.cat([
                current_pooled.view(B, C, -1),
                prev_pooled.view(B, C, -1)
            ], dim=2)

            # Process through 1D conv
            temporal_out = self.temporal_processor(temporal_input)

            # Blend features
            pool_h, pool_w = current_pooled.shape[2], current_pooled.shape[3]
            temporal_out_fuse = (
                self.alpha * temporal_out[:, :, :pool_h * pool_w].view(B, C, pool_h, pool_w) +
                (1 - self.alpha) * temporal_out[:, :, -pool_h * pool_w:].view(B, C, pool_h, pool_w)
            )

            # Upsample back to original resolution
            temporal_out_fuse = F.interpolate(
                temporal_out_fuse,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )

            # Add temporal contribution (small weight to avoid artifacts)
            current_features = current_features + 0.1 * temporal_out_fuse

        return self.fuse(current_features + self.skip(x))

    def reset_temporal(self):
        """Reset temporal memory."""
        self.prev_features = None
