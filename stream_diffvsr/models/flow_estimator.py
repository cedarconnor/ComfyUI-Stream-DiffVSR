"""
Optical flow estimation for Stream-DiffVSR.

Uses RAFT-Large from torchvision for high-quality optical flow estimation
between consecutive frames. The flow is used to warp the previous HQ frame
to align with the current frame for temporal guidance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FlowEstimator(nn.Module):
    """
    RAFT-Large optical flow estimator with tiled processing support.

    Estimates optical flow between consecutive frames for temporal alignment.
    The flow is computed on bicubic-upscaled 4x images (not LQ resolution).
    
    For high-resolution inputs, uses tiled processing to avoid OOM errors.
    """

    def __init__(
        self,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        tile_size: int = 512,
        tile_overlap: int = 128,
    ):
        """
        Initialize RAFT-Large flow estimator.

        Args:
            device: Target device (default: cuda if available)
            dtype: Model dtype (RAFT works best in float32)
            tile_size: Default tile size for tiled processing
            tile_overlap: Default overlap between tiles (larger = better quality)
        """
        super().__init__()
        
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.dtype = dtype
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        # Load RAFT-Large from torchvision
        try:
            from torchvision.models.optical_flow import raft_large, Raft_Large_Weights
            self.model = raft_large(weights=Raft_Large_Weights.DEFAULT)
            self.model = self.model.to(self.device).eval()
            self.model.requires_grad_(False)
        except ImportError:
            raise ImportError(
                "torchvision with optical flow support is required. "
                "Install with: pip install torchvision>=0.15.0"
            )

    def _to_255_range(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert tensor to [0, 255] range for RAFT."""
        if tensor.min() < 0:
            # [-1, 1] range
            return ((tensor + 1) / 2 * 255).clamp(0, 255)
        elif tensor.max() <= 1:
            # [0, 1] range
            return (tensor * 255).clamp(0, 255)
        else:
            # Already in [0, 255]
            return tensor

    def _compute_flow_single(
        self,
        target_255: torch.Tensor,
        source_255: torch.Tensor,
    ) -> torch.Tensor:
        """Compute flow for a single pair (internal, expects 255 range)."""
        flows = self.model(target_255, source_255)
        return flows[-1]  # Best prediction (B, 2, H, W)

    def _get_tile_coords(
        self,
        height: int,
        width: int,
        tile_size: int,
        overlap: int,
    ) -> list:
        """Generate tile coordinates with overlap."""
        tiles = []
        stride = tile_size - overlap

        # Generate y coordinates
        y = 0
        while y < height:
            y_end = min(y + tile_size, height)
            # Generate x coordinates
            x = 0
            while x < width:
                x_end = min(x + tile_size, width)
                tiles.append((y, y_end, x, x_end))
                if x_end == width:
                    break
                x += stride
            if y_end == height:
                break
            y += stride

        return tiles

    def _create_feather_mask(
        self,
        tile_h: int,
        tile_w: int,
        overlap: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create feathering mask for blending tiles."""
        mask = torch.ones(1, 1, tile_h, tile_w, device=device, dtype=dtype)

        if overlap <= 0:
            return mask

        # Create linear ramps for edges
        ramp = torch.linspace(0, 1, overlap, device=device, dtype=dtype)

        # Apply to all edges (be careful with small tiles)
        if tile_h > overlap:
            mask[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
            mask[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
        if tile_w > overlap:
            mask[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
            mask[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)

        return mask

    @torch.inference_mode()
    def forward_tiled(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        tile_size: int = None,
        overlap: int = None,
    ) -> torch.Tensor:
        """
        Compute optical flow using tiled processing.

        This method is used for high-resolution inputs that would OOM
        with full-frame processing.

        Args:
            target: Target frame (B, C, H, W) - current frame
            source: Source frame (B, C, H, W) - previous frame
            tile_size: Tile size (default: self.tile_size)
            overlap: Overlap between tiles (default: self.tile_overlap)

        Returns:
            Flow field (B, H, W, 2) in pixel coordinates
        """
        tile_size = tile_size or self.tile_size
        overlap = overlap or self.tile_overlap

        target = target.to(self.device, self.dtype)
        source = source.to(self.device, self.dtype)

        B, C, H, W = target.shape

        # Convert to 255 range
        target_255 = self._to_255_range(target)
        source_255 = self._to_255_range(source)

        # Get tile coordinates
        tiles = self._get_tile_coords(H, W, tile_size, overlap)
        print(f'[Stream-DiffVSR] Computing flow in {len(tiles)} tiles ({tile_size}px, {overlap}px overlap)')

        # Accumulate flow and weights
        flow_accum = torch.zeros(B, 2, H, W, device=self.device, dtype=self.dtype)
        weight_accum = torch.zeros(1, 1, H, W, device=self.device, dtype=self.dtype)

        for i, (y1, y2, x1, x2) in enumerate(tiles):
            # Extract tiles
            target_tile = target_255[:, :, y1:y2, x1:x2]
            source_tile = source_255[:, :, y1:y2, x1:x2]

            # Compute flow for this tile
            tile_flow = self._compute_flow_single(target_tile, source_tile)

            # Create feather mask
            tile_h, tile_w = y2 - y1, x2 - x1
            mask = self._create_feather_mask(tile_h, tile_w, overlap, self.device, self.dtype)

            # Accumulate with feathering
            flow_accum[:, :, y1:y2, x1:x2] += tile_flow * mask
            weight_accum[:, :, y1:y2, x1:x2] += mask

        # Normalize by weights
        flow = flow_accum / weight_accum.clamp(min=1e-8)

        # Convert to (B, H, W, 2) format
        flow = flow.permute(0, 2, 3, 1)

        return flow

    @torch.inference_mode()
    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        rescale_factor: int = 1,
        enable_tiling: bool = True,
        tile_size: int = None,
        tile_overlap: int = None,
    ) -> torch.Tensor:
        """
        Estimate optical flow from source to target.

        Automatically falls back to tiled processing if full-frame OOMs.

        Args:
            target: Target frame (B, C, H, W) - current frame
            source: Source frame (B, C, H, W) - previous frame
            rescale_factor: Optional downscale factor for flow computation
            enable_tiling: Whether to allow tiled fallback on OOM
            tile_size: Tile size for tiled processing
            tile_overlap: Overlap for tiled processing

        Returns:
            Flow field (B, H, W, 2) in pixel coordinates
            
        Note:
            Flow convention: flow[y, x] = (dx, dy) where
            source[y + dy, x + dx] ≈ target[y, x]
        """
        # Ensure inputs are on correct device
        target = target.to(self.device, self.dtype)
        source = source.to(self.device, self.dtype)

        # Convert to 255 range
        target_255 = self._to_255_range(target)
        source_255 = self._to_255_range(source)

        # Try full-frame first, fall back to tiled on OOM
        try:
            flow = self._compute_flow_single(target_255, source_255)
        except torch.cuda.OutOfMemoryError:
            if not enable_tiling:
                raise
            print('[Stream-DiffVSR] OOM on full-frame flow, switching to tiled processing...')
            torch.cuda.empty_cache()
            return self.forward_tiled(
                target, source,
                tile_size=tile_size or self.tile_size,
                overlap=tile_overlap or self.tile_overlap,
            )

        # Optionally rescale flow
        if rescale_factor != 1:
            flow = F.interpolate(
                flow / rescale_factor,  # Use float division, not floor division
                scale_factor=1 / rescale_factor,
                mode='bilinear',
                align_corners=False,
            )

        # Convert to (B, H, W, 2) format
        flow = flow.permute(0, 2, 3, 1)

        return flow

    def to(self, device=None, dtype=None, *args, **kwargs):
        """Move model to device."""
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        if dtype is not None:
            self.dtype = dtype
        return self


class ZeroFlowEstimator(nn.Module):
    """
    Dummy flow estimator that returns zero flow.

    Used when optical flow estimation is disabled or as fallback.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        target: torch.Tensor,
        source: torch.Tensor,
        rescale_factor: int = 1,
    ) -> torch.Tensor:
        """Return zero flow field."""
        B, C, H, W = target.shape
        return torch.zeros(B, H, W, 2, device=target.device, dtype=target.dtype)

    def to(self, *args, **kwargs):
        """No-op for compatibility."""
        return self


def compute_flows(
    flow_estimator: FlowEstimator,
    images: list,
    rescale_factor: int = 1,
) -> list:
    """
    Compute forward optical flows for a sequence of images.

    Args:
        flow_estimator: FlowEstimator instance
        images: List of image tensors (B, C, H, W)
        rescale_factor: Optional downscale factor

    Returns:
        List of flow tensors, one for each consecutive pair
        flows[i] warps images[i] to images[i+1]
    """
    print('[Stream-DiffVSR] Computing optical flows...')
    
    forward_flows = []
    for i in range(1, len(images)):
        prev_image = images[i - 1]
        cur_image = images[i]
        flow = flow_estimator(cur_image, prev_image, rescale_factor=rescale_factor)
        forward_flows.append(flow)
    
    return forward_flows


def get_flow_estimator(
    device: torch.device = None,
    dtype: torch.dtype = torch.float32,
    use_raft: bool = True,
) -> FlowEstimator:
    """
    Get optical flow estimator.

    First checks if RAFT is available from torchvision.
    Falls back to ZeroFlowEstimator if not available.

    Args:
        device: Target device
        dtype: Model dtype (RAFT works best in float32)
        use_raft: Whether to use RAFT (if False, returns ZeroFlowEstimator)

    Returns:
        FlowEstimator or ZeroFlowEstimator instance
    """
    if not use_raft:
        print("[Stream-DiffVSR] Flow estimation disabled")
        return ZeroFlowEstimator()

    try:
        estimator = FlowEstimator(device=device, dtype=dtype)
        print("[Stream-DiffVSR] Using RAFT-Large for optical flow")
        return estimator
    except ImportError as e:
        print(f"[Stream-DiffVSR] RAFT not available: {e}")
        print("[Stream-DiffVSR] Falling back to zero flow (no temporal guidance)")
        return ZeroFlowEstimator()
