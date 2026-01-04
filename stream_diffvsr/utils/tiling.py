"""
Temporal-aware tiling for memory-efficient processing.

CRITICAL: Naive spatial tiling breaks temporal consistency because
optical flow must be computed on full frames. This module implements
a tiling strategy that preserves temporal coherence.

Strategy:
1. Compute optical flow on FULL LQ frames (no tiling)
2. Warp FULL previous HQ frame (no tiling)
3. Tile only the diffusion steps
4. Blend tiles with overlap feathering
5. Store FULL HQ result in state
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Generator


def get_tile_coords(
    height: int,
    width: int,
    tile_size: int = 512,
    overlap: int = 64,
) -> List[Tuple[int, int, int, int]]:
    """
    Generate tile coordinates with overlap.

    Args:
        height: Image height
        width: Image width
        tile_size: Size of each tile
        overlap: Overlap between adjacent tiles

    Returns:
        List of (y1, y2, x1, x2) tuples defining each tile
    """
    tiles = []
    stride = tile_size - overlap

    # Generate y coordinates
    y_coords = []
    y = 0
    while y < height:
        y_end = min(y + tile_size, height)
        y_coords.append((y, y_end))
        if y_end == height:
            break
        y += stride

    # Generate x coordinates
    x_coords = []
    x = 0
    while x < width:
        x_end = min(x + tile_size, width)
        x_coords.append((x, x_end))
        if x_end == width:
            break
        x += stride

    # Combine into tiles
    for y1, y2 in y_coords:
        for x1, x2 in x_coords:
            tiles.append((y1, y2, x1, x2))

    return tiles


def create_feather_mask(
    tile_height: int,
    tile_width: int,
    overlap: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Create a feathering mask for blending tiles.

    The mask has linear falloff in the overlap regions to ensure
    smooth blending between adjacent tiles.

    Args:
        tile_height: Height of tile
        tile_width: Width of tile
        overlap: Overlap size
        device: Target device
        dtype: Data type

    Returns:
        Feather mask (1, 1, H, W)
    """
    mask = torch.ones(1, 1, tile_height, tile_width, device=device, dtype=dtype)

    if overlap <= 0:
        return mask

    # Create linear ramps for edges
    ramp = torch.linspace(0, 1, overlap, device=device, dtype=dtype)

    # Apply to all edges
    # Top edge
    mask[:, :, :overlap, :] *= ramp.view(1, 1, -1, 1)
    # Bottom edge
    mask[:, :, -overlap:, :] *= ramp.flip(0).view(1, 1, -1, 1)
    # Left edge
    mask[:, :, :, :overlap] *= ramp.view(1, 1, 1, -1)
    # Right edge
    mask[:, :, :, -overlap:] *= ramp.flip(0).view(1, 1, 1, -1)

    return mask


def tile_generator(
    tensor: torch.Tensor,
    tile_size: int = 512,
    overlap: int = 64,
) -> Generator[Tuple[torch.Tensor, Tuple[int, int, int, int]], None, None]:
    """
    Generate tiles from a tensor.

    Args:
        tensor: Input tensor (B, C, H, W)
        tile_size: Size of each tile
        overlap: Overlap between tiles

    Yields:
        (tile, coords) where tile is the extracted tile and
        coords is (y1, y2, x1, x2)
    """
    _, _, H, W = tensor.shape
    coords = get_tile_coords(H, W, tile_size, overlap)

    for y1, y2, x1, x2 in coords:
        tile = tensor[:, :, y1:y2, x1:x2]
        yield tile, (y1, y2, x1, x2)


def blend_tiles(
    tiles: List[Tuple[torch.Tensor, Tuple[int, int, int, int]]],
    output_shape: Tuple[int, int, int, int],
    overlap: int = 64,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    """
    Blend tiles back into a single image with feathering.

    Args:
        tiles: List of (tile, coords) tuples
        output_shape: (B, C, H, W) shape of output
        overlap: Overlap used during tiling
        device: Target device
        dtype: Data type

    Returns:
        Blended output tensor
    """
    B, C, H, W = output_shape

    if device is None:
        device = tiles[0][0].device
    if dtype is None:
        dtype = tiles[0][0].dtype

    output = torch.zeros(B, C, H, W, device=device, dtype=dtype)
    weights = torch.zeros(1, 1, H, W, device=device, dtype=dtype)

    for tile, (y1, y2, x1, x2) in tiles:
        tile_h, tile_w = y2 - y1, x2 - x1
        mask = create_feather_mask(tile_h, tile_w, overlap, device, dtype)

        output[:, :, y1:y2, x1:x2] += tile * mask
        weights[:, :, y1:y2, x1:x2] += mask

    # Normalize by weights (avoid division by zero)
    output = output / weights.clamp(min=1e-8)

    return output


class TemporalAwareTiler:
    """
    Tiling processor that preserves temporal consistency.

    Key insight: Flow and warping happen BEFORE tiling.
    Only the diffusion/decoding steps are tiled.
    """

    def __init__(
        self,
        tile_size: int = 512,
        overlap: int = 64,
    ):
        """
        Initialize tiler.

        Args:
            tile_size: Size of each tile (default 512)
            overlap: Overlap between tiles (default 64)
        """
        self.tile_size = tile_size
        self.overlap = overlap

    def get_tiles(
        self,
        height: int,
        width: int,
    ) -> List[Tuple[int, int, int, int]]:
        """Get tile coordinates for given dimensions."""
        return get_tile_coords(height, width, self.tile_size, self.overlap)

    def extract_tile(
        self,
        tensor: torch.Tensor,
        coords: Tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Extract a tile from tensor."""
        y1, y2, x1, x2 = coords
        return tensor[:, :, y1:y2, x1:x2]

    def extract_lq_tile(
        self,
        lq_tensor: torch.Tensor,
        hq_coords: Tuple[int, int, int, int],
        scale: int = 4,
    ) -> torch.Tensor:
        """
        Extract corresponding LQ tile for HQ coordinates.

        Args:
            lq_tensor: Low-quality tensor (B, C, H, W)
            hq_coords: Coordinates in HQ space (y1, y2, x1, x2)
            scale: Upscale factor

        Returns:
            LQ tile
        """
        y1, y2, x1, x2 = hq_coords
        ly1, ly2 = y1 // scale, y2 // scale
        lx1, lx2 = x1 // scale, x2 // scale
        return lq_tensor[:, :, ly1:ly2, lx1:lx2]
