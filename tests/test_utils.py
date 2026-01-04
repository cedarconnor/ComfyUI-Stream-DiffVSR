"""
Tests for utility functions.
"""

import pytest
import torch

from stream_diffvsr.utils.image_utils import (
    bhwc_to_bchw,
    bchw_to_bhwc,
    normalize_to_neg1_1,
    denormalize_from_neg1_1,
    get_image_size,
)
from stream_diffvsr.utils.flow_utils import (
    warp_image,
    upscale_flow,
    create_meshgrid,
)
from stream_diffvsr.utils.device_utils import (
    get_device,
    get_dtype,
)


class TestImageUtils:
    """Test image utility functions."""

    def test_bhwc_to_bchw(self):
        """Test BHWC to BCHW conversion."""
        bhwc = torch.randn(2, 64, 64, 3)
        bchw = bhwc_to_bchw(bhwc)

        assert bchw.shape == (2, 3, 64, 64)
        assert torch.allclose(bchw[0, 0], bhwc[0, :, :, 0])

    def test_bchw_to_bhwc(self):
        """Test BCHW to BHWC conversion."""
        bchw = torch.randn(2, 3, 64, 64)
        bhwc = bchw_to_bhwc(bchw)

        assert bhwc.shape == (2, 64, 64, 3)
        assert torch.allclose(bhwc[0, :, :, 0], bchw[0, 0])

    def test_roundtrip_conversion(self):
        """Test BHWC -> BCHW -> BHWC roundtrip."""
        original = torch.randn(1, 128, 128, 3)
        roundtrip = bchw_to_bhwc(bhwc_to_bchw(original))

        assert torch.allclose(original, roundtrip)

    def test_normalization(self):
        """Test [0,1] to [-1,1] normalization."""
        tensor = torch.tensor([0.0, 0.5, 1.0])

        normalized = normalize_to_neg1_1(tensor)
        assert torch.allclose(normalized, torch.tensor([-1.0, 0.0, 1.0]))

        denormalized = denormalize_from_neg1_1(normalized)
        assert torch.allclose(denormalized, tensor)

    def test_get_image_size(self):
        """Test getting image dimensions."""
        bhwc = torch.randn(1, 480, 640, 3)
        bchw = torch.randn(1, 3, 480, 640)

        assert get_image_size(bhwc, "BHWC") == (480, 640)
        assert get_image_size(bchw, "BCHW") == (480, 640)


class TestFlowUtils:
    """Test optical flow utilities."""

    def test_create_meshgrid(self):
        """Test meshgrid creation."""
        grid_x, grid_y = create_meshgrid(64, 128, torch.device("cpu"))

        assert grid_x.shape == (64, 128)
        assert grid_y.shape == (64, 128)
        # First row of grid_y should be all zeros
        assert (grid_y[0] == 0).all()
        # First column of grid_x should be all zeros
        assert (grid_x[:, 0] == 0).all()

    def test_warp_with_zero_flow(self):
        """Test warping with zero flow (should return original)."""
        image = torch.randn(1, 3, 64, 64)
        flow = torch.zeros(1, 2, 64, 64)

        warped = warp_image(image, flow)

        # With zero flow, output should be very close to input
        assert torch.allclose(warped, image, atol=1e-5)

    def test_upscale_flow(self):
        """Test flow upscaling."""
        flow = torch.randn(1, 2, 64, 64)
        upscaled = upscale_flow(flow, scale_factor=4)

        assert upscaled.shape == (1, 2, 256, 256)
        # Values should be scaled by 4
        assert upscaled.abs().mean() > flow.abs().mean() * 3


class TestDeviceUtils:
    """Test device utility functions."""

    def test_get_device_cpu(self):
        """Test getting CPU device."""
        device = get_device("cpu")
        assert device.type == "cpu"

    def test_get_device_auto(self):
        """Test auto device detection."""
        device = get_device("auto")
        assert device.type in ("cpu", "cuda")

    def test_get_dtype(self):
        """Test dtype parsing."""
        assert get_dtype("float16") == torch.float16
        assert get_dtype("fp16") == torch.float16
        assert get_dtype("bfloat16") == torch.bfloat16
        assert get_dtype("float32") == torch.float32

    def test_get_dtype_invalid(self):
        """Test invalid dtype raises error."""
        with pytest.raises(ValueError):
            get_dtype("invalid_dtype")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
