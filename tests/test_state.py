"""
Tests for StreamDiffVSRState.
"""

import pytest
import torch
import tempfile
import os

from stream_diffvsr.state import StreamDiffVSRState


class TestStreamDiffVSRState:
    """Test cases for state management."""

    def test_empty_state_creation(self):
        """Test creating empty state."""
        state = StreamDiffVSRState()

        assert state.previous_hq is None
        assert state.previous_lq is None
        assert state.frame_index == 0
        assert not state.has_previous
        assert state.is_first_frame

    def test_state_with_tensors(self):
        """Test state with tensor data."""
        hq = torch.randn(1, 3, 256, 256)
        lq = torch.randn(1, 3, 64, 64)

        state = StreamDiffVSRState(
            previous_hq=hq,
            previous_lq=lq,
            frame_index=1,
        )

        assert state.has_previous
        assert not state.is_first_frame
        assert state.previous_hq.shape == (1, 3, 256, 256)
        assert state.previous_lq.shape == (1, 3, 64, 64)

    def test_state_clone(self):
        """Test cloning state."""
        original = StreamDiffVSRState(
            previous_hq=torch.randn(1, 3, 256, 256),
            previous_lq=torch.randn(1, 3, 64, 64),
            frame_index=5,
            metadata={"test": "value"},
        )

        cloned = original.clone()

        assert cloned.frame_index == original.frame_index
        assert torch.allclose(cloned.previous_hq, original.previous_hq)
        assert cloned.metadata == original.metadata
        # Verify it's a deep copy
        assert cloned.previous_hq is not original.previous_hq

    def test_state_reset(self):
        """Test resetting state."""
        state = StreamDiffVSRState(
            previous_hq=torch.randn(1, 3, 256, 256),
            frame_index=10,
        )

        new_state = state.reset()

        assert new_state.previous_hq is None
        assert new_state.frame_index == 0

    def test_state_to_device(self):
        """Test moving state to device."""
        state = StreamDiffVSRState(
            previous_hq=torch.randn(1, 3, 256, 256),
            previous_lq=torch.randn(1, 3, 64, 64),
        )

        # Move to CPU (should work regardless of CUDA availability)
        moved = state.to_device(torch.device("cpu"))

        assert moved.previous_hq.device.type == "cpu"
        assert moved.previous_lq.device.type == "cpu"

    def test_state_serialization(self):
        """Test saving and loading state."""
        original = StreamDiffVSRState(
            previous_hq=torch.randn(1, 3, 256, 256),
            previous_lq=torch.randn(1, 3, 64, 64),
            frame_index=42,
            metadata={"key": "value"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.safetensors")
            original.save(path)

            loaded = StreamDiffVSRState.load(path)

            assert loaded.frame_index == original.frame_index
            assert torch.allclose(loaded.previous_hq, original.previous_hq)
            assert torch.allclose(loaded.previous_lq, original.previous_lq)

    def test_state_repr(self):
        """Test state string representation."""
        state = StreamDiffVSRState(
            previous_hq=torch.randn(1, 3, 256, 256),
            frame_index=5,
        )

        repr_str = repr(state)

        assert "frame_index=5" in repr_str
        assert "has_previous=" in repr_str


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
