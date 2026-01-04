"""
State management for Stream-DiffVSR auto-regressive processing.

The state stores information needed to process subsequent frames with
temporal guidance from previous frames.

TENSOR LAYOUT CONVENTION:
All tensors in state use BCHW format (model-native) to avoid
repeated permutations during processing. Conversion to ComfyUI's
BHWC format happens only at node I/O boundaries.

VALUE RANGE CONVENTION:
- previous_hq: BCHW, float32, [-1, 1] (normalized for warping/encode)
- previous_lq_upscaled: BCHW, float32, [0, 1] (for flow computation)
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import torch


@dataclass
class StreamDiffVSRState:
    """
    Auto-regressive state for Stream-DiffVSR.

    Stores information needed to process subsequent frames with
    temporal guidance from previous frames.

    Attributes:
        previous_hq: Previous high-quality output frame.
                     Shape: (1, 3, H*4, W*4), BCHW, float32, [-1,1]
                     Used for warping and ControlNet conditioning.
        previous_lq_upscaled: Previous bicubic 4x upscaled LQ frame.
                              Shape: (1, 3, H*4, W*4), BCHW, float32, [0,1]
                              Used for optical flow estimation.
        frame_index: Frame index in current sequence (0-indexed).
        metadata: Optional metadata (resolution, dtype, etc.)
    """

    # Previous high-quality output frame (BCHW, float32, [-1,1])
    previous_hq: Optional[torch.Tensor] = None

    # Previous bicubic 4x upscaled LQ frame (BCHW, float32, [0,1])
    # Flow is computed between upscaled images, not original LQ
    previous_lq_upscaled: Optional[torch.Tensor] = None

    # Frame index in current sequence (0-indexed)
    frame_index: int = 0

    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_previous(self) -> bool:
        """Check if previous frame data is available for temporal guidance."""
        return self.previous_hq is not None and self.previous_lq_upscaled is not None

    @property
    def is_first_frame(self) -> bool:
        """Check if this is the first frame in the sequence."""
        return self.frame_index == 0 or not self.has_previous

    def reset(self) -> "StreamDiffVSRState":
        """Create a fresh state for a new sequence."""
        return StreamDiffVSRState()

    def clone(self) -> "StreamDiffVSRState":
        """Create a deep copy of this state."""
        return StreamDiffVSRState(
            previous_hq=self.previous_hq.clone() if self.previous_hq is not None else None,
            previous_lq_upscaled=self.previous_lq_upscaled.clone() if self.previous_lq_upscaled is not None else None,
            frame_index=self.frame_index,
            metadata=self.metadata.copy(),
        )

    def to_device(self, device: torch.device) -> "StreamDiffVSRState":
        """Move state tensors to specified device."""
        return StreamDiffVSRState(
            previous_hq=self.previous_hq.to(device) if self.previous_hq is not None else None,
            previous_lq_upscaled=self.previous_lq_upscaled.to(device) if self.previous_lq_upscaled is not None else None,
            frame_index=self.frame_index,
            metadata=self.metadata.copy(),
        )

    def save(self, path: str) -> None:
        """
        Save state to disk for later resumption.

        Uses safetensors format for efficient tensor serialization.
        NEVER converts tensors to Python lists.

        Args:
            path: File path (will create .safetensors file)
        """
        from safetensors.torch import save_file

        if not path.endswith(".safetensors"):
            path = path + ".safetensors"

        tensors = {}
        metadata = {"frame_index": str(self.frame_index)}

        # Add any custom metadata
        for key, value in self.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                metadata[f"meta_{key}"] = str(value)

        if self.previous_hq is not None:
            tensors["previous_hq"] = self.previous_hq.contiguous().cpu()
        if self.previous_lq_upscaled is not None:
            tensors["previous_lq_upscaled"] = self.previous_lq_upscaled.contiguous().cpu()

        # safetensors requires at least one tensor
        if not tensors:
            tensors["_empty"] = torch.zeros(1)

        save_file(tensors, path, metadata=metadata)

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None) -> "StreamDiffVSRState":
        """
        Load state from disk.

        Args:
            path: Path to .safetensors file
            device: Optional device to load tensors to

        Returns:
            Restored StreamDiffVSRState
        """
        from safetensors import safe_open
        from safetensors.torch import load_file

        tensors = load_file(path, device=str(device) if device else "cpu")

        # Read metadata
        with safe_open(path, framework="pt") as f:
            file_metadata = f.metadata() or {}

        frame_index = int(file_metadata.get("frame_index", 0))

        # Reconstruct custom metadata
        metadata = {}
        for key, value in file_metadata.items():
            if key.startswith("meta_"):
                metadata[key[5:]] = value

        # Get tensors (excluding placeholder)
        previous_hq = tensors.get("previous_hq")
        previous_lq_upscaled = tensors.get("previous_lq_upscaled")
        
        # Backwards compatibility: check for old previous_lq key
        if previous_lq_upscaled is None:
            previous_lq_upscaled = tensors.get("previous_lq")

        return cls(
            previous_hq=previous_hq,
            previous_lq_upscaled=previous_lq_upscaled,
            frame_index=frame_index,
            metadata=metadata,
        )

    def __repr__(self) -> str:
        hq_shape = tuple(self.previous_hq.shape) if self.previous_hq is not None else None
        lq_shape = tuple(self.previous_lq_upscaled.shape) if self.previous_lq_upscaled is not None else None
        return (
            f"StreamDiffVSRState("
            f"frame_index={self.frame_index}, "
            f"has_previous={self.has_previous}, "
            f"hq_shape={hq_shape}, "
            f"lq_upscaled_shape={lq_shape})"
        )
